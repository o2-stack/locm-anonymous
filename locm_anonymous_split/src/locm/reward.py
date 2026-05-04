from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.agents.sac import weights_init_
from src.config import N_AGENTS, device
from src.locm.trajectory import PhaseSummary

@dataclass
class PhaseControl:
    lambda_counter: float = 0.0
    lambda_switch: float = 0.0
    lambda_volatility: float = 0.0
    intensity_scale: float = 1.0
    collective_pull: float = 0.0
    safety_penalty_weight: float = 0.0
    focus: str = "none"


def vectorize_phase_summary(summary: PhaseSummary, baseline_ref: dict = None,
                            prev_control: Optional[PhaseControl] = None) -> np.ndarray:
    o = summary.overview if summary and summary.overview else {}

    avg_vbias = float(o.get("avg_vbias", 0.0))
    avg_ploss = float(o.get("avg_p_loss", 0.0))
    avg_counter = float(o.get("avg_counteraction_rate", 0.0))
    avg_switch = float(o.get("avg_role_switch_rate", 0.0))
    avg_smooth = float(o.get("avg_action_smoothness", 0.0))
    avg_burden = float(o.get("avg_control_burden_trend", 0.0))
    avg_dom = float(o.get("avg_dominance_index", 0.0))
    avg_conv = float(o.get("avg_convergence_quality", 0.0))

    if baseline_ref is None:
        vb_ratio = 1.0
        pl_ratio = 1.0
    else:
        base_vb = float(baseline_ref.get("avg_vbias", max(avg_vbias, 1e-6)))
        base_pl = float(baseline_ref.get("avg_p_loss", max(avg_ploss, 1e-6)))
        vb_ratio = avg_vbias / (base_vb + 1e-6)
        pl_ratio = avg_ploss / (base_pl + 1e-6)

    if prev_control is None:
        prev_vec = [0.0, 0.0, 0.0, 1.0, 0.0]
    else:
        prev_vec = [
            float(prev_control.lambda_counter),
            float(prev_control.lambda_switch),
            float(prev_control.lambda_volatility),
            float(prev_control.intensity_scale),
            float(prev_control.collective_pull),
        ]

    x = np.array([
        avg_counter,
        avg_switch,
        avg_smooth,
        avg_burden,
        avg_dom,
        avg_conv,
        avg_vbias,
        avg_ploss,
        vb_ratio,
        pl_ratio,
        *prev_vec,
    ], dtype=np.float32)
    return x


class PhaseAdapter(nn.Module):
    def __init__(self, in_dim=15, hidden_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_lambda = nn.Linear(hidden_dim, 3)
        self.head_bias = nn.Linear(hidden_dim, 2)
        self.apply(weights_init_)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.backbone(x)
        raw_lambda = self.head_lambda(h)
        raw_bias = self.head_bias(h)

        lambda_counter = 0.60 * torch.sigmoid(raw_lambda[..., 0:1])
        lambda_switch = 0.50 * torch.sigmoid(raw_lambda[..., 1:2])
        lambda_volatility = 0.50 * torch.sigmoid(raw_lambda[..., 2:3])

        intensity_scale = 0.90 + 0.20 * torch.sigmoid(raw_bias[..., 0:1])
        collective_pull = 0.08 * torch.sigmoid(raw_bias[..., 1:2])

        return {
            "lambda_counter": lambda_counter,
            "lambda_switch": lambda_switch,
            "lambda_volatility": lambda_volatility,
            "intensity_scale": intensity_scale,
            "collective_pull": collective_pull,
        }


def portrait_to_soft_target(portrait: dict) -> dict:
    level_map = {
        "off": 0.00,
        "weak": 0.05,
        "medium": 0.10,
        "strong": 0.20,
        "very_strong": 0.35,
    }
    intensity_map = {
        "very_conservative": 0.85,
        "conservative": 0.92,
        "slightly_conservative": 0.97,
        "moderate": 1.00,
        "slightly_aggressive": 1.03,
        "aggressive": 1.08,
    }
    collect_map = {
        "weak": 0.00,
        "medium": 0.02,
        "strong": 0.03,
    }

    portrait = portrait or {}
    dr = portrait.get("direction_relation_preference", {})
    rh = portrait.get("rhythm_preference", {})
    st = portrait.get("phase_action_style", {})
    focus = portrait.get("phase_reward_focus", "none")

    lam_c = level_map.get(dr.get("counteraction_suppression", "medium"), 0.10)
    lam_s = level_map.get(rh.get("role_switch_suppression", "medium"), 0.10)
    lam_v = level_map.get(rh.get("temporal_consistency_encouragement", "medium"), 0.10)

    if focus == "reduce_counteraction":
        lam_c *= 1.5
    elif focus == "stabilize_roles":
        lam_s *= 1.3
        lam_v *= 1.2

    return {
        "lambda_counter": float(lam_c),
        "lambda_switch": float(lam_s),
        "lambda_volatility": float(lam_v),
        "intensity_scale": float(intensity_map.get(st.get("overall_intensity", "moderate"), 1.0)),
        "collective_pull": float(collect_map.get(st.get("collectiveness", "medium"), 0.02)),
        "focus": focus,
    }


def make_phase_control_from_teacher(teacher: dict, safety_penalty_weight: float = 0.0) -> PhaseControl:
    teacher = teacher or {}
    return PhaseControl(
        lambda_counter=float(teacher.get("lambda_counter", 0.0)),
        lambda_switch=float(teacher.get("lambda_switch", 0.0)),
        lambda_volatility=float(teacher.get("lambda_volatility", 0.0)),
        intensity_scale=float(teacher.get("intensity_scale", 1.0)),
        collective_pull=float(teacher.get("collective_pull", 0.0)),
        safety_penalty_weight=float(safety_penalty_weight),
        focus=str(teacher.get("focus", "none")),
    )


class PhaseSafetyDual:
    def __init__(self, init_lambda=0.0, lr=0.05):
        self.lmbda = float(init_lambda)
        self.lr = float(lr)

    def update(self, phase_cost: float) -> float:
        self.lmbda = max(0.0, self.lmbda + self.lr * float(phase_cost))
        return self.lmbda


def compute_phase_cost(summary: PhaseSummary, baseline_ref: dict) -> float:
    if baseline_ref is None:
        return 0.0
    o = summary.overview if summary and summary.overview else {}
    cur_vb = float(o.get("avg_vbias", 0.0))
    cur_pl = float(o.get("avg_p_loss", 0.0))
    base_vb = float(baseline_ref.get("avg_vbias", max(cur_vb, 1e-6)))
    base_pl = float(baseline_ref.get("avg_p_loss", max(cur_pl, 1e-6)))

    vb_excess = max(0.0, cur_vb / (base_vb + 1e-6) - 1.02)
    pl_excess = max(0.0, cur_pl / (base_pl + 1e-6) - 1.05)
    return float(vb_excess + 0.5 * pl_excess)


# ==============================================================================
# Basin Reward Engine
# ==============================================================================
class BasinRewardEngine:
    """
    极简非物理 basin reward：
    - step级：counteraction penalty
    - episode级：role switch penalty + action volatility penalty
    """

    LEVEL_MAP = {
        "off": 0.0,
        "weak": 0.05,
        "medium": 0.10,
        "strong": 0.20,
        "very_strong": 0.35,
    }

    def __init__(self, n_agents=N_AGENTS):
        self.n_agents = n_agents
        self.current_control = None
        self.current_portrait = None

        self.lambda_counter = 0.0
        self.lambda_switch = 0.0
        self.lambda_volatility = 0.0
        self.phase_focus = None

        self.prev_actions = None
        self.action_history = []
        self.dominant_history = []

    def _level(self, text):
        return self.LEVEL_MAP.get(str(text).lower(), 0.0)

    def start_phase(self, control_or_portrait):
        if isinstance(control_or_portrait, PhaseControl):
            self.current_control = control_or_portrait
            self.current_portrait = None
            self.phase_focus = control_or_portrait.focus
            self.lambda_counter = float(control_or_portrait.lambda_counter)
            self.lambda_switch = float(control_or_portrait.lambda_switch)
            self.lambda_volatility = float(control_or_portrait.lambda_volatility)
            return

        portrait = control_or_portrait or {}
        self.current_portrait = portrait
        self.current_control = None
        self.phase_focus = portrait.get("phase_reward_focus", None)

        dr = portrait.get("direction_relation_preference", {})
        rh = portrait.get("rhythm_preference", {})

        self.lambda_counter = self._level(dr.get("counteraction_suppression", "medium"))
        self.lambda_switch = self._level(rh.get("role_switch_suppression", "medium"))
        self.lambda_volatility = self._level(rh.get("temporal_consistency_encouragement", "medium"))

        if self.phase_focus == "reduce_counteraction":
            self.lambda_counter *= 1.5
        elif self.phase_focus == "stabilize_roles":
            self.lambda_switch *= 1.3
            self.lambda_volatility *= 1.2

    def start_episode(self):
        self.prev_actions = None
        self.action_history = []
        self.dominant_history = []

    def step_reward(self, actions: np.ndarray) -> float:
        a = np.asarray(actions, np.float32).reshape(-1)
        abs_a = np.abs(a)
        sum_abs = float(np.sum(abs_a))
        sum_signed = float(np.sum(a))

        if sum_abs < 1e-6:
            counteraction = 0.0
            dominant = -1
        else:
            counteraction = (sum_abs - abs(sum_signed)) / (sum_abs + 1e-6)
            dominant = int(np.argmax(abs_a))

        r_counter = -self.lambda_counter * counteraction

        self.prev_actions = a.copy()
        self.action_history.append(a.copy())
        self.dominant_history.append(dominant)

        return float(r_counter)

    def end_episode_reward(self) -> float:
        r_switch = 0.0
        valid_hist = [x for x in self.dominant_history if x >= 0]
        if len(valid_hist) >= 2:
            switches = 0
            for i in range(1, len(valid_hist)):
                if valid_hist[i] != valid_hist[i - 1]:
                    switches += 1
            role_switch_rate = switches / (len(valid_hist) - 1)
            r_switch = -self.lambda_switch * role_switch_rate

        r_volatility = 0.0
        if len(self.action_history) >= 2:
            actions = np.asarray(self.action_history, np.float32)
            action_diff = float(np.mean(np.abs(actions[1:] - actions[:-1])))
            r_volatility = -self.lambda_volatility * action_diff

        return float(r_switch + r_volatility)

    def current_config(self) -> dict:
        cfg = {
            "phase_focus": self.phase_focus,
            "lambda_counter": self.lambda_counter,
            "lambda_switch": self.lambda_switch,
            "lambda_volatility": self.lambda_volatility,
        }
        if self.current_control is not None:
            cfg["safety_penalty_weight"] = self.current_control.safety_penalty_weight
        return cfg


class PhaseActionBias:
    """
    轻量级 phase 动作风格偏置
    不直接替代actor，只做很小的相位风格修正
    """

    INTENSITY_MAP = {
        "very_conservative": 0.85,
        "conservative": 0.92,
        "slightly_conservative": 0.97,
        "moderate": 1.00,
        "slightly_aggressive": 1.03,
        "aggressive": 1.08,
    }

    COLLECTIVENESS_MAP = {
        "weak": 0.0,
        "medium": 0.02,
        "strong": 0.03,
    }

    def __init__(self):
        self.intensity_scale = 1.0
        self.collective_pull = 0.0

    def start_phase(self, control_or_portrait):
        if isinstance(control_or_portrait, PhaseControl):
            self.intensity_scale = float(control_or_portrait.intensity_scale)
            self.collective_pull = float(control_or_portrait.collective_pull)
            return

        style = (control_or_portrait or {}).get("phase_action_style", {})
        intensity = style.get("overall_intensity", "moderate")
        collectiveness = style.get("collectiveness", "medium")

        self.intensity_scale = self.INTENSITY_MAP.get(intensity, 1.0)
        self.collective_pull = self.COLLECTIVENESS_MAP.get(collectiveness, 0.0)

    def apply(self, actions: np.ndarray) -> np.ndarray:
        a = np.asarray(actions, np.float32).copy()
        a = a * self.intensity_scale
        mean_a = float(np.mean(a))
        a = a + self.collective_pull * (mean_a - a)
        return np.clip(a, -1.0, 1.0)
