from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.traffic.config import ACTION_DELTAS, N_AGENTS, device
from src.traffic.discrete_sac import weights_init_
from src.traffic.trajectory import PhaseSummary

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

    avg_delay = float(o.get("avg_delay", 0.0))
    avg_overflow = float(o.get("avg_queue_overflow", 0.0))
    avg_counter = float(o.get("avg_counteraction_rate", 0.0))
    avg_switch = float(o.get("avg_role_switch_rate", 0.0))
    avg_smooth = float(o.get("avg_action_smoothness", 0.0))
    avg_burden = float(o.get("avg_control_burden_trend", 0.0))
    avg_dom = float(o.get("avg_dominance_index", 0.0))
    avg_conv = float(o.get("avg_convergence_quality", 0.0))

    if baseline_ref is None:
        delay_ratio = 1.0
        overflow_ratio = 1.0
    else:
        base_delay = float(baseline_ref.get("avg_delay", max(avg_delay, 1e-6)))
        base_overflow = float(baseline_ref.get("avg_queue_overflow", max(avg_overflow, 1e-3)))
        delay_ratio = avg_delay / (base_delay + 1e-6)
        overflow_ratio = avg_overflow / (base_overflow + 1e-3)

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
        avg_delay,
        avg_overflow,
        delay_ratio,
        overflow_ratio,
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
    cur_delay = float(o.get("avg_delay", 0.0))
    cur_overflow = float(o.get("avg_queue_overflow", 0.0))

    base_delay = float(baseline_ref.get("avg_delay", max(cur_delay, 1e-6)))
    base_overflow = float(baseline_ref.get("avg_queue_overflow", max(cur_overflow, 1e-3)))

    delay_excess = max(0.0, cur_delay / (base_delay + 1e-6) - 1.02)
    overflow_excess = max(0.0, cur_overflow / (base_overflow + 1e-3) - 1.05)
    return float(delay_excess + 0.5 * overflow_excess)


# ==============================================================================
# Basin Reward Engine
# ==============================================================================
class BasinRewardEngine:
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

        self.current_control = None
        self.current_portrait = control_or_portrait or {}
        self.phase_focus = self.current_portrait.get("phase_reward_focus", None)
        dr = self.current_portrait.get("direction_relation_preference", {})
        rh = self.current_portrait.get("rhythm_preference", {})
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

    def step_reward(self, exec_deltas: np.ndarray) -> float:
        """
        计算单步的盆地奖励
        传入实际执行的 delta 值（如 -0.10, 0, +0.05 等）
        """
        a = np.asarray(exec_deltas, np.float32).reshape(-1)
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


# ==============================================================================
# Phase Action Bias（离散动作版）
# ==============================================================================
class PhaseActionBias:
    """
    离散版 phase 动作风格偏置。
    兼容两种输入：
    1) portrait：直接按规则映射；
    2) PhaseControl：使用 adapter 输出的 intensity_scale / collective_pull。
    """
    INTENSITY_POINTS = [0.85, 0.92, 0.97, 1.00, 1.03, 1.08]
    KEEP_POINTS = [0.25, 0.15, 0.08, 0.00, 0.00, 0.00]
    UPGRADE_POINTS = [0.00, 0.00, 0.00, 0.00, 0.05, 0.10]

    INTENSITY_MAP = {
        "very_conservative": 0.25,
        "conservative": 0.15,
        "slightly_conservative": 0.08,
        "moderate": 0.00,
        "slightly_aggressive": -0.05,
        "aggressive": -0.10,
    }
    COLLECTIVENESS_MAP = {
        "weak": 0.00,
        "medium": 0.10,
        "strong": 0.20,
    }

    def __init__(self):
        self.keep_probability = 0.0
        self.upgrade_probability = 0.0
        self.collective_pull = 0.0

    def _set_from_continuous_control(self, intensity_scale: float, collective_pull: float):
        s = float(np.clip(intensity_scale, self.INTENSITY_POINTS[0], self.INTENSITY_POINTS[-1]))
        self.keep_probability = float(np.interp(s, self.INTENSITY_POINTS, self.KEEP_POINTS))
        self.upgrade_probability = float(np.interp(s, self.INTENSITY_POINTS, self.UPGRADE_POINTS))
        c = float(max(0.0, collective_pull))
        self.collective_pull = float(np.interp(c, [0.0, 0.02, 0.03, 0.08], [0.0, 0.10, 0.20, 0.25]))

    def start_phase(self, control_or_portrait):
        if isinstance(control_or_portrait, PhaseControl):
            self._set_from_continuous_control(
                control_or_portrait.intensity_scale,
                control_or_portrait.collective_pull,
            )
            return

        style = (control_or_portrait or {}).get("phase_action_style", {})
        intensity = style.get("overall_intensity", "moderate")
        collectiveness = style.get("collectiveness", "medium")

        raw = self.INTENSITY_MAP.get(intensity, 0.0)
        if raw > 0:
            self.keep_probability = raw
            self.upgrade_probability = 0.0
        elif raw < 0:
            self.keep_probability = 0.0
            self.upgrade_probability = abs(raw)
        else:
            self.keep_probability = 0.0
            self.upgrade_probability = 0.0

        self.collective_pull = self.COLLECTIVENESS_MAP.get(collectiveness, 0.0)

    def apply(self, actions: np.ndarray) -> np.ndarray:
        """对离散动作施加偏置，输入/输出均为 (n_agents,) int 数组。"""
        a = np.asarray(actions, np.int32).copy()

        if self.keep_probability > 0:
            for i in range(len(a)):
                if a[i] != 0 and np.random.rand() < self.keep_probability:
                    a[i] = 0

        if self.upgrade_probability > 0:
            for i in range(len(a)):
                if np.random.rand() < self.upgrade_probability:
                    if a[i] == 1:
                        a[i] = 3
                    elif a[i] == 2:
                        a[i] = 4

        if self.collective_pull > 0:
            deltas = np.array([ACTION_DELTAS[int(ai)] for ai in a], dtype=np.float32)
            mean_delta = float(np.mean(deltas))
            if abs(mean_delta) > 0.02:
                for i in range(len(a)):
                    if np.random.rand() < self.collective_pull:
                        if mean_delta > 0 and deltas[i] < 0:
                            a[i] = 1
                        elif mean_delta < 0 and deltas[i] > 0:
                            a[i] = 2

        return a
