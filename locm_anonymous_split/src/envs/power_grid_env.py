from collections import deque
from dataclasses import dataclass

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from src.config import (
    A_SCALE,
    AGENT_BUSES,
    MAX_STEPS,
    N_AGENTS,
    SOC_CAP,
    SOC_LIMIT,
)

class HardSettingsLite:
    obs_noise_std: float = 0.02
    act_delay: int = 1
    exec_noise_std: float = 0.06
    exec_dropout_p: float = 0.10
    reward_t_df: float = 3.0
    reward_t_scale: float = 0.04
    drift_per_ep: float = 0.018
    curriculum: bool = True
    max_difficulty_amp: float = 0.9

    # 第一轮 baseline 用的物理奖励
    w_vbias: float = 0.6
    w_soc: float = 0.4
    soc_penalty_scale: float = 40.0

    def scale_with_progress(self, progress: float):
        if not self.curriculum:
            return 1.0
        p = float(np.clip(progress, 0.0, 1.0))
        ease = 0.2 + 0.8 * (p ** 1.5)
        return min(1.0, ease) * self.max_difficulty_amp


def make_preset(preset: str = "MEDIUM") -> HardSettingsLite:
    p = preset.upper()
    if p == "EASY":
        return HardSettingsLite(
            obs_noise_std=0.01, exec_noise_std=0.04, drift_per_ep=0.01,
            max_difficulty_amp=0.6, w_vbias=0.5, w_soc=0.5, soc_penalty_scale=30.0,
            act_delay=1, exec_dropout_p=0.02, reward_t_df=4.0, reward_t_scale=0.015
        )
    if p == "HARD":
        return HardSettingsLite(
            obs_noise_std=0.02, exec_noise_std=0.08, drift_per_ep=0.02,
            max_difficulty_amp=1.0, w_vbias=0.7, w_soc=0.3, soc_penalty_scale=50.0,
            act_delay=1, exec_dropout_p=0.08, reward_t_df=2.5, reward_t_scale=0.03
        )
    if p in ("STATIONARY", "STABLE", "NO_NOISE"):
        return HardSettingsLite(
            obs_noise_std=0.0, act_delay=0, exec_noise_std=0.0, exec_dropout_p=0.0,
            reward_t_df=3.0, reward_t_scale=0.0, drift_per_ep=0.0, curriculum=False,
            max_difficulty_amp=0.0, w_vbias=0.6, w_soc=0.4, soc_penalty_scale=40.0,
        )
    return HardSettingsLite()


class GridSim33BWLite:
    def __init__(self, pv_buses, wt_buses, batt_buses, cap=0.4):
        self.net = pn.case33bw()
        self.cap = cap
        self.base_p = self.net.load.p_mw.values.copy()
        self.base_q = self.net.load.q_mvar.values.copy()
        self.pv_idx = [pp.create_sgen(self.net, b, p_mw=0.0, q_mvar=0.0) for b in pv_buses]
        self.wt_idx = [pp.create_sgen(self.net, b, p_mw=0.0, q_mvar=0.0) for b in wt_buses]
        self.batt_idx = [pp.create_storage(self.net, b, p_mw=0.0, max_e_mwh=30.0, soc_percent=50.0) for b in batt_buses]
        self.net.max_vm_pu = 1.2
        self.net.min_vm_pu = 0.8
        self._pf_opts = dict(recycle=True, calculate_voltage_angles=False, enforce_q_lims=False)
        self.last_p_loss = 0.0
        self.last_v_max = 1.0
        self.last_v_min = 1.0

    def step(self, L_scale, PV_scale, WT_scale, batt_p_list):
        self.net.load.p_mw = self.base_p * float(L_scale)
        self.net.load.q_mvar = self.base_q * float(L_scale)
        pv_p = float(PV_scale) * self.cap
        wt_p = float(WT_scale) * self.cap

        for i in self.pv_idx:
            self.net.sgen.at[i, "p_mw"] = pv_p
        for i in self.wt_idx:
            self.net.sgen.at[i, "p_mw"] = wt_p
        for k, idx in enumerate(self.batt_idx):
            self.net.storage.at[idx, "p_mw"] = float(batt_p_list[k])

        try:
            pp.runpp(self.net, **self._pf_opts)
            vm = self.net.res_bus.vm_pu.values
            vbias = float(np.mean(np.minimum(1.0, np.abs(1.0 - vm) / 0.07)))
            self.last_p_loss = float(self.net.res_line.pl_mw.sum()) if hasattr(self.net.res_line, "pl_mw") else 0.0
            self.last_v_max = float(np.max(vm))
            self.last_v_min = float(np.min(vm))
        except Exception:
            vbias = 1.0
            self.last_p_loss = 0.0
            self.last_v_max = 1.0
            self.last_v_min = 1.0

        return vbias


class MultiAgentPowerGridEnvLite:
    def __init__(self, n_agents=N_AGENTS, hard: HardSettingsLite = HardSettingsLite()):
        assert n_agents == len(AGENT_BUSES)
        self.n = n_agents
        self.hard = hard
        self.MAX_STEPS = MAX_STEPS
        self.progress = 0.0

        self.WT_raw = [7.795145639, 30, 35, 8.846969231, 16.46602734, 18.13731014, 0, 22.51716352, 0, 17.9231927,
                       14.15372591, 0, 11.6233221, 4, 35.35145853, 32, 49, 14.33133905, 19.00167218, 50, 30.48483598,
                       36.18525123, 21.04862466, 58.55085761, 7, 13.38712664, 0.9, 3.056863165, 32, 9.720436673, 17,
                       2.491215512, 15.4306888, 13, 23.60703705, 50, 50, 66, 29.75268875, 19.94024882, 65.28505115,
                       62.98944949, 62.75321569, 25.41053571, 95.88858417, 86.48719301, 75.17048763, 51.40892866,
                       96.45673321, 98.95653492, 58.69891281, 57.17791032, 98, 97.33019532, 96.52678304, 72.53758377,
                       62.91378467, 89.78739585, 69.47327471, 68.24764212, 72.57372903, 100, 99.7924927, 87.26289182,
                       73.78566709, 93, 89, 100, 78.12244919, 100, 99, 91.5508517, 75.62210048, 100, 71.04460315,
                       83.93701513, 86.05545333, 86.1450672, 88.33546098, 74.33299008, 89.54168021, 89.98797207,
                       80.2885353, 100, 96.13676328, 61.66369873, 80.96525581, 86, 40.71576154, 63.2995258, 32.5655823,
                       56.56493033, 46.10773804, 11.23771407, 0, 21.58711212]
        self.PV_raw = [0, 0, 0, 1.534893928, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 13.23842331, 18, 17, 12, 6.294037376,
                       30, 33.7784774, 40.12795867, 41.45072669, 12.87897941, 20, 48.62089279, 45.56991566, 37.72844958,
                       55.03924065, 57.27395133, 77.1460199, 31.53641078, 82.4481725, 54.45656023, 36.45647462, 92,
                       80.02727971, 66.11332888, 50.72330911, 74.05187436, 52.62290957, 57.07280675, 17.50162697,
                       25.76591962, 27.25622161, 45.01228529, 23.71170186, 44.56052686, 14.11180894, 43.54698389,
                       8.261743441, 40.91542105, 51.00898113, 18.82405249, 31.85294629, 31.10056238, 42.19804925,
                       11.14644011, 41.20804216, 34.76803192, 20.388283, 21.32341377, 7.464497931, 7.823391371,
                       3.058413701, 0, 0, 12, 0, 8.458569964, 0, 0, 0, 2.663710742, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0]
        self.L_raw = [38.52680745, 48, 52.83431338, 12.70716651, 41.09933268, 32.368543, 51.33887579, 49.72165616,
                      10.96586938, 47.17449319, 10.1811827, 28.39494669, 17.52102577, 21.49084683, 50.26449404,
                      16.33932982, 35.66857677, -1.122451028, 35.27626195, 25.33153535, 16.57206761, 48.81216394,
                      1.06183333, 45.06336446, 38.92842238, 48.35322762, 9.118658638, 34.59910003, 1.290028902,
                      54.91118903, 1.506844108, 18.66403543, 36.01146046, 42.94016099, 44.48765596, 17.51547332,
                      30.24421748, 30, 52.28523656, 43.90483901, 82.03554115, 44.69683367, 79.31542833, 43.51056931,
                      83.19949454, 85.16741374, 99.51924713, 55.10400638, 69.72695304, 75.10100365, 70.58174233, 100,
                      71.56391619, 73.54736895, 90.84908664, 99.5, 98.40613288, 51.87624311, 40.5054955, 98.63365864,
                      26.89196698, 64.37961463, 50.75045943, 65.50168208, 17.1951119, 57.76211918, 22.28244356,
                      29.79231207, 89.71000347, 93.12346184, 71.6642721, 42.21021461, 100, 100, 86.32109956,
                      74.83646558, 83.61669166, 100, 64.05377977, 98.3, 79.35033602, 84.60830134, 100, 100, 82.97375079,
                      81.63787632, 61.00385313, 99.6, 80.94409778, 74.55806944, 54.95072907, 83.05763655, 59.12786812,
                      62.72374535, 43.65248929, 10.09138061]
        self.PV = np.asarray(self.PV_raw, np.float32) / 100.0
        self.WT = np.asarray(self.WT_raw, np.float32) / 100.0
        self.L = np.asarray(self.L_raw, np.float32) / 100.0

        self.grid = GridSim33BWLite(
            pv_buses=(9, 13, 23, 24, 22, 32, 31, 12),
            wt_buses=(16, 17, 19, 20, 21, 26, 27),
            batt_buses=AGENT_BUSES, cap=0.4
        )
        self._act_buf = deque(maxlen=max(1, self.hard.act_delay + 1))
        self.p_loss_trace = []
        self.v_max_trace = []
        self.v_min_trace = []
        self.reset()

    def set_progress(self, progress: float):
        self.progress = float(np.clip(progress, 0.0, 1.0))

    def _actor_obs_i(self, i: int) -> np.ndarray:
        idx = min(self.t, len(self.L_ep) - 1, len(self.PV_ep) - 1, len(self.WT_ep) - 1)
        base = np.array([
            self.t / float(self.MAX_STEPS),
            self.soc[i],
            float(self.L_ep[idx]),
            float(self.PV_ep[idx]),
            float(self.WT_ep[idx])
        ], np.float32)
        amp = self.hard.scale_with_progress(self.progress)
        if self.hard.obs_noise_std > 0:
            noise = np.zeros_like(base)
            noise[1:] = np.random.normal(0.0, self.hard.obs_noise_std * amp, size=4)
            base = base + noise
        return base

    def _critic_obs_i(self, i: int) -> np.ndarray:
        base = self._actor_obs_i(i)
        extra = np.array([
            self.a_sum_last,
            self.conflict_last,
            float(np.mean(self.soc)),
            float(np.std(self.soc) + 1e-8)
        ], np.float32)
        return np.concatenate([base, extra], axis=0)

    def reset(self):
        # 1. 生成平滑波动的分布式噪声（累积的高斯随机游走）
        # 我们将每一步的波动率设为最大幅度的 1/10，保证曲线平滑
        step_noise_pv = np.random.normal(0, self.hard.drift_per_ep / 10, size=self.PV.shape)
        step_noise_wt = np.random.normal(0, self.hard.drift_per_ep / 10, size=self.WT.shape)
        step_noise_l  = np.random.normal(0, self.hard.drift_per_ep / 10, size=self.L.shape)

        # 2. 累积每一步的增量，并加上基础值 1.0
        drift_pv = 1.0 + np.cumsum(step_noise_pv)
        drift_wt = 1.0 + np.cumsum(step_noise_wt)
        drift_l  = 1.0 + np.cumsum(step_noise_l)

        # 3. 严格限制波动边界，确保它不会超出 EASY/HARD 模式设定的最大幅度
        drift_pv = np.clip(drift_pv, 1.0 - self.hard.drift_per_ep, 1.0 + self.hard.drift_per_ep)
        drift_wt = np.clip(drift_wt, 1.0 - self.hard.drift_per_ep, 1.0 + self.hard.drift_per_ep)
        drift_l  = np.clip(drift_l,  1.0 - self.hard.drift_per_ep, 1.0 + self.hard.drift_per_ep)

        # 4. 数组与数组逐元素相乘，此时每个时间步的乘数都不一样了
        self.PV_ep = np.clip(self.PV * drift_pv, 0.0, 1.5)
        self.WT_ep = np.clip(self.WT * drift_wt, 0.0, 1.5)
        self.L_ep  = np.clip(self.L * drift_l, 0.0, 1.5)

        self.t = 0
        self.soc = np.zeros(self.n, np.float32)
        self.a_sum_last = 0.0
        # ... 后面的代码完全保持不变 ...
        self.conflict_last = 0.0
        self.vbias_trace = []
        self.soc_trace = [[] for _ in range(self.n)]
        self._act_buf.clear()
        for _ in range(max(1, self.hard.act_delay)):
            self._act_buf.append(np.zeros(self.n, np.float32))
        self.p_loss_trace = []
        self.v_max_trace = []
        self.v_min_trace = []
        obsA = np.stack([self._actor_obs_i(i) for i in range(self.n)], 0)
        obsC = np.stack([self._critic_obs_i(i) for i in range(self.n)], 0)
        return obsA, obsC

    def _apply_delay_and_dropout(self, actions: np.ndarray) -> np.ndarray:
        self._act_buf.append(np.asarray(actions, np.float32).reshape(-1))
        exec_a = self._act_buf[0].copy()
        amp = self.hard.scale_with_progress(self.progress)
        if self.hard.exec_noise_std > 0:
            exec_a = (exec_a + np.random.normal(0.0, self.hard.exec_noise_std * amp, size=exec_a.shape)).clip(-1.0, 1.0)
        if self.hard.exec_dropout_p > 0:
            mask = (np.random.rand(self.n) >= self.hard.exec_dropout_p).astype(np.float32)
            exec_a = exec_a * mask
        return exec_a

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, np.float32).reshape(-1).clip(-1.0, 1.0)
        exec_a = self._apply_delay_and_dropout(actions)
        self.soc = self.soc + (exec_a * A_SCALE) * 0.25 / SOC_CAP
        idx = min(self.t, len(self.L_ep) - 1, len(self.PV_ep) - 1, len(self.WT_ep) - 1)

        pv, wt, load = self.PV_ep[idx], self.WT_ep[idx], self.L_ep[idx]
        vbias = self.grid.step(load, pv, wt, (exec_a * A_SCALE).tolist())
        p_loss = self.grid.last_p_loss
        v_max = self.grid.last_v_max
        v_min = self.grid.last_v_min

        terminated = (self.t == self.MAX_STEPS - 1)
        penalty_soc = 0.0
        if terminated:
            over = np.maximum(0.0, np.abs(self.soc) - SOC_LIMIT)
            penalty_soc = float(np.mean(over) * self.hard.soc_penalty_scale)

        # baseline参考物理奖励（仅第一轮训练用；后续轮只记录）
        r_clean = -(self.hard.w_vbias * vbias + self.hard.w_soc * penalty_soc)

        if self.hard.reward_t_scale > 0:
            amp = self.hard.scale_with_progress(self.progress)
            t_noise = np.random.standard_t(df=max(2.1, self.hard.reward_t_df)) * (
                self.hard.reward_t_scale * amp
            )
            r = float(r_clean + t_noise)
        else:
            r = float(r_clean)

        self.vbias_trace.append(vbias)
        for i in range(self.n):
            self.soc_trace[i].append(float(self.soc[i]))
        self.p_loss_trace.append(p_loss)
        self.v_max_trace.append(v_max)
        self.v_min_trace.append(v_min)
        self.a_sum_last = float(np.sum(exec_a))
        self.conflict_last = float(np.mean(np.abs(exec_a)) - abs(self.a_sum_last) / max(1, self.n))
        self.t += 1
        obsA = np.stack([self._actor_obs_i(i) for i in range(self.n)], 0)
        obsC = np.stack([self._critic_obs_i(i) for i in range(self.n)], 0)
        info = {
            "vbias": vbias,
            "penalty_soc": penalty_soc,
            "a_sum": self.a_sum_last,
            "conflict": self.conflict_last,
            "p_loss": p_loss,
            "v_max": v_max,
            "v_min": v_min,
            "env_reward": r,        # 物理奖励，后续轮仅记录
            "exec_actions": exec_a.copy(),
        }
        return (obsA, obsC), r, bool(terminated), info
    def episode_metrics(self):
        vb = float(np.mean(self.vbias_trace)) if self.vbias_trace else 0.0
        sv = float(np.mean([np.var(tr) if len(tr) > 1 else 0.0 for tr in self.soc_trace]))
        p_loss_avg = float(np.mean(self.p_loss_trace)) if self.p_loss_trace else 0.0
        v_max_avg = float(np.mean(self.v_max_trace)) if self.v_max_trace else 1.0
        v_min_avg = float(np.mean(self.v_min_trace)) if self.v_min_trace else 1.0
        return vb, sv, p_loss_avg, v_max_avg, v_min_avg
