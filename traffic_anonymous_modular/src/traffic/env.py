from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.traffic.config import (
    ACTION_DELTAS, AGENT_INTERSECTIONS, GREEN_SPLIT_INIT, GREEN_SPLIT_MAX,
    GREEN_SPLIT_MIN, MAX_QUEUE, MAX_STEPS, N_ACTIONS, N_AGENTS,
)

class HardSettingsLite:
    obs_noise_std: float = 0.02
    act_delay: int = 0
    exec_noise_std: float = 0.06
    exec_dropout_p: float = 0.10
    reward_t_df: float = 3.0
    reward_t_scale: float = 0.04
    drift_per_ep: float = 0.018
    curriculum: bool = True
    max_difficulty_amp: float = 0.9

    w_delay: float = 0.6
    w_green_penalty: float = 0.4
    green_penalty_scale: float = 30.0

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
            max_difficulty_amp=0.6, w_delay=0.5, w_green_penalty=0.5,
            green_penalty_scale=20.0,
            act_delay=0, exec_dropout_p=0.02, reward_t_df=4.0, reward_t_scale=0.015
        )
    if p == "HARD":
        return HardSettingsLite(
            obs_noise_std=0.02, exec_noise_std=0.08, drift_per_ep=0.02,
            max_difficulty_amp=1.0, w_delay=0.7, w_green_penalty=0.3,
            green_penalty_scale=40.0,
            act_delay=0, exec_dropout_p=0.08, reward_t_df=2.5, reward_t_scale=0.03
        )
    if p in ("STATIONARY", "STABLE", "NO_NOISE"):
        return HardSettingsLite(
            obs_noise_std=0.0, act_delay=0, exec_noise_std=0.0, exec_dropout_p=0.0,
            reward_t_df=3.0, reward_t_scale=0.0, drift_per_ep=0.0, curriculum=False,
            max_difficulty_amp=0.0, w_delay=0.6, w_green_penalty=0.4,
            green_penalty_scale=30.0,
        )
    return HardSettingsLite()


# ==============================================================================
# 交通仿真器
# ==============================================================================
class TrafficGridSim:
    def __init__(self, n_intersections=N_AGENTS,
                 max_queue=MAX_QUEUE,
                 saturation_rate=28.0,
                 coupling_rate=0.6):
        self.n = n_intersections
        self.max_queue = max_queue
        self.sat_rate = saturation_rate
        self.coupling = coupling_rate
        self.queues = np.zeros((n_intersections, 2), dtype=np.float32)
        self.last_avg_delay = 0.0
        self.last_max_queue_ratio = 0.0
        self.last_min_throughput = 1.0
        self.last_total_overflow = 0.0
        self.last_total_throughput = 0.0

    def reset(self):
        self.queues = np.zeros((self.n, 2), dtype=np.float32)

    def step(self, demand_ns, demand_ew, green_splits):
        green_splits = np.clip(np.asarray(green_splits, np.float32),
                               GREEN_SPLIT_MIN, GREEN_SPLIT_MAX)
        base_arrival = 15.0
        arr_ns = float(demand_ns) * base_arrival
        arr_ew = float(demand_ew) * base_arrival

        total_throughput = 0.0
        total_overflow = 0.0
        delays = []

        for i in range(self.n):
            ns_arrival = arr_ns + np.random.uniform(-1.0, 1.0)
            ns_capacity = self.sat_rate * green_splits[i]
            ns_total = self.queues[i, 0] + max(0, ns_arrival)
            ns_served = min(ns_total, ns_capacity)
            self.queues[i, 0] = max(0.0, ns_total - ns_served)

            ew_arrival = arr_ew + np.random.uniform(-1.0, 1.0)
            if i > 0:
                upstream_ew_served = min(
                    self.sat_rate * (1.0 - green_splits[i - 1]),
                    15.0
                )
                ew_arrival += upstream_ew_served * self.coupling
            ew_capacity = self.sat_rate * (1.0 - green_splits[i])
            ew_total = self.queues[i, 1] + max(0, ew_arrival)
            ew_served = min(ew_total, ew_capacity)
            self.queues[i, 1] = max(0.0, ew_total - ew_served)

            for d in range(2):
                if self.queues[i, d] > self.max_queue:
                    total_overflow += (self.queues[i, d] - self.max_queue)
                    self.queues[i, d] = self.max_queue

            d_ns = self.queues[i, 0] / (ns_capacity + 1e-6)
            d_ew = self.queues[i, 1] / (ew_capacity + 1e-6)
            delays.append((d_ns + d_ew) / 2.0)

            total_throughput += ns_served + ew_served

        raw_delay = float(np.mean(delays))
        avg_delay = float(raw_delay / 5.0)

        self.last_avg_delay = avg_delay
        queue_ratios = self.queues / self.max_queue
        self.last_max_queue_ratio = float(np.max(queue_ratios))
        self.last_min_throughput = float(np.min([
            total_throughput / (self.n * 2 * self.sat_rate * 0.5 + 1e-6)
        ]))
        self.last_total_overflow = float(total_overflow)
        self.last_total_throughput = float(total_throughput)

        return avg_delay


# ==============================================================================
# 多智能体交通环境（离散动作版）
# ==============================================================================
class MultiAgentTrafficEnvLite:
    def __init__(self, n_agents=N_AGENTS, hard: HardSettingsLite = HardSettingsLite()):
        assert n_agents == len(AGENT_INTERSECTIONS)
        self.n = n_agents
        self.hard = hard
        self.MAX_STEPS = MAX_STEPS
        self.progress = 0.0

        try:
            self.demand_NS, self.demand_EW = self._load_real_traffic_data("Metro_Interstate_Traffic_Volume.csv")
            print("[环境信息] 成功加载真实世界交通数据集！")
        except Exception as e:
            print(f"[环境信息] 加载真实数据失败 ({e})，回退到合成数据。")
            self.demand_NS = self._generate_synthetic_profile(0.85, 0.55, 0.20, 8, 18)
            self.demand_EW = self._generate_synthetic_profile(0.50, 0.80, 0.18, 7.5, 17.5)

        self.grid = TrafficGridSim(n_intersections=n_agents)
        self._act_buf = deque(maxlen=max(1, self.hard.act_delay + 1))

        self.delay_trace = []
        self.green_split_trace = [[] for _ in range(self.n)]
        self.overflow_trace = []
        self.max_queue_trace = []
        self.throughput_trace = []
        self.reset()

    def _load_real_traffic_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df.set_index('date_time', inplace=True)
        day_data = df.loc[(df.index >= '2018-05-15') & (df.index < '2018-05-16'), 'traffic_volume']
        day_data = day_data[~day_data.index.duplicated(keep='first')]
        day_data_15min = day_data.resample('15min').interpolate(method='linear')
        real_flow = day_data_15min.values[:self.MAX_STEPS]
        if len(real_flow) < self.MAX_STEPS:
            pad = np.zeros(self.MAX_STEPS - len(real_flow))
            real_flow = np.concatenate([real_flow, pad])
        max_vol = np.max(real_flow)
        min_vol = np.min(real_flow)
        demand_ns = 0.15 + (real_flow - min_vol) / (max_vol - min_vol + 1e-6) * 1.05
        demand_ew = np.roll(demand_ns, shift=4) * 0.8 + np.random.uniform(-0.05, 0.05, size=self.MAX_STEPS)
        demand_ew = np.clip(demand_ew, 0.1, 1.2)
        return demand_ns.astype(np.float32), demand_ew.astype(np.float32)

    @staticmethod
    def _generate_synthetic_profile(morning_peak, evening_peak, base, morning_hour, evening_hour):
        t = np.linspace(0, 24, 96)
        morning = morning_peak * np.exp(-0.5 * ((t - morning_hour) / 1.5) ** 2)
        evening = evening_peak * np.exp(-0.5 * ((t - evening_hour) / 2.0) ** 2)
        lunch = 0.15 * np.exp(-0.5 * ((t - 12.5) / 1.0) ** 2)
        profile = base + morning + evening + lunch
        return np.clip(profile, 0.05, 1.0).astype(np.float32)

    def set_progress(self, progress: float):
        self.progress = float(np.clip(progress, 0.0, 1.0))

    def _actor_obs_i(self, i: int) -> np.ndarray:
        idx = min(self.t, len(self.demand_NS_ep) - 1)
        base = np.array([
            self.t / float(self.MAX_STEPS),
            self.green_split[i],
            self.grid.queues[i, 0] / MAX_QUEUE,
            self.grid.queues[i, 1] / MAX_QUEUE,
            (float(self.demand_NS_ep[idx]) + float(self.demand_EW_ep[idx])) / 2.0
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
            float(np.mean(self.green_split)),
            float(np.std(self.green_split) + 1e-8)
        ], np.float32)
        return np.concatenate([base, extra], axis=0)

    def reset(self):
        drift = 1.0 + np.random.uniform(-self.hard.drift_per_ep, self.hard.drift_per_ep)
        self.demand_NS_ep = np.clip(self.demand_NS * drift, 0.0, 1.5)
        self.demand_EW_ep = np.clip(self.demand_EW * drift, 0.0, 1.5)
        self.t = 0
        self.green_split = np.full(self.n, GREEN_SPLIT_INIT, dtype=np.float32)
        self.a_sum_last = 0.0
        self.conflict_last = 0.0
        self.grid.reset()
        self.delay_trace = []
        self.green_split_trace = [[] for _ in range(self.n)]
        self.overflow_trace = []
        self.max_queue_trace = []
        self.throughput_trace = []
        self._act_buf.clear()
        for _ in range(max(1, self.hard.act_delay)):
            self._act_buf.append(np.zeros(self.n, np.int32))
        obsA = np.stack([self._actor_obs_i(i) for i in range(self.n)], 0)
        obsC = np.stack([self._critic_obs_i(i) for i in range(self.n)], 0)
        return obsA, obsC

    def _apply_delay_and_dropout(self, actions: np.ndarray) -> np.ndarray:
        """
        对离散动作施加延迟、噪声和丢包
        输入:  actions (n_agents,) int 数组，值 0~4
        输出:  exec_a (n_agents,) int 数组，实际执行的动作
        """
        self._act_buf.append(np.asarray(actions, np.int32).reshape(-1))
        exec_a = self._act_buf[0].copy()
        amp = self.hard.scale_with_progress(self.progress)

        # 执行噪声：以一定概率随机替换为其他动作
        if self.hard.exec_noise_std > 0:
            noise_prob = self.hard.exec_noise_std * amp
            for i in range(self.n):
                if np.random.rand() < noise_prob:
                    exec_a[i] = np.random.randint(0, N_ACTIONS)

        # 执行丢包：以一定概率强制替换为"保持不变"(动作0)
        if self.hard.exec_dropout_p > 0:
            for i in range(self.n):
                if np.random.rand() < self.hard.exec_dropout_p:
                    exec_a[i] = 0

        return exec_a

    def step(self, actions: np.ndarray):
        """
        执行一步
        输入 actions 是整数数组 (n_agents,)，每个值 0~4
        通过 ACTION_DELTAS 查表映射为 green_split 的变化量
        """
        actions = np.asarray(actions, np.int32).reshape(-1)
        exec_a = self._apply_delay_and_dropout(actions)

        # 离散动作 → 查表 → 绿信比变化
        deltas = np.array([ACTION_DELTAS[int(a)] for a in exec_a], dtype=np.float32)
        self.green_split = np.clip(
            self.green_split + deltas,
            GREEN_SPLIT_MIN, GREEN_SPLIT_MAX
        )

        idx = min(self.t, len(self.demand_NS_ep) - 1, len(self.demand_EW_ep) - 1)
        d_ns = self.demand_NS_ep[idx]
        d_ew = self.demand_EW_ep[idx]

        avg_delay = self.grid.step(d_ns, d_ew, self.green_split)
        queue_overflow = self.grid.last_total_overflow
        max_queue_ratio = self.grid.last_max_queue_ratio
        throughput = self.grid.last_total_throughput

        terminated = (self.t == self.MAX_STEPS - 1)

        penalty_green = 0.0
        if terminated:
            deviation = np.abs(self.green_split - 0.5)
            extreme = np.maximum(0.0, deviation - 0.25)
            penalty_green = float(np.mean(extreme) * self.hard.green_penalty_scale)

        # 【修改点 2】：增加真实的物理硬约束惩罚
        # 车辆溢出（超出容量导致消失）是严重事故，直接与溢出数量挂钩
        overflow_penalty = float(queue_overflow * 5.0)
        # 当排队占用率超过 85% 时，给予高压警告惩罚，防止逼近极限
        queue_warning = float(max(0.0, max_queue_ratio - 0.85) * 50.0)

        # 将所有物理惩罚全部汇入基础环境奖励
        r_clean = -(self.hard.w_delay * avg_delay +
                    self.hard.w_green_penalty * penalty_green +
                    overflow_penalty +
                    queue_warning)

        if self.hard.reward_t_scale > 0:
            amp = self.hard.scale_with_progress(self.progress)
            t_noise = np.random.standard_t(
                df=max(2.1, self.hard.reward_t_df)
            ) * (self.hard.reward_t_scale * amp)
            r = float(r_clean + t_noise)
        else:
            r = float(r_clean)

        self.delay_trace.append(avg_delay)
        for i in range(self.n):
            self.green_split_trace[i].append(float(self.green_split[i]))
        self.overflow_trace.append(queue_overflow)
        self.max_queue_trace.append(max_queue_ratio)
        self.throughput_trace.append(throughput)

        # 用 deltas 计算 a_sum 和 conflict
        self.a_sum_last = float(np.sum(deltas))
        self.conflict_last = float(
            np.mean(np.abs(deltas)) - abs(self.a_sum_last) / max(1, self.n)
        )
        self.t += 1

        obsA = np.stack([self._actor_obs_i(i) for i in range(self.n)], 0)
        obsC = np.stack([self._critic_obs_i(i) for i in range(self.n)], 0)

        info = {
            "avg_delay": avg_delay,
            "penalty_green": penalty_green,
            "a_sum": self.a_sum_last,
            "conflict": self.conflict_last,
            "queue_overflow": queue_overflow,
            "max_queue_ratio": max_queue_ratio,
            "throughput": throughput,
            "env_reward": r,
            "exec_actions": exec_a.copy(),       # int 数组：实际执行的离散动作
            "exec_deltas": deltas.copy(),         # float 数组：对应的绿信比变化量
        }
        return (obsA, obsC), r, bool(terminated), info

    def episode_metrics(self):
        avg_delay = float(np.mean(self.delay_trace)) if self.delay_trace else 0.0
        gs_var = float(np.mean([
            np.var(tr) if len(tr) > 1 else 0.0
            for tr in self.green_split_trace
        ]))
        overflow_avg = float(np.mean(self.overflow_trace)) if self.overflow_trace else 0.0
        max_q_avg = float(np.mean(self.max_queue_trace)) if self.max_queue_trace else 0.0
        throughput_avg = float(np.mean(self.throughput_trace)) if self.throughput_trace else 0.0
        return avg_delay, gs_var, overflow_avg, max_q_avg, throughput_avg
