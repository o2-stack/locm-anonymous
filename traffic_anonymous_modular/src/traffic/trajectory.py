from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.traffic.config import GREEN_SPLIT_INIT, MAX_STEPS, N_AGENTS
from src.traffic.utils import safe_json_dump

class EpisodeTrajectory:
    def __init__(self, episode_id, n_agents=N_AGENTS, max_steps=MAX_STEPS):
        self.episode_id = episode_id
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.actions = np.zeros((max_steps, n_agents), np.int32)        # 离散动作编号
        self.exec_actions = np.zeros((max_steps, n_agents), np.float32) # delta 值
        self.green_split = np.zeros((max_steps, n_agents), np.float32)
        self.delay = np.zeros(max_steps, np.float32)
        self.rewards = np.zeros(max_steps, np.float32)
        self.queue_overflow = np.zeros(max_steps, np.float32)
        self.max_queue_ratio = np.zeros(max_steps, np.float32)
        self.throughput = np.zeros(max_steps, np.float32)
        self.conflict = np.zeros(max_steps, np.float32)
        self.a_sum = np.zeros(max_steps, np.float32)
        self.length = 0
        self.total_reward = 0.0
        self.avg_delay = 0.0
        self.gs_variance = 0.0
        self.avg_overflow = 0.0
        self.features = {}

    def record_step(self, t, actions, exec_deltas, green_split, delay, reward,
                    queue_overflow, max_queue_ratio, throughput, conflict, a_sum):
        self.actions[t] = actions              # int 数组：离散动作编号
        self.exec_actions[t] = exec_deltas     # float 数组：绿信比变化量
        self.green_split[t] = green_split
        self.delay[t] = delay
        self.rewards[t] = reward
        self.queue_overflow[t] = queue_overflow
        self.max_queue_ratio[t] = max_queue_ratio
        self.throughput[t] = throughput
        self.conflict[t] = conflict
        self.a_sum[t] = a_sum
        self.length = t + 1

    def finalize(self, total_reward, avg_delay, gs_variance, avg_overflow):
        self.total_reward = total_reward
        self.avg_delay = avg_delay
        self.gs_variance = gs_variance
        self.avg_overflow = avg_overflow
        self.features = compute_episode_features(self)


def compute_episode_features(traj: EpisodeTrajectory) -> dict:
    L = traj.length
    if L < 4:
        return {}
    actions = traj.exec_actions[:L]
    abs_actions = np.abs(actions)
    half = L // 2
    burden_first = float(np.mean(abs_actions[:half]))
    burden_second = float(np.mean(abs_actions[half:]))
    control_burden_trend = (burden_second - burden_first) / (burden_first + 1e-6)
    delay_first = float(np.mean(traj.delay[:half]))
    delay_second = float(np.mean(traj.delay[half:]))
    convergence_quality = -(delay_second - delay_first) / (delay_first + 1e-6)
    counteraction_steps = 0
    for t in range(L):
        signs = np.sign(actions[t])
        if np.any(signs > 0) and np.any(signs < 0):
            counteraction_steps += 1
    counteraction_rate = counteraction_steps / L
    agent_means = np.mean(abs_actions, axis=0)
    dominance_index = float(np.max(agent_means) / (np.sum(agent_means) + 1e-6))
    if L > 1:
        diffs = np.abs(actions[1:] - actions[:-1])
        action_smoothness = float(np.mean(diffs))
    else:
        action_smoothness = 0.0
    green_split_drift = float(np.max(np.abs(traj.green_split[:L] - GREEN_SPLIT_INIT)))
    q4_start = 3 * L // 4
    late_action_intensity = float(np.mean(abs_actions[q4_start:]))
    dominant_agent_per_step = np.argmax(abs_actions, axis=1)
    if L > 1:
        role_switches = np.sum(dominant_agent_per_step[1:] != dominant_agent_per_step[:-1])
        role_switch_rate = float(role_switches) / (L - 1)
    else:
        role_switch_rate = 0.0
    return {
        "control_burden_trend": round(control_burden_trend, 4),
        "convergence_quality": round(convergence_quality, 4),
        "counteraction_rate": round(counteraction_rate, 4),
        "dominance_index": round(dominance_index, 4),
        "action_smoothness": round(action_smoothness, 4),
        "green_split_drift": round(green_split_drift, 4),
        "late_action_intensity": round(late_action_intensity, 4),
        "role_switch_rate": round(role_switch_rate, 4),
    }


class PhaseTrajectoryRecorder:
    def __init__(self):
        self._current_phase: List[EpisodeTrajectory] = []

    def add_episode(self, traj: EpisodeTrajectory):
        self._current_phase.append(traj)

    def phase_episode_count(self) -> int:
        return len(self._current_phase)

    def get_current_phase(self) -> List[EpisodeTrajectory]:
        return self._current_phase

    def start_new_phase(self):
        self._current_phase = []

    def flush_remaining(self):
        self._current_phase = []


@dataclass
class PhaseSummary:
    phase_idx: int = 0
    episode_range: str = ""
    overview: dict = field(default_factory=dict)
    archetypes: list = field(default_factory=list)
    boundaries: list = field(default_factory=list)
    patterns: dict = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    llm_prompt_package: str = ""


class PhaseSummarizer:
    def summarize(self, episodes: List[EpisodeTrajectory],
                  phase_idx: int = 0,
                  knowledge=None,
                  prev_summary: PhaseSummary = None,
                  search_mode: dict = None,
                  retrieved_cases: list = None,
                  verbose=True) -> PhaseSummary:
        if not episodes:
            return PhaseSummary(phase_idx=phase_idx)
        for ep in episodes:
            if not ep.features:
                ep.features = compute_episode_features(ep)
        overview = self._compute_overview(episodes)
        archetypes = self._select_archetypes(episodes)
        boundaries = self._find_boundaries(episodes)
        patterns = self._find_patterns(episodes)
        comparison = self._compare_with_prev(overview, prev_summary)
        ep_ids = [ep.episode_id for ep in episodes]
        ep_range = f"{min(ep_ids)}-{max(ep_ids)}" if ep_ids else "?"
        llm_pkg = self._format_for_llm(
            phase_idx, ep_range, overview, archetypes,
            boundaries, patterns, comparison,
            knowledge, search_mode, retrieved_cases
        )
        summary = PhaseSummary(
            phase_idx=phase_idx, episode_range=ep_range,
            overview=overview, archetypes=archetypes,
            boundaries=boundaries, patterns=patterns,
            comparison=comparison, llm_prompt_package=llm_pkg,
        )
        if verbose:
            self._print_summary(summary)
        return summary

    def _compute_overview(self, episodes):
        rewards = [ep.total_reward for ep in episodes]
        delays = [ep.avg_delay for ep in episodes]
        overflows = [ep.avg_overflow for ep in episodes]
        gsvars = [ep.gs_variance for ep in episodes]
        features_list = [ep.features for ep in episodes if ep.features]

        max_queues = []
        throughputs = []
        for ep in episodes:
            if ep.length > 0:
                max_queues.append(float(np.mean(ep.max_queue_ratio[:ep.length])))
                throughputs.append(float(np.mean(ep.throughput[:ep.length])))
            else:
                max_queues.append(0.0)
                throughputs.append(0.0)

        def _avg_feature(key):
            vals = [f[key] for f in features_list if key in f]
            return float(np.mean(vals)) if vals else 0.0

        half = len(rewards) // 2
        if half > 0:
            r_trend = "improving" if np.mean(rewards[half:]) > np.mean(rewards[:half]) else "declining"
            d_trend = "improving" if np.mean(delays[half:]) < np.mean(delays[:half]) else "worsening"
        else:
            r_trend = d_trend = "insufficient_data"

        return {
            "n_episodes": len(episodes),
            "avg_reward": round(float(np.mean(rewards)), 4),
            "std_reward": round(float(np.std(rewards)), 4),
            "reward_trend": r_trend,
            "avg_delay": round(float(np.mean(delays)), 4),
            "delay_trend": d_trend,
            "avg_queue_overflow": round(float(np.mean(overflows)), 4),
            "avg_gs_var": round(float(np.mean(gsvars)), 4),
            "avg_max_queue": round(float(np.mean(max_queues)), 4),
            "avg_throughput": round(float(np.mean(throughputs)), 4),
            "avg_control_burden_trend": round(_avg_feature("control_burden_trend"), 4),
            "avg_convergence_quality": round(_avg_feature("convergence_quality"), 4),
            "avg_counteraction_rate": round(_avg_feature("counteraction_rate"), 4),
            "avg_dominance_index": round(_avg_feature("dominance_index"), 4),
            "avg_action_smoothness": round(_avg_feature("action_smoothness"), 4),
            "avg_role_switch_rate": round(_avg_feature("avg_role_switch_rate") if _avg_feature("avg_role_switch_rate") != 0 else _avg_feature("role_switch_rate"), 4),
        }

    def _select_archetypes(self, episodes):
        if len(episodes) < 3:
            return [self._archetype_entry(ep, "only") for ep in episodes]
        sorted_by_r = sorted(episodes, key=lambda e: e.total_reward)
        best = sorted_by_r[-1]
        worst = sorted_by_r[0]
        with_conv = [(ep, ep.features.get("convergence_quality", 0))
                     for ep in episodes if ep.features]
        most_convergent = max(with_conv, key=lambda x: x[1])[0] if with_conv else sorted_by_r[len(sorted_by_r) // 2]
        median_r = float(np.median([ep.total_reward for ep in episodes]))
        boundary_candidates = sorted(
            episodes,
            key=lambda e: abs(e.total_reward - median_r) + abs(e.features.get("control_burden_trend", 0))
        )
        boundary = boundary_candidates[0]
        selected = []
        seen_ids = set()
        for ep, label in [
            (best, "best_performing"),
            (worst, "worst_performing"),
            (most_convergent, "most_convergent"),
            (boundary, "boundary")
        ]:
            if ep.episode_id not in seen_ids:
                selected.append(self._archetype_entry(ep, label))
                seen_ids.add(ep.episode_id)
        return selected

    def _archetype_entry(self, ep: EpisodeTrajectory, label: str) -> dict:
        L = ep.length
        periods = []
        for name, s, e in [("Night 0-6h", 0, 24), ("Morning 6-12h", 24, 48),
                           ("Afternoon 12-18h", 48, 72), ("Evening 18-24h", 72, 96)]:
            e = min(e, L)
            if s >= L:
                break
            seg_actions = ep.exec_actions[s:e]
            seg_delay = ep.delay[s:e]
            periods.append({
                "name": name,
                "avg_action_per_agent": [round(float(np.mean(np.abs(seg_actions[:, i]))), 3)
                                         for i in range(ep.n_agents)],
                "avg_delay": round(float(np.mean(seg_delay)), 4),
                "avg_abs_action": round(float(np.mean(np.abs(seg_actions))), 3),
            })
        return {
            "episode_id": ep.episode_id,
            "label": label,
            "total_reward": round(ep.total_reward, 4),
            "avg_delay": round(ep.avg_delay, 4),
            "features": ep.features,
            "periods": periods,
        }

    def _find_boundaries(self, episodes):
        boundary_episodes = []
        for ep in episodes:
            if not ep.features or ep.length < 10:
                continue
            L = ep.length
            half = L // 2
            abs_a = np.abs(ep.exec_actions[:L])
            burden_first = float(np.mean(abs_a[:half]))
            burden_second = float(np.mean(abs_a[half:]))
            d_first = float(np.mean(ep.delay[:half]))
            d_second = float(np.mean(ep.delay[half:]))
            burden_change = abs(burden_second - burden_first) / (burden_first + 1e-6)
            d_change = abs(d_second - d_first) / (d_first + 1e-6)
            transition_score = burden_change + d_change
            if transition_score > 0.3:
                if burden_second > burden_first and d_second > d_first:
                    boundary_type = "stable_to_congested"
                elif burden_second < burden_first and d_second < d_first:
                    boundary_type = "congested_to_stable"
                elif burden_second > burden_first and d_second <= d_first:
                    boundary_type = "increasing_effort"
                else:
                    boundary_type = "mixed_transition"
                boundary_episodes.append({
                    "episode_id": ep.episode_id,
                    "transition_score": round(transition_score, 3),
                    "boundary_type": boundary_type,
                    "burden_first_half": round(burden_first, 4),
                    "burden_second_half": round(burden_second, 4),
                    "delay_first_half": round(d_first, 4),
                    "delay_second_half": round(d_second, 4),
                })
        boundary_episodes.sort(key=lambda x: x["transition_score"], reverse=True)
        return boundary_episodes[:3]

    def _find_patterns(self, episodes):
        features_list = [ep.features for ep in episodes if ep.features]
        if len(features_list) == 0:
            return {}

        def _rate(key, threshold, above=True):
            vals = [f[key] for f in features_list if key in f]
            if not vals:
                return 0.0
            if above:
                return float(np.mean([v > threshold for v in vals]))
            return float(np.mean([v < threshold for v in vals]))

        bad = []
        counter_rate = _rate("counteraction_rate", 0.3, above=True)
        if counter_rate > 0.3:
            bad.append(f"High counteraction: {counter_rate:.0%} of episodes show >30% counteraction rate")
        burden_up_rate = _rate("control_burden_trend", 0.1, above=True)
        if burden_up_rate > 0.3:
            bad.append(f"Increasing control burden: {burden_up_rate:.0%} of episodes show burden increasing over time")
        dom_rate = _rate("dominance_index", 0.5, above=True)
        if dom_rate > 0.4:
            bad.append(f"Single-intersection dominance: {dom_rate:.0%} of episodes show one agent doing >50% of work")
        switch_rate = _rate("role_switch_rate", 0.3, above=True)
        if switch_rate > 0.3:
            bad.append(f"Frequent role switching: {switch_rate:.0%} of episodes show >30% role switch rate")
        good = []
        conv_rate = _rate("convergence_quality", 0.1, above=True)
        if conv_rate > 0.2:
            good.append(f"Convergent behavior: {conv_rate:.0%} of episodes show improving delay trend")
        burden_down_rate = _rate("control_burden_trend", -0.1, above=False)
        if burden_down_rate > 0.2:
            good.append(f"Decreasing control burden: {burden_down_rate:.0%} of episodes show agents needing less effort over time")
        return {"confirmed_bad": bad, "weak_good": good}

    def _compare_with_prev(self, overview, prev_summary):
        if prev_summary is None or not prev_summary.overview:
            return {"available": False}
        prev = prev_summary.overview
        changes = {}
        for key in ["avg_reward", "avg_delay", "avg_counteraction_rate",
                     "avg_dominance_index", "avg_control_burden_trend"]:
            if key in overview and key in prev:
                diff = overview[key] - prev[key]
                changes[key] = {
                    "current": overview[key],
                    "previous": prev[key],
                    "change": round(diff, 4),
                    "direction": "improved" if (
                        (key == "avg_reward" and diff > 0) or
                        (key != "avg_reward" and diff < 0)
                    ) else "worsened"
                }
        return {"available": True, "changes": changes}

    def _format_for_llm(self, phase_idx, ep_range, overview, archetypes,
                        boundaries, patterns, comparison,
                        knowledge, search_mode, retrieved_cases=None):
        lines = []
        lines.append("=" * 60)
        lines.append(f"PHASE {phase_idx} ANALYSIS REQUEST")
        lines.append(f"Episodes: {ep_range} ({overview['n_episodes']} episodes)")
        lines.append("Domain: Multi-intersection traffic signal control (Discrete Actions)")
        if search_mode:
            lines.append("Search mode: " + ", ".join(f"{k}={v:.0%}" for k, v in search_mode.items()))
        lines.append("=" * 60)

        lines.append("\n--- Phase Overview ---")
        lines.append(f"Avg Reward: {overview['avg_reward']} (trend: {overview['reward_trend']})")
        lines.append(f"Avg Delay: {overview['avg_delay']} (trend: {overview['delay_trend']})")
        lines.append(f"Avg Queue Overflow: {overview.get('avg_queue_overflow', '?')}")
        lines.append(f"Avg Green-split Variance: {overview.get('avg_gs_var', '?')}")
        lines.append(f"Avg Max Queue Ratio: {overview.get('avg_max_queue', '?')}")
        lines.append(f"Avg Throughput: {overview.get('avg_throughput', '?')}")
        lines.append(f"Avg Control Burden Trend: {overview['avg_control_burden_trend']}")
        lines.append(f"Avg Convergence Quality: {overview['avg_convergence_quality']}")
        lines.append(f"Avg Counteraction Rate: {overview['avg_counteraction_rate']}")
        lines.append(f"Avg Dominance Index: {overview['avg_dominance_index']}")
        lines.append(f"Avg Role Switch Rate: {overview['avg_role_switch_rate']}")

        lines.append("\n--- Representative Trajectories ---")
        for arch in archetypes:
            lines.append(f"\n[{arch['label'].upper()}] Episode {arch['episode_id']}")
            lines.append(f"  Reward: {arch['total_reward']}, Delay: {arch['avg_delay']}")
            f = arch["features"]
            lines.append(f"  Burden trend: {f.get('control_burden_trend', '?')}, "
                         f"Convergence: {f.get('convergence_quality', '?')}, "
                         f"Counteraction: {f.get('counteraction_rate', '?')}")

        if boundaries:
            lines.append("\n--- Boundary Episodes (mid-episode transitions) ---")
            for b in boundaries:
                lines.append(f"  Ep {b['episode_id']}: {b['boundary_type']} "
                             f"(score={b['transition_score']})")

        if patterns:
            if patterns.get("confirmed_bad"):
                lines.append("\n--- Recurring BAD Patterns ---")
                for p in patterns["confirmed_bad"]:
                    lines.append(f"  ✗ {p}")
            if patterns.get("weak_good"):
                lines.append("\n--- Weak GOOD Clues ---")
                for p in patterns["weak_good"]:
                    lines.append(f"  ✓ {p}")

        if comparison.get("available"):
            lines.append("\n--- Compared to Previous Phase ---")
            for key, info in comparison.get("changes", {}).items():
                lines.append(f"  {key}: {info['previous']} → {info['current']} "
                             f"({info['direction']}, Δ={info['change']:+.4f})")

        if knowledge and knowledge.get_summary():
            lines.append("\n--- Accumulated Knowledge ---")
            lines.append(knowledge.get_summary())

        if retrieved_cases:
            lines.append("\n--- Retrieved Similar Historical Phases ---")
            for i, case in enumerate(retrieved_cases):
                ov = case.get("overview", {})
                lines.append(
                    f"Case {i + 1}: Round {case.get('round_idx', '?')} "
                    f"Phase {case.get('phase_idx', '?')} "
                    f"(dist={case.get('distance', '?')})"
                )
                lines.append(
                    f"  stage={case.get('basin_stage', '?')}, "
                    f"potential={case.get('stability_potential', '?')}, "
                    f"focus={case.get('phase_reward_focus', '?')}"
                )
                lines.append(
                    f"  counter={ov.get('avg_counteraction_rate', '?')}, "
                    f"dominance={ov.get('avg_dominance_index', '?')}, "
                    f"role_switch={ov.get('avg_role_switch_rate', '?')}, "
                    f"burden={ov.get('avg_control_burden_trend', '?')}, "
                    f"convergence={ov.get('avg_convergence_quality', '?')}"
                )

        lines.append("\n" + "=" * 60)
        lines.append("Output strict JSON in this schema:")
        lines.append("```json")
        lines.append("{")
        lines.append('  "basin_stage": "...",')
        lines.append('  "stability_potential": 0.0,')
        lines.append('  "key_observations": ["...", "..."],')
        lines.append('  "phase_reward_focus": "reduce_counteraction",')
        lines.append('  "direction_relation_preference": {')
        lines.append('    "counteraction_suppression": "very_strong",')
        lines.append('    "alignment_encouragement": "medium",')
        lines.append('    "allow_local_opposition": "weak"')
        lines.append('  },')
        lines.append('  "role_structure_preference": {')
        lines.append('    "dominance_suppression": "medium",')
        lines.append('    "stable_role_encouragement": "strong",')
        lines.append('    "balanced_contribution_encouragement": "medium"')
        lines.append('  },')
        lines.append('  "rhythm_preference": {')
        lines.append('    "role_switch_suppression": "strong",')
        lines.append('    "temporal_consistency_encouragement": "strong",')
        lines.append('    "late_stage_smoothing": "medium"')
        lines.append('  },')
        lines.append('  "burden_evolution_preference": {')
        lines.append('    "burden_increase_suppression": "very_strong",')
        lines.append('    "low_maintenance_encouragement": "strong",')
        lines.append('    "early_to_late_relief_encouragement": "strong"')
        lines.append('  },')
        lines.append('  "contribution_preference": {')
        lines.append('    "shared_participation_encouragement": "medium",')
        lines.append('    "idle_agent_suppression": "medium"')
        lines.append('  },')
        lines.append('  "phase_action_style": {')
        lines.append('    "overall_intensity": "moderate",')
        lines.append('    "collectiveness": "medium"')
        lines.append('  },')
        lines.append('  "good_patterns_to_strengthen": ["...", "..."],')
        lines.append('  "bad_patterns_to_suppress": ["...", "..."],')
        lines.append('  "uncertainty_areas": ["...", "..."],')
        lines.append('  "reasoning": "..."')
        lines.append("}")
        lines.append("```")
        return "\n".join(lines)

    def _print_summary(self, summary: PhaseSummary):
        o = summary.overview
        print(f"\n  [Phase {summary.phase_idx}] Episodes {summary.episode_range}")
        print(f"  Reward: {o.get('avg_reward', '?')} ({o.get('reward_trend', '?')})")
        print(f"  Delay:  {o.get('avg_delay', '?')} ({o.get('delay_trend', '?')})")
        print(f"  Counteraction: {o.get('avg_counteraction_rate', '?')}")
        print(f"  Dominance: {o.get('avg_dominance_index', '?')}")
        print(f"  Burden trend: {o.get('avg_control_burden_trend', '?')}")


# ==============================================================================
# Basin Knowledge Base
# ==============================================================================
class BasinKnowledgeBase:
    def __init__(self):
        self.bad_patterns: List[str] = []
        self.good_clues: List[str] = []
        self.uncertainty_areas: List[str] = []
        self.phase_portraits: List[dict] = []
        self.round_reflections: List[dict] = []
        self.maturity = 0.0

    def update_from_portrait(self, portrait: dict, summary: PhaseSummary):
        if not portrait:
            return
        self.phase_portraits.append(portrait)
        for p in portrait.get("bad_patterns_to_suppress", []):
            if p not in self.bad_patterns:
                self.bad_patterns.append(p)
        self.bad_patterns = self.bad_patterns[-15:]
        for p in portrait.get("good_patterns_to_strengthen", []):
            if p not in self.good_clues:
                self.good_clues.append(p)
        self.good_clues = self.good_clues[-10:]
        self.uncertainty_areas = portrait.get("uncertainty_areas", [])[-5:]
        self._update_maturity()

    def update_round_reflection(self, reflection: dict):
        if not reflection:
            return
        self.round_reflections.append(reflection)
        self.round_reflections = self.round_reflections[-10:]

    def _update_maturity(self):
        n = len(self.phase_portraits)
        if n == 0:
            self.maturity = 0.0
            return
        knowledge_volume = min(1.0, (len(self.bad_patterns) + len(self.good_clues)) / 15.0)
        consistency = 0.5
        if n >= 2:
            last_stage = self.phase_portraits[-1].get("basin_stage", "")
            prev_stage = self.phase_portraits[-2].get("basin_stage", "")
            stages = ["divergence_prone", "fragile_suppression",
                      "coordination_forming", "convergence_building",
                      "stability_consolidated"]
            if last_stage in stages and prev_stage in stages:
                diff = abs(stages.index(last_stage) - stages.index(prev_stage))
                consistency = 1.0 - diff * 0.2
        self.maturity = min(1.0, 0.3 * knowledge_volume + 0.3 * consistency + 0.4 * min(1.0, n / 8.0))

    def get_summary(self) -> str:
        if not self.bad_patterns and not self.good_clues and not self.round_reflections:
            return ""
        lines = []
        if self.bad_patterns:
            lines.append(f"Known bad patterns ({len(self.bad_patterns)}):")
            for p in self.bad_patterns[-5:]:
                lines.append(f"  ✗ {p}")
        if self.good_clues:
            lines.append(f"Known good clues ({len(self.good_clues)}):")
            for p in self.good_clues[-5:]:
                lines.append(f"  ✓ {p}")
        if self.phase_portraits:
            recent = self.phase_portraits[-1]
            lines.append(f"Last basin stage: {recent.get('basin_stage', '?')}")
            lines.append(f"Last stability potential: {recent.get('stability_potential', '?')}")
        if self.round_reflections:
            rr = self.round_reflections[-1]
            lines.append("Last round reflection:")
            lines.append(f"  suspected_bad_basin: {rr.get('suspected_bad_basin', '?')}")
            memo = rr.get("short_memo", "")
            if memo:
                lines.append(f"  memo: {memo}")
            dirs = rr.get("next_round_search_directions", [])
            if dirs:
                for d in dirs[:3]:
                    lines.append(f"  next: {d}")
        lines.append(f"Cognition maturity: {self.maturity:.2f}")
        return "\n".join(lines)

    def get_search_mode(self, phase_idx: int, total_phases: int) -> dict:
        if self.maturity < 0.3:
            return {"exploration": 0.7, "verification": 0.2, "exploitation": 0.1}
        elif self.maturity < 0.6:
            return {"exploration": 0.3, "verification": 0.5, "exploitation": 0.2}
        else:
            return {"exploration": 0.1, "verification": 0.3, "exploitation": 0.6}

    def to_dict(self):
        return {
            "bad_patterns": self.bad_patterns,
            "good_clues": self.good_clues,
            "uncertainty_areas": self.uncertainty_areas,
            "phase_portraits": self.phase_portraits,
            "round_reflections": self.round_reflections,
            "maturity": self.maturity,
        }

    def save(self, path):
        safe_json_dump(self.to_dict(), path)

    @classmethod
    def from_dict(cls, data):
        kb = cls()
        kb.bad_patterns = data.get("bad_patterns", [])
        kb.good_clues = data.get("good_clues", [])
        kb.uncertainty_areas = data.get("uncertainty_areas", [])
        kb.phase_portraits = data.get("phase_portraits", [])
        kb.round_reflections = data.get("round_reflections", [])
        kb.maturity = data.get("maturity", 0.0)
        return kb
