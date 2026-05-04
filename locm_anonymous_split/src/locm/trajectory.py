from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.config import MAX_STEPS, N_AGENTS

class EpisodeTrajectory:
    def __init__(self, episode_id, n_agents=N_AGENTS, max_steps=MAX_STEPS):
        self.episode_id = episode_id
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.actions = np.zeros((max_steps, n_agents), np.float32)
        self.exec_actions = np.zeros((max_steps, n_agents), np.float32)
        self.soc = np.zeros((max_steps, n_agents), np.float32)
        self.vbias = np.zeros(max_steps, np.float32)
        self.rewards = np.zeros(max_steps, np.float32)
        self.p_loss = np.zeros(max_steps, np.float32)
        self.v_max = np.zeros(max_steps, np.float32)
        self.v_min = np.zeros(max_steps, np.float32)
        self.conflict = np.zeros(max_steps, np.float32)
        self.a_sum = np.zeros(max_steps, np.float32)
        self.length = 0
        self.total_reward = 0.0
        self.avg_vbias = 0.0
        self.soc_variance = 0.0
        self.avg_p_loss = 0.0
        self.features = {}

    def record_step(self, t, actions, exec_actions, soc, vbias, reward,
                    p_loss, v_max, v_min, conflict, a_sum):
        self.actions[t] = actions
        self.exec_actions[t] = exec_actions
        self.soc[t] = soc
        self.vbias[t] = vbias
        self.rewards[t] = reward
        self.p_loss[t] = p_loss
        self.v_max[t] = v_max
        self.v_min[t] = v_min
        self.conflict[t] = conflict
        self.a_sum[t] = a_sum
        self.length = t + 1

    def finalize(self, total_reward, avg_vbias, soc_variance, avg_p_loss):
        self.total_reward = total_reward
        self.avg_vbias = avg_vbias
        self.soc_variance = soc_variance
        self.avg_p_loss = avg_p_loss
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

    vb_first = float(np.mean(traj.vbias[:half]))
    vb_second = float(np.mean(traj.vbias[half:]))
    convergence_quality = -(vb_second - vb_first) / (vb_first + 1e-6)

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

    soc_drift = float(np.max(np.abs(traj.soc[:L])))
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
        "soc_drift": round(soc_drift, 4),
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
            phase_idx=phase_idx,
            episode_range=ep_range,
            overview=overview,
            archetypes=archetypes,
            boundaries=boundaries,
            patterns=patterns,
            comparison=comparison,
            llm_prompt_package=llm_pkg,
        )
        if verbose:
            self._print_summary(summary)
        return summary

    def _compute_overview(self, episodes):
        rewards = [ep.total_reward for ep in episodes]
        vbiases = [ep.avg_vbias for ep in episodes]
        plosses = [ep.avg_p_loss for ep in episodes]
        socvars = [ep.soc_variance for ep in episodes]
        features_list = [ep.features for ep in episodes if ep.features]

        # 从轨迹里提取 v_max / v_min 的平均
        vmaxes = []
        vmins = []
        for ep in episodes:
            if ep.length > 0:
                vmaxes.append(float(np.mean(ep.v_max[:ep.length])))
                vmins.append(float(np.mean(ep.v_min[:ep.length])))
            else:
                vmaxes.append(1.0)
                vmins.append(1.0)
        def _avg_feature(key):
            vals = [f[key] for f in features_list if key in f]
            return float(np.mean(vals)) if vals else 0.0

        half = len(rewards) // 2
        if half > 0:
            r_trend = "improving" if np.mean(rewards[half:]) > np.mean(rewards[:half]) else "declining"
            vb_trend = "improving" if np.mean(vbiases[half:]) < np.mean(vbiases[:half]) else "worsening"
        else:
            r_trend = vb_trend = "insufficient_data"

        return {
            "n_episodes": len(episodes),
            "avg_reward": round(float(np.mean(rewards)), 4),
            "std_reward": round(float(np.std(rewards)), 4),
            "reward_trend": r_trend,
            "avg_vbias": round(float(np.mean(vbiases)), 4),
            "vbias_trend": vb_trend,
            "avg_p_loss": round(float(np.mean(plosses)), 4),
            "avg_socvar": round(float(np.mean(socvars)), 4),
            "avg_v_max": round(float(np.mean(vmaxes)), 4),
            "avg_v_min": round(float(np.mean(vmins)), 4),
            "avg_control_burden_trend": round(_avg_feature("control_burden_trend"), 4),
            "avg_convergence_quality": round(_avg_feature("convergence_quality"), 4),
            "avg_counteraction_rate": round(_avg_feature("counteraction_rate"), 4),
            "avg_dominance_index": round(_avg_feature("dominance_index"), 4),
            "avg_action_smoothness": round(_avg_feature("action_smoothness"), 4),
            "avg_role_switch_rate": round(_avg_feature("role_switch_rate"), 4),
        }

    def _select_archetypes(self, episodes):
        if len(episodes) < 3:
            return [self._archetype_entry(ep, "only") for ep in episodes]

        sorted_by_r = sorted(episodes, key=lambda e: e.total_reward)
        best = sorted_by_r[-1]
        worst = sorted_by_r[0]

        with_conv = [(ep, ep.features.get("convergence_quality", 0)) for ep in episodes if ep.features]
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
            seg_vbias = ep.vbias[s:e]
            periods.append({
                "name": name,
                "avg_action_per_agent": [round(float(np.mean(np.abs(seg_actions[:, i]))), 3)
                                         for i in range(ep.n_agents)],
                "avg_vbias": round(float(np.mean(seg_vbias)), 4),
                "avg_abs_action": round(float(np.mean(np.abs(seg_actions))), 3),
            })

        return {
            "episode_id": ep.episode_id,
            "label": label,
            "total_reward": round(ep.total_reward, 4),
            "avg_vbias": round(ep.avg_vbias, 4),
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
            vb_first = float(np.mean(ep.vbias[:half]))
            vb_second = float(np.mean(ep.vbias[half:]))

            burden_change = abs(burden_second - burden_first) / (burden_first + 1e-6)
            vb_change = abs(vb_second - vb_first) / (vb_first + 1e-6)
            transition_score = burden_change + vb_change

            if transition_score > 0.3:
                boundary_type = "unknown"
                if burden_second > burden_first and vb_second > vb_first:
                    boundary_type = "stable_to_unstable"
                elif burden_second < burden_first and vb_second < vb_first:
                    boundary_type = "unstable_to_stable"
                elif burden_second > burden_first and vb_second <= vb_first:
                    boundary_type = "increasing_effort"
                else:
                    boundary_type = "mixed_transition"

                boundary_episodes.append({
                    "episode_id": ep.episode_id,
                    "transition_score": round(transition_score, 3),
                    "boundary_type": boundary_type,
                    "burden_first_half": round(burden_first, 4),
                    "burden_second_half": round(burden_second, 4),
                    "vbias_first_half": round(vb_first, 4),
                    "vbias_second_half": round(vb_second, 4),
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
            bad.append(f"Single-agent dominance: {dom_rate:.0%} of episodes show one agent doing >50% of work")

        switch_rate = _rate("role_switch_rate", 0.3, above=True)
        if switch_rate > 0.3:
            bad.append(f"Frequent role switching: {switch_rate:.0%} of episodes show >30% role switch rate")

        good = []
        conv_rate = _rate("convergence_quality", 0.1, above=True)
        if conv_rate > 0.2:
            good.append(f"Convergent behavior: {conv_rate:.0%} of episodes show improving vbias trend")

        burden_down_rate = _rate("control_burden_trend", -0.1, above=False)
        if burden_down_rate > 0.2:
            good.append(f"Decreasing control burden: {burden_down_rate:.0%} of episodes show agents needing less effort over time")

        return {
            "confirmed_bad": bad,
            "weak_good": good,
        }

    def _compare_with_prev(self, overview, prev_summary):
        if prev_summary is None or not prev_summary.overview:
            return {"available": False}
        prev = prev_summary.overview
        changes = {}
        for key in ["avg_reward", "avg_vbias", "avg_counteraction_rate",
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
        if search_mode:
            lines.append("Search mode: " + ", ".join(f"{k}={v:.0%}" for k, v in search_mode.items()))
        lines.append("=" * 60)

        lines.append("\n--- Phase Overview ---")
        lines.append(f"Avg Reward: {overview['avg_reward']} (trend: {overview['reward_trend']})")
        lines.append(f"Avg VBias: {overview['avg_vbias']} (trend: {overview['vbias_trend']})")
        lines.append(f"Avg Power Loss: {overview.get('avg_p_loss', '?')}")
        lines.append(f"Avg SOC Variance: {overview.get('avg_socvar', '?')}")
        lines.append(f"Avg Vmax: {overview.get('avg_v_max', '?')}")
        lines.append(f"Avg Vmin: {overview.get('avg_v_min', '?')}")
        lines.append(f"Avg Control Burden Trend: {overview['avg_control_burden_trend']}")
        lines.append(f"Avg Convergence Quality: {overview['avg_convergence_quality']}")
        lines.append(f"Avg Counteraction Rate: {overview['avg_counteraction_rate']}")
        lines.append(f"Avg Dominance Index: {overview['avg_dominance_index']}")
        lines.append(f"Avg Role Switch Rate: {overview['avg_role_switch_rate']}")
        lines.append("\n--- Representative Trajectories ---")
        for arch in archetypes:
            lines.append(f"\n[{arch['label'].upper()}] Episode {arch['episode_id']}")
            lines.append(f"  Reward: {arch['total_reward']}, VBias: {arch['avg_vbias']}")
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
        print(f"  VBias:  {o.get('avg_vbias', '?')} ({o.get('vbias_trend', '?')})")
        print(f"  Counteraction: {o.get('avg_counteraction_rate', '?')}")
        print(f"  Dominance: {o.get('avg_dominance_index', '?')}")
        print(f"  Burden trend: {o.get('avg_control_burden_trend', '?')}")
