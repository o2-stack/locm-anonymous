import json
import re
from typing import List, Optional

import numpy as np

from src.locm.knowledge import BasinKnowledgeBase
from src.locm.trajectory import PhaseSummary

class SimpleLLMClient:
    def __init__(self, api_key=None, base_url=None, model="gpt-4o-mini"):
        self.model = model
        self.available = False
        self.call_count = 0
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                self.available = True
                print(f"[LLM] Connected: {model}")
            except Exception as e:
                print(f"[LLM] Init failed: {e}")

    def chat(self, system: str, user: str, temperature=0.5) -> Optional[str]:
        if not self.available:
            return None
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=2500,
            )
            self.call_count += 1
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[LLM] Call failed: {e}")
            return None


class BasinLLMAnalyzer:
    SYSTEM_PROMPT = (
        "You are an expert in complex system dynamics and multi-agent coordination.\n"
        "You analyze behavioral trajectories of a power-grid-like multi-agent system.\n"
        "Do NOT directly turn physical metrics into reward terms.\n"
        "However, you MUST treat physical feasibility and safety as hard constraints.\n"
        "Any recommended coordination basin must remain within acceptable physical bounds.\n"
        "If the current behavior becomes more stable but physical outcomes become clearly worse, "
        "do NOT continue reinforcing that basin.\n"
        "Instead, redirect the next phase toward exploring alternative coordination structures.\n"
        "Use the given JSON schema exactly.\n"
    )
    def __init__(self, llm_client: SimpleLLMClient):
        self.llm = llm_client

    def analyze(self, summary: PhaseSummary,
                knowledge: BasinKnowledgeBase = None,
                search_mode: dict = None,
                human_feedback: str = "") -> dict:
        prompt = summary.llm_prompt_package

        if human_feedback:
            print("\n[LLM] Human feedback received:")
            print(human_feedback.strip())
            prompt += "\n\n--- Human Supervisory Feedback ---\n"
            prompt += human_feedback.strip()
        else:
            print("\n[LLM] No human feedback received.")

        if self.llm and self.llm.available:
            response = self.llm.chat(self.SYSTEM_PROMPT, prompt, temperature=0.4)
            if response:
                portrait = self._parse_response(response)
                if portrait:
                    self._print_portrait(summary.phase_idx, portrait)
                    return portrait

        portrait = self._rule_based_analysis(summary)
        self._print_portrait(summary.phase_idx, portrait)
        return portrait

    def round_review(self, phase_summaries: List[PhaseSummary], knowledge: BasinKnowledgeBase = None) -> dict:
        """
        每轮训练结束后，对这一轮所有phase做一次总复盘
        """
        if not phase_summaries:
            review = {
                "n_phases": 0,
                "round_stage": "unknown",
                "main_problem": "no_phase_data",
                "next_round_focus": "collect_more_evidence",
                "summary": "No phase summaries available."
            }
            print("\n[Round Review]")
            print(json.dumps(review, ensure_ascii=False, indent=2))
            return review

        overviews = [s.overview for s in phase_summaries if s and s.overview]
        if not overviews:
            review = {
                "n_phases": len(phase_summaries),
                "round_stage": "unknown",
                "main_problem": "empty_overview",
                "next_round_focus": "collect_more_evidence",
                "summary": "Phase summaries exist but overview is empty."
            }
            print("\n[Round Review]")
            print(json.dumps(review, ensure_ascii=False, indent=2))
            return review

        avg_reward = float(np.mean([o.get("avg_reward", 0.0) for o in overviews]))
        avg_vbias = float(np.mean([o.get("avg_vbias", 0.0) for o in overviews]))
        avg_counter = float(np.mean([o.get("avg_counteraction_rate", 0.0) for o in overviews]))
        avg_dom = float(np.mean([o.get("avg_dominance_index", 0.0) for o in overviews]))
        avg_switch = float(np.mean([o.get("avg_role_switch_rate", 0.0) for o in overviews]))
        avg_burden = float(np.mean([o.get("avg_control_burden_trend", 0.0) for o in overviews]))
        avg_conv = float(np.mean([o.get("avg_convergence_quality", 0.0) for o in overviews]))

        last_overview = overviews[-1]
        last_reward = float(last_overview.get("avg_reward", 0.0))
        last_vbias = float(last_overview.get("avg_vbias", 0.0))

        first_overview = overviews[0]
        reward_change = last_reward - float(first_overview.get("avg_reward", 0.0))
        vbias_change = last_vbias - float(first_overview.get("avg_vbias", 0.0))

        if avg_counter > 0.45 and avg_burden > 0.10:
            round_stage = "divergence_prone"
            main_problem = "counteraction_and_burden_growth"
            next_focus = "reduce_counteraction"
        elif avg_burden > 0.05 and avg_conv < 0.05:
            round_stage = "fragile_suppression"
            main_problem = "effort_keeps_rising_without_real_convergence"
            next_focus = "reduce_burden_growth"
        elif avg_dom > 0.50:
            round_stage = "imbalanced_coordination"
            main_problem = "single_agent_dominance"
            next_focus = "balance_contribution"
        elif avg_switch > 0.30:
            round_stage = "unstable_roles"
            main_problem = "frequent_role_switching"
            next_focus = "stabilize_roles"
        elif avg_conv > 0.10 and avg_burden < 0.0 and avg_counter < 0.30:
            round_stage = "convergence_building"
            main_problem = "none_major"
            next_focus = "strengthen_emerging_coordination"
        else:
            round_stage = "mixed_transition"
            main_problem = "unclear_mixed_pattern"
            next_focus = "verify_current_basin"

        review = {
            "n_phases": len(phase_summaries),
            "round_stage": round_stage,
            "avg_reward": round(avg_reward, 4),
            "avg_vbias": round(avg_vbias, 4),
            "avg_counteraction_rate": round(avg_counter, 4),
            "avg_dominance_index": round(avg_dom, 4),
            "avg_role_switch_rate": round(avg_switch, 4),
            "avg_control_burden_trend": round(avg_burden, 4),
            "avg_convergence_quality": round(avg_conv, 4),
            "reward_change_first_to_last_phase": round(reward_change, 4),
            "vbias_change_first_to_last_phase": round(vbias_change, 4),
            "main_problem": main_problem,
            "next_round_focus": next_focus,
            "knowledge_maturity": round(float(knowledge.maturity), 4) if knowledge else None,
            "summary": (
                f"This round contains {len(phase_summaries)} phases. "
                f"Average counteraction={avg_counter:.3f}, dominance={avg_dom:.3f}, "
                f"role_switch={avg_switch:.3f}, burden_trend={avg_burden:.3f}, "
                f"convergence_quality={avg_conv:.3f}. "
                f"Recommended next-round focus: {next_focus}."
            )
        }

        print("\n[Round Review]")
        print(json.dumps(review, ensure_ascii=False, indent=2))
        return review
    def reflect_round(self, phase_summaries: List[PhaseSummary], knowledge: BasinKnowledgeBase = None) -> dict:
        """
        每轮结束后的高层反思：
        - 为什么这一轮会差
        - 可能陷入了什么错误盆地
        - 下一轮该怎么搜索更好盆地
        """
        if not phase_summaries:
            reflection = {
                "main_failure_modes": ["no_phase_data"],
                "suspected_bad_basin": "unknown",
                "what_not_to_repeat": ["insufficient_data"],
                "next_round_search_directions": ["collect_more_evidence"],
                "search_bias_update": {
                    "exploration": 0.6,
                    "verification": 0.3,
                    "exploitation": 0.1
                },
                "short_memo": "No phase summaries available."
            }
            print("\n[Round Reflection]")
            print(json.dumps(reflection, ensure_ascii=False, indent=2))
            return reflection

        overviews = [s.overview for s in phase_summaries if s and s.overview]
        if not overviews:
            reflection = {
                "main_failure_modes": ["empty_overview"],
                "suspected_bad_basin": "unknown",
                "what_not_to_repeat": ["insufficient_summary_quality"],
                "next_round_search_directions": ["improve_phase_analysis"],
                "search_bias_update": {
                    "exploration": 0.5,
                    "verification": 0.4,
                    "exploitation": 0.1
                },
                "short_memo": "Phase summaries exist but overviews are empty."
            }
            print("\n[Round Reflection]")
            print(json.dumps(reflection, ensure_ascii=False, indent=2))
            return reflection

        avg_counter = float(np.mean([o.get("avg_counteraction_rate", 0.0) for o in overviews]))
        avg_dom = float(np.mean([o.get("avg_dominance_index", 0.0) for o in overviews]))
        avg_switch = float(np.mean([o.get("avg_role_switch_rate", 0.0) for o in overviews]))
        avg_burden = float(np.mean([o.get("avg_control_burden_trend", 0.0) for o in overviews]))
        avg_conv = float(np.mean([o.get("avg_convergence_quality", 0.0) for o in overviews]))
        avg_vbias = float(np.mean([o.get("avg_vbias", 0.0) for o in overviews]))

        failure_modes = []
        what_not_to_repeat = []
        next_directions = []

        # 1) 高 counteraction
        if avg_counter > 0.5:
            failure_modes.append("persistent_high_counteraction")
            what_not_to_repeat.append("do not keep reinforcing coordination patterns with strong action opposition")
            next_directions.append("search for coordination modes with lower unnecessary opposition")

        # 2) 高 role switch
        if avg_switch > 0.35:
            failure_modes.append("unstable_role_structure")
            what_not_to_repeat.append("avoid rapidly changing dominant roles without clear benefit")
            next_directions.append("encourage more stable role allocation across the episode")

        # 3) 高 dominance
        if avg_dom > 0.5:
            failure_modes.append("over_dominant_single_agent_behavior")
            what_not_to_repeat.append("do not over-concentrate control burden on a single agent")
            next_directions.append("explore more distributed yet still effective coordination")

        # 4) burden 持续增长
        if avg_burden > 0.1:
            failure_modes.append("burden_growth_over_time")
            what_not_to_repeat.append("avoid basins that require increasing maintenance in late stages")
            next_directions.append("search for basins with earlier stabilization and lower late-stage effort")

        # 5) convergence 很差
        if avg_conv < 0:
            failure_modes.append("lack_of_true_convergence")
            what_not_to_repeat.append("do not confuse superficial behavioral regularity with actual convergence")
            next_directions.append("test alternative basins instead of consolidating the current one")

        # 6) 物理结果差（这里只作为反思，不进入reward）
        if avg_vbias > 0.4:
            failure_modes.append("behaviorally_structured_but_physically_weak")
            what_not_to_repeat.append("do not continue strengthening basins that look coordinated but correlate with poor physical outcomes")
            next_directions.append("explore alternative coordination structures that may align better with physical regulation")

        if not failure_modes:
            failure_modes.append("no_major_failure_mode_detected")
            next_directions.append("continue verifying and refining the current basin")

        # 判断疑似错误盆地类型
        if avg_counter > 0.5 and avg_switch > 0.35:
            bad_basin = "chaotic_conflict_basin"
        elif avg_vbias > 0.4 and avg_counter < 0.4:
            bad_basin = "stable_but_physically_misaligned_basin"
        elif avg_dom > 0.5:
            bad_basin = "single_agent_dominance_basin"
        elif avg_burden > 0.1:
            bad_basin = "high_maintenance_basin"
        else:
            bad_basin = "uncertain_or_mixed_basin"

        # 搜索偏置更新
        if "behaviorally_structured_but_physically_weak" in failure_modes:
            search_bias = {
                "exploration": 0.6,
                "verification": 0.3,
                "exploitation": 0.1
            }
        elif len(failure_modes) >= 3:
            search_bias = {
                "exploration": 0.7,
                "verification": 0.2,
                "exploitation": 0.1
            }
        else:
            search_bias = {
                "exploration": 0.4,
                "verification": 0.4,
                "exploitation": 0.2
            }

        reflection = {
            "main_failure_modes": failure_modes,
            "suspected_bad_basin": bad_basin,
            "what_not_to_repeat": what_not_to_repeat[:5],
            "next_round_search_directions": next_directions[:5],
            "search_bias_update": search_bias,
            "short_memo": (
                f"This round likely converged toward {bad_basin}. "
                f"Main issues: {', '.join(failure_modes[:3])}. "
                f"Next round should emphasize: {', '.join(next_directions[:2])}."
            )
        }

        print("\n[Round Reflection]")
        print(json.dumps(reflection, ensure_ascii=False, indent=2))
        return reflection

    def _parse_response(self, response: str) -> dict:
        try:
            m = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if m:
                return json.loads(m.group(1))
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
        return {}

    def _rule_based_analysis(self, summary: PhaseSummary) -> dict:
        o = summary.overview
        counter = o.get("avg_counteraction_rate", 0.5)
        burden = o.get("avg_control_burden_trend", 0)
        conv = o.get("avg_convergence_quality", 0)
        dom = o.get("avg_dominance_index", 0.5)
        switch = o.get("avg_role_switch_rate", 0.3)

        avg_vbias = float(o.get("avg_vbias", 0.0))
        avg_vmax = float(o.get("avg_v_max", 1.0))
        avg_vmin = float(o.get("avg_v_min", 1.0))
        avg_ploss = float(o.get("avg_p_loss", 0.0))

        if burden > 0.2 and counter > 0.4:
            stage = "divergence_prone"
            potential = 0.1
        elif burden > 0 and conv < 0:
            stage = "fragile_suppression"
            potential = 0.2
        elif counter < 0.35 and conv > 0:
            stage = "coordination_forming"
            potential = 0.45
        elif burden < -0.05 and conv > 0.1:
            stage = "convergence_building"
            potential = 0.65
        else:
            stage = "fragile_suppression"
            potential = 0.25

        physically_unacceptable = (
            avg_vbias > 0.35 or
            avg_vmax > 1.05 or
            avg_vmin < 0.95
        )

        focus = "reduce_counteraction"
        if physically_unacceptable:
            focus = "explore_alternative_basin"
        elif burden > 0.15:
            focus = "reduce_burden_growth"
        elif switch > 0.3:
            focus = "stabilize_roles"
        elif dom > 0.5:
            focus = "balance_contribution"

        def level_from_value(v, low=0.2, med=0.4, high=0.6):
            if v >= high:
                return "very_strong"
            elif v >= med:
                return "strong"
            elif v >= low:
                return "medium"
            elif v > 0:
                return "weak"
            return "off"

        portrait = {
            "basin_stage": stage,
            "stability_potential": round(potential, 2),
                        "key_observations": [
                f"counteraction_rate={counter:.3f}",
                f"control_burden_trend={burden:.3f}",
                f"dominance_index={dom:.3f}",
                f"role_switch_rate={switch:.3f}",
                f"avg_vbias={avg_vbias:.3f}",
                f"avg_vmax={avg_vmax:.3f}",
                f"avg_vmin={avg_vmin:.3f}",
                f"avg_p_loss={avg_ploss:.3f}",
            ],
            "phase_reward_focus": focus,
            "direction_relation_preference": {
                "counteraction_suppression": level_from_value(counter, 0.15, 0.3, 0.5),
                "alignment_encouragement": "medium" if counter > 0.3 else "strong",
                "allow_local_opposition": "weak",
            },
            "role_structure_preference": {
                "dominance_suppression": level_from_value(dom, 0.3, 0.45, 0.55),
                "stable_role_encouragement": "strong" if switch > 0.25 else "medium",
                "balanced_contribution_encouragement": "medium",
            },
            "rhythm_preference": {
                "role_switch_suppression": level_from_value(switch, 0.15, 0.25, 0.35),
                "temporal_consistency_encouragement": "strong",
                "late_stage_smoothing": "medium",
            },
                        "burden_evolution_preference": {
                "burden_increase_suppression": level_from_value(max(0.0, burden), 0.05, 0.12, 0.2),
                "low_maintenance_encouragement": "strong" if burden > 0 else "medium",
                "early_to_late_relief_encouragement": "strong",
            },
            "contribution_preference": {
                "shared_participation_encouragement": "medium" if dom > 0.45 else "strong",
                "idle_agent_suppression": "medium" if dom > 0.45 else "weak",
            },
            "phase_action_style": {
                "overall_intensity": "moderate" if burden > 0 else "slightly_conservative",
                "collectiveness": "strong" if counter > 0.4 else "medium",
            },
            "good_patterns_to_strengthen": [
                "more stable roles over time",
                "lower action opposition across agents",
            ],
            "bad_patterns_to_suppress": [
                "high counteraction",
                "growing control burden",
            ],
            "uncertainty_areas": ["whether recent convergence is genuine or only superficial suppression"],
            "reasoning": "Rule-based fallback portrait generated from behavioral phase summary."
        }
        return portrait

    def _print_portrait(self, phase_idx, portrait):
        print(f"\n  [Basin Portrait - Phase {phase_idx}]")
        print(f"  Stage: {portrait.get('basin_stage', '?')}")
        print(f"  Stability potential: {portrait.get('stability_potential', '?')}")
        print(f"  Focus: {portrait.get('phase_reward_focus', '?')}")
        if portrait.get("good_patterns_to_strengthen"):
            for p in portrait["good_patterns_to_strengthen"][:2]:
                print(f"  ✓ {p}")
        if portrait.get("bad_patterns_to_suppress"):
            for p in portrait["bad_patterns_to_suppress"][:2]:
                print(f"  ✗ {p}")
