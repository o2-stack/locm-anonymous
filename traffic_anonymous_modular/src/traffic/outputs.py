import csv
import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src.traffic.trajectory import BasinKnowledgeBase, PhaseSummary
from src.traffic.utils import safe_json_dump

def basin_stage_to_int(stage: str) -> int:
    mapping = {
        "divergence_prone": 1,
        "fragile_suppression": 2,
        "coordination_forming": 3,
        "convergence_building": 4,
        "stability_consolidated": 5,
    }
    return mapping.get(stage, 0)


def phase_summary_to_dict(summary: PhaseSummary) -> dict:
    return {
        "phase_idx": summary.phase_idx,
        "episode_range": summary.episode_range,
        "overview": summary.overview,
        "archetypes": summary.archetypes,
        "boundaries": summary.boundaries,
        "patterns": summary.patterns,
        "comparison": summary.comparison,
        "llm_prompt_package": summary.llm_prompt_package,
    }


def save_phase_metrics_csv(outdir, phase_summaries: List[PhaseSummary], knowledge):
    path = os.path.join(outdir, "phase_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "phase_idx", "episode_range",
            "avg_reward", "avg_delay",
            "avg_counteraction_rate", "avg_dominance_index",
            "avg_role_switch_rate", "avg_control_burden_trend",
            "avg_convergence_quality",
            "basin_stage", "basin_stage_id", "stability_potential",
            "phase_reward_focus"
        ])
        for i, s in enumerate(phase_summaries):
            o = s.overview if s and s.overview else {}
            portrait = knowledge.phase_portraits[i] if (knowledge and i < len(knowledge.phase_portraits)) else {}
            stage = portrait.get("basin_stage", "")
            w.writerow([
                s.phase_idx,
                s.episode_range,
                o.get("avg_reward", ""),
                o.get("avg_delay", ""),
                o.get("avg_counteraction_rate", ""),
                o.get("avg_dominance_index", ""),
                o.get("avg_role_switch_rate", ""),
                o.get("avg_control_burden_trend", ""),
                o.get("avg_convergence_quality", ""),
                stage,
                basin_stage_to_int(stage),
                portrait.get("stability_potential", ""),
                portrait.get("phase_reward_focus", ""),
            ])


def save_phase_summaries_json(outdir, phase_summaries: List[PhaseSummary]):
    path = os.path.join(outdir, "phase_summaries.json")
    data = [phase_summary_to_dict(s) for s in phase_summaries]
    safe_json_dump(data, path)


def save_knowledge_snapshot(outdir, knowledge, round_idx=None):
    if knowledge is None:
        return
    data = {
        "round_idx": round_idx,
        "bad_patterns": knowledge.bad_patterns,
        "good_clues": knowledge.good_clues,
        "uncertainty_areas": knowledge.uncertainty_areas,
        "phase_portraits": knowledge.phase_portraits,
        "maturity": knowledge.maturity,
    }
    safe_json_dump(data, os.path.join(outdir, "knowledge_snapshot.json"))


def load_knowledge_snapshot(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BasinKnowledgeBase.from_dict(data)


def load_phase_archive(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def append_phase_archive(path, round_idx, summary: PhaseSummary, portrait: dict):
    archive = load_phase_archive(path)
    record = {
        "round_idx": round_idx,
        "phase_idx": summary.phase_idx,
        "episode_range": summary.episode_range,
        "overview": summary.overview,
        "patterns": summary.patterns,
        "portrait": portrait if portrait is not None else {},
    }
    archive.append(record)
    safe_json_dump(archive, path)


def plot_basin_stage_evolution(phase_summaries, knowledge, fname):
    if not phase_summaries:
        return
    x, y, labels = [], [], []
    for i, s in enumerate(phase_summaries):
        portrait = knowledge.phase_portraits[i] if (knowledge and i < len(knowledge.phase_portraits)) else {}
        stage = portrait.get("basin_stage", "")
        x.append(i)
        y.append(basin_stage_to_int(stage))
        labels.append(stage)
    if not x:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.yticks([1, 2, 3, 4, 5], [
        "divergence", "fragile_suppr", "coord_forming",
        "conv_building", "stable"
    ])
    plt.xlabel("Phase")
    plt.ylabel("Basin Stage")
    plt.title("Basin Stage Evolution")
    plt.grid(True, alpha=0.3)
    for xi, yi, lab in zip(x, y, labels):
        plt.text(xi, yi + 0.05, lab[:10], fontsize=8, ha="center")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def plot_stability_potential(phase_summaries, knowledge, fname):
    if not phase_summaries:
        return
    vals = []
    for i in range(len(phase_summaries)):
        portrait = knowledge.phase_portraits[i] if (knowledge and i < len(knowledge.phase_portraits)) else {}
        vals.append(float(portrait.get("stability_potential", 0.0)))
    plt.figure(figsize=(8, 5))
    plt.plot(vals, marker="o", linewidth=2, label="stability_potential")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Phase")
    plt.ylabel("Potential")
    plt.title("Stability Potential per Phase")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def plot_phase_behavior_metrics(phase_summaries, fname):
    if not phase_summaries:
        return
    counter, dom, role, burden = [], [], [], []
    for s in phase_summaries:
        o = s.overview if s and s.overview else {}
        counter.append(float(o.get("avg_counteraction_rate", 0.0)))
        dom.append(float(o.get("avg_dominance_index", 0.0)))
        role.append(float(o.get("avg_role_switch_rate", 0.0)))
        burden.append(float(o.get("avg_control_burden_trend", 0.0)))
    x = np.arange(len(phase_summaries))
    plt.figure(figsize=(9, 6))
    plt.plot(x, counter, marker="o", label="counteraction_rate")
    plt.plot(x, dom, marker="s", label="dominance_index")
    plt.plot(x, role, marker="^", label="role_switch_rate")
    plt.plot(x, burden, marker="d", label="control_burden_trend")
    plt.xlabel("Phase")
    plt.ylabel("Value")
    plt.title("Phase Behavior Metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def plot_cognition_maturity(maturity_values, fname):
    if not maturity_values:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(maturity_values, marker="o", linewidth=2)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Round")
    plt.ylabel("Cognition Maturity")
    plt.title("Cognition Maturity Across Rounds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def plot_round_comparison(all_round_results, fname):
    if not all_round_results:
        return
    r50 = [float(np.mean(r["R"][-50:])) for r in all_round_results]
    d50 = [float(np.mean(r["DELAY"][-50:])) for r in all_round_results]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(all_round_results))
    ax1.plot(x, r50, marker="o", color="tab:blue", label="Final 50 Reward")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, d50, marker="s", color="tab:red", label="Final 50 Delay")
    ax2.set_ylabel("Delay", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    plt.title("Round Comparison")
    fig.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def export_round_outputs(outdir, phase_summaries, knowledge, round_idx=None):
    save_phase_metrics_csv(outdir, phase_summaries, knowledge)
    save_phase_summaries_json(outdir, phase_summaries)
    save_knowledge_snapshot(outdir, knowledge, round_idx=round_idx)
    if knowledge is not None and len(knowledge.phase_portraits) >= len(phase_summaries):
        plot_basin_stage_evolution(
            phase_summaries, knowledge,
            os.path.join(outdir, "basin_stage_evolution.png")
        )
        plot_stability_potential(
            phase_summaries, knowledge,
            os.path.join(outdir, "stability_potential.png")
        )
    plot_phase_behavior_metrics(
        phase_summaries,
        os.path.join(outdir, "phase_behavior_metrics.png")
    )


def save_all_agents(agents, save_dir, tag="latest"):
    os.makedirs(save_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        agent.save(os.path.join(save_dir, f"agent{i}_{tag}.pt"))
