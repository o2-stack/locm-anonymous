import numpy as np

from src.traffic.outputs import load_phase_archive
from src.traffic.trajectory import PhaseSummary

class ExperienceRetriever:
    FEATURE_KEYS = [
        "avg_counteraction_rate",
        "avg_dominance_index",
        "avg_role_switch_rate",
        "avg_control_burden_trend",
        "avg_convergence_quality",
    ]

    def __init__(self, archive_path):
        self.archive_path = archive_path

    def _feature_vector_from_overview(self, overview: dict):
        return np.array([float(overview.get(k, 0.0)) for k in self.FEATURE_KEYS], dtype=np.float32)

    def retrieve_similar_cases(self, current_summary: PhaseSummary, top_k=5):
        archive = load_phase_archive(self.archive_path)
        if not archive:
            return []
        cur_vec = self._feature_vector_from_overview(current_summary.overview)
        candidates = []
        for rec in archive:
            ov = rec.get("overview", {})
            hist_vec = self._feature_vector_from_overview(ov)
            dist = float(np.linalg.norm(cur_vec - hist_vec))
            portrait = rec.get("portrait", {})
            candidates.append({
                "round_idx": rec.get("round_idx", -1),
                "phase_idx": rec.get("phase_idx", -1),
                "episode_range": rec.get("episode_range", ""),
                "distance": round(dist, 4),
                "basin_stage": portrait.get("basin_stage", ""),
                "stability_potential": portrait.get("stability_potential", 0.0),
                "phase_reward_focus": portrait.get("phase_reward_focus", ""),
                "overview": ov,
                "patterns": rec.get("patterns", {}),
            })
        candidates.sort(key=lambda x: x["distance"])
        return candidates[:top_k]
