from typing import List

from src.utils import safe_json_dump

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
