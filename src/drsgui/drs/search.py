"""UI perception followed by MCTS dynamic region search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from drsgui.drs.geometry import full_image_region
from drsgui.drs.mcts import MCTSActionPlanner
from drsgui.drs.perception import SemanticScorer, UIElementPerceptor
from drsgui.drs.types import DRSSearchOutput
from drsgui.models.base import ImageInput, load_image


class RegionSearcher(Protocol):
    def search(
        self,
        image: ImageInput,
        instruction: str,
        *,
        application: str | None = None,
        platform: str | None = None,
    ) -> DRSSearchOutput: ...


@dataclass(slots=True)
class DynamicRegionSearcher:
    perceptor: UIElementPerceptor
    semantic_scorer: SemanticScorer
    planner: MCTSActionPlanner

    def search(
        self,
        image: ImageInput,
        instruction: str,
        *,
        application: str | None = None,
        platform: str | None = None,
    ) -> DRSSearchOutput:
        screenshot = load_image(image)
        full_region = full_image_region(*screenshot.size)
        parsed = self.perceptor.parse(screenshot, full_region)
        scored = self.semantic_scorer.score(
            instruction,
            parsed,
            application=application,
            platform=platform,
        )
        result = self.planner.search(full_region=full_region, elements=scored)
        return DRSSearchOutput(
            search=result,
            elements=tuple(scored),
            perceptor=self.perceptor.source_name,
            semantic_scorer=self.semantic_scorer.source_name,
        )
