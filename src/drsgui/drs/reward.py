"""Composite region quality reward from paper Sec. 3.4."""

from __future__ import annotations

import math
from collections.abc import Sequence

from drsgui.dataset import BBox
from drsgui.drs.config import DRSConfig
from drsgui.drs.geometry import intersection_area, visible_elements
from drsgui.drs.types import RewardBreakdown, UIElement, require_scored


class RegionQualityReward:
    def __init__(self, config: DRSConfig | None = None) -> None:
        self.config = config or DRSConfig()

    def evaluate(
        self, region: BBox, all_elements: Sequence[UIElement]
    ) -> RewardBreakdown:
        if region.area <= 0:
            raise ValueError("reward region must have positive area")
        require_scored(all_elements)
        elements = visible_elements(all_elements, region)
        if not elements:
            return RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0)

        weights = [
            1.0 if element.interactive else self.config.non_interactive_weight
            for element in elements
        ]
        scores = [float(element.relevance) for element in elements]
        relevance = sum(
            weight * score for weight, score in zip(weights, scores, strict=True)
        ) / (sum(weights) + self.config.numerical_epsilon)

        # Cached elements are stored globally. Intersect boxes with the crop to
        # emulate boxes returned by a parser operating on that crop.
        coverage = sum(
            intersection_area(element.bbox, region) for element in elements
        ) / region.area

        logits = [score / self.config.semantic_temperature for score in scores]
        maximum = max(logits)
        exponentials = [math.exp(logit - maximum) for logit in logits]
        partition = sum(exponentials)
        probabilities = [value / partition for value in exponentials]
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0
        )
        concentration = 1.0 - entropy / math.log(
            len(elements) + self.config.numerical_epsilon
        )
        concentration = min(1.0, max(0.0, concentration))

        total = (
            self.config.reward_alpha * relevance
            + self.config.reward_beta * coverage
            + self.config.reward_gamma * concentration
        )
        return RewardBreakdown(
            interaction_weighted_relevance=relevance,
            ui_coverage_consistency=coverage,
            semantic_concentration=concentration,
            total=total,
            element_count=len(elements),
        )
