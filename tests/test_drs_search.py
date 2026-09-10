from __future__ import annotations

from PIL import Image

from drsgui.dataset import BBox
from drsgui.drs.mcts import MCTSActionPlanner
from drsgui.drs.perception import CachedUIElementPerceptor, PrecomputedSemanticScorer
from drsgui.drs.search import DynamicRegionSearcher
from drsgui.drs.types import UIElement


def test_searcher_connects_perception_scoring_and_planner() -> None:
    elements = [
        UIElement(
            str(index),
            BBox(100 + index, 100, 105 + index, 110),
            f"Element {index}",
            True,
            1 - index / 100,
        )
        for index in range(20)
    ]
    searcher = DynamicRegionSearcher(
        CachedUIElementPerceptor(elements),
        PrecomputedSemanticScorer(),
        MCTSActionPlanner(),
    )
    output = searcher.search(
        Image.new("RGB", (1000, 600)),
        "Click element",
        application="Test",
        platform="Desktop",
    )
    assert output.perceptor == "cached_ui_elements"
    assert output.semantic_scorer == "precomputed_relevance"
    assert len(output.elements) == 20
    assert output.search.initial_region != output.search.full_region
