from __future__ import annotations

from pathlib import Path

from PIL import Image

from drsgui.dataset import BBox
from drsgui.drs.mcts import MCTSActionPlanner
from drsgui.drs.perception import CachedUIElementPerceptor, PrecomputedSemanticScorer
from drsgui.drs.search import DynamicRegionSearcher
from drsgui.drs.types import UIElement
from drsgui.drs.visualization import render_search_visualization


def test_visualization_preserves_original_image_size(tmp_path: Path) -> None:
    image = Image.new("RGB", (200, 100), color="white")
    elements = [
        UIElement(
            str(index),
            BBox(20 + index, 20, 25 + index, 30),
            f"Element {index}",
            True,
            1 - index / 100,
        )
        for index in range(20)
    ]
    output = DynamicRegionSearcher(
        CachedUIElementPerceptor(elements),
        PrecomputedSemanticScorer(),
        MCTSActionPlanner(),
    ).search(image, "Click element")
    destination = tmp_path / "visualization.png"
    render_search_visualization(
        image, output, destination, gt_bbox=BBox(20, 20, 25, 30)
    )
    with Image.open(destination) as rendered:
        assert rendered.size == image.size
