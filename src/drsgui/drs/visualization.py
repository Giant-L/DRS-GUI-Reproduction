"""Search-region visualization for auditable small demos."""

from __future__ import annotations

from pathlib import Path

from PIL import ImageDraw

from drsgui.dataset import BBox
from drsgui.drs.types import DRSSearchOutput, PerceptualAction
from drsgui.models.base import ImageInput, load_image


ACTION_COLORS = {
    PerceptualAction.FOCUS: (0, 120, 255),
    PerceptualAction.SHIFT: (255, 140, 0),
    PerceptualAction.SCATTER: (180, 0, 220),
}


def render_search_visualization(
    image: ImageInput,
    output: DRSSearchOutput,
    destination: str | Path,
    *,
    gt_bbox: BBox | None = None,
) -> Path:
    screenshot = load_image(image).convert("RGB")
    draw = ImageDraw.Draw(screenshot)
    line_width = max(2, round(min(screenshot.size) / 500))

    for node in output.search.nodes:
        if node.action is None:
            continue
        color = ACTION_COLORS[node.action]
        draw.rectangle(node.region.to_list(), outline=color, width=line_width)
        draw.text(
            (node.region.x1 + line_width, node.region.y1 + line_width),
            f"{node.node_id}:{node.action.value} {node.reward.total:.3f}",
            fill=color,
            stroke_width=max(1, line_width // 2),
            stroke_fill=(255, 255, 255),
        )

    draw.rectangle(
        output.search.initial_region.to_list(),
        outline=(255, 0, 0),
        width=line_width,
    )
    draw.rectangle(
        output.search.best_region.to_list(),
        outline=(0, 200, 0),
        width=line_width * 2,
    )
    if gt_bbox is not None:
        draw.rectangle(gt_bbox.to_list(), outline=(255, 215, 0), width=line_width * 2)

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot.save(destination_path, format="PNG")
    return destination_path
