"""Deterministic bounding-box operations for region search."""

from __future__ import annotations

import math
from collections.abc import Iterable

from drsgui.dataset import BBox
from drsgui.drs.types import UIElement


def full_image_region(width: int, height: int) -> BBox:
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    return BBox(0.0, 0.0, float(width), float(height))


def bbox_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2)


def enclosing_bbox(boxes: Iterable[BBox]) -> BBox:
    materialized = tuple(boxes)
    if not materialized:
        raise ValueError("cannot enclose an empty collection of boxes")
    return BBox(
        min(box.x1 for box in materialized),
        min(box.y1 for box in materialized),
        max(box.x2 for box in materialized),
        max(box.y2 for box in materialized),
    )


def intersection_area(left: BBox, right: BBox) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return width * height


def bbox_iou(left: BBox, right: BBox) -> float:
    intersection = intersection_area(left, right)
    union = left.area + right.area - intersection
    return 0.0 if union <= 0 else intersection / union


def clip_bbox(box: BBox, bounds: BBox) -> BBox:
    clipped = BBox(
        max(box.x1, bounds.x1),
        max(box.y1, bounds.y1),
        min(box.x2, bounds.x2),
        min(box.y2, bounds.y2),
    )
    if clipped.width <= 0 or clipped.height <= 0:
        raise ValueError(f"box {box.to_list()} does not overlap {bounds.to_list()}")
    return clipped


def center_inside(element: UIElement, region: BBox) -> bool:
    x, y = element.center
    return region.x1 <= x <= region.x2 and region.y1 <= y <= region.y2


def visible_elements(
    elements: Iterable[UIElement], region: BBox
) -> tuple[UIElement, ...]:
    """Use element-center membership for cached global parser output."""

    return tuple(element for element in elements if center_inside(element, region))


def region_diagonal(region: BBox) -> float:
    return math.hypot(region.width, region.height)


def same_bbox(left: BBox, right: BBox, *, tolerance: float = 1e-6) -> bool:
    return all(
        math.isclose(a, b, abs_tol=tolerance, rel_tol=0.0)
        for a, b in zip(left.to_list(), right.to_list(), strict=True)
    )


def recentered_region(
    *,
    center: tuple[float, float],
    width: float,
    height: float,
    bounds: BBox,
) -> BBox:
    if width <= 0 or height <= 0:
        raise ValueError("region dimensions must be positive")
    width = min(width, bounds.width)
    height = min(height, bounds.height)
    x1 = center[0] - width / 2
    y1 = center[1] - height / 2
    x1 = min(max(x1, bounds.x1), bounds.x2 - width)
    y1 = min(max(y1, bounds.y1), bounds.y2 - height)
    return BBox(x1, y1, x1 + width, y1 + height)


def pixel_crop_box(region: BBox, full_region: BBox) -> tuple[int, int, int, int]:
    clipped = clip_bbox(region, full_region)
    left = max(int(math.floor(clipped.x1)), int(full_region.x1))
    top = max(int(math.floor(clipped.y1)), int(full_region.y1))
    right = min(int(math.ceil(clipped.x2)), int(full_region.x2))
    bottom = min(int(math.ceil(clipped.y2)), int(full_region.y2))
    if right <= left or bottom <= top:
        raise ValueError(f"region does not define a non-empty pixel crop: {region}")
    return left, top, right, bottom
