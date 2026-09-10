from __future__ import annotations

import pytest

from drsgui.dataset import BBox
from drsgui.drs.geometry import (
    bbox_iou,
    clip_bbox,
    enclosing_bbox,
    full_image_region,
    pixel_crop_box,
    recentered_region,
    visible_elements,
)
from drsgui.drs.types import UIElement


def element(element_id: str, bbox: BBox) -> UIElement:
    return UIElement(element_id, bbox, element_id, True, 0.5)


def test_enclosing_bbox_and_iou() -> None:
    assert enclosing_bbox([BBox(10, 20, 30, 40), BBox(20, 10, 50, 25)]) == BBox(
        10, 10, 50, 40
    )
    assert bbox_iou(BBox(0, 0, 10, 10), BBox(5, 0, 15, 10)) == pytest.approx(
        1 / 3
    )


def test_clip_and_pixel_crop_round_outward() -> None:
    full = full_image_region(100, 50)
    clipped = clip_bbox(BBox(-5, 4.2, 102, 20.1), full)
    assert clipped == BBox(0, 4.2, 100, 20.1)
    assert pixel_crop_box(clipped, full) == (0, 4, 100, 21)


def test_recentered_region_stays_inside_screen() -> None:
    assert recentered_region(
        center=(98, 48), width=20, height=10, bounds=BBox(0, 0, 100, 50)
    ) == BBox(80, 40, 100, 50)


def test_visible_elements_use_center_membership() -> None:
    elements = (
        element("inside", BBox(10, 10, 20, 20)),
        element("outside", BBox(70, 10, 80, 20)),
    )
    assert [item.element_id for item in visible_elements(elements, BBox(0, 0, 50, 50))] == [
        "inside"
    ]
