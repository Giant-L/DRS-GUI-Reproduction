from __future__ import annotations

import pytest

from drsgui.dataset import BBox
from drsgui.drs.actions import PerceptualActions
from drsgui.drs.config import DRSConfig
from drsgui.drs.geometry import bbox_iou
from drsgui.drs.types import UIElement


def make_element(
    index: int, bbox: BBox, relevance: float, *, interactive: bool = True
) -> UIElement:
    return UIElement(
        element_id=f"e{index:02d}",
        bbox=bbox,
        description=f"Element {index}",
        interactive=interactive,
        relevance=relevance,
    )


def test_focus_keeps_paper_top_fifteen_percent() -> None:
    elements = [
        make_element(index, BBox(100 + index, 100, 105 + index, 110), 1 - index / 100)
        for index in range(20)
    ]
    proposal = PerceptualActions().focus(
        current_region=BBox(0, 0, 1000, 1000),
        all_elements=elements,
        full_region=BBox(0, 0, 1000, 1000),
    )
    assert proposal is not None
    assert proposal.selected_element_ids == ("e00", "e01", "e02")
    assert proposal.region == BBox(100, 100, 107, 110)


def test_focus_prunes_farthest_until_assumed_target_ratio() -> None:
    config = DRSConfig(
        focus_top_fraction=1,
        focus_outlier_distance_fraction=1,
        focus_target_area_ratio=0.2,
    )
    elements = [
        make_element(0, BBox(5, 5, 15, 15), 1.0),
        make_element(1, BBox(45, 45, 55, 55), 0.9),
        make_element(2, BBox(85, 85, 95, 95), 0.8),
    ]
    proposal = PerceptualActions(config).focus(
        current_region=BBox(0, 0, 100, 100),
        all_elements=elements,
        full_region=BBox(0, 0, 100, 100),
    )
    assert proposal is not None
    assert proposal.region.area <= 2000
    assert len(proposal.selected_element_ids) == 1


def test_shift_uses_external_anchor_group_and_respects_iou_cap() -> None:
    current = BBox(100, 100, 300, 300)
    elements = [
        make_element(index, BBox(600 + index, 180, 610 + index, 190), 1 - index / 100)
        for index in range(20)
    ]
    proposal = PerceptualActions().shift(
        current_region=current,
        all_elements=elements,
        full_region=BBox(0, 0, 1000, 600),
    )
    assert proposal is not None
    assert proposal.selected_element_ids == ("e00", "e01", "e02")
    assert proposal.details["direction"] == "right"
    assert proposal.region.width == current.width
    assert proposal.region.height == current.height
    assert bbox_iou(current, proposal.region) <= 0.3


def test_shift_returns_none_when_screen_cannot_satisfy_overlap() -> None:
    current = BBox(0, 0, 100, 100)
    elements = [make_element(0, BBox(95, 45, 100, 55), 1.0)]
    proposal = PerceptualActions().shift(
        current_region=current,
        all_elements=elements,
        full_region=BBox(0, 0, 100, 100),
    )
    assert proposal is None


def test_scatter_adds_external_elements_without_exceeding_area_cap() -> None:
    current = BBox(0, 0, 100, 100)
    elements = [
        make_element(0, BBox(105, 20, 120, 40), 1.0),
        make_element(1, BBox(500, 20, 520, 40), 0.9),
    ]
    config = DRSConfig(scatter_external_fraction=1)
    proposal = PerceptualActions(config).scatter(
        current_region=current,
        all_elements=elements,
        full_region=BBox(0, 0, 1000, 1000),
    )
    assert proposal is not None
    assert proposal.selected_element_ids == ("e00",)
    assert proposal.region == BBox(0, 0, 120, 100)
    assert proposal.region.area <= current.area * 1.5 + 1e-9


def test_actions_reject_unscored_elements() -> None:
    element = UIElement("e", BBox(0, 0, 10, 10), "Element", True)
    with pytest.raises(ValueError, match="missing"):
        PerceptualActions().focus(
            current_region=BBox(0, 0, 100, 100),
            all_elements=[element],
            full_region=BBox(0, 0, 100, 100),
        )
