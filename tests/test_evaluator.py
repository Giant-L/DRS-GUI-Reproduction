from __future__ import annotations

import pytest

from drsgui.dataset import BBox
from drsgui.evaluator import (
    PixelPoint,
    crop_local_to_global,
    normalized_to_pixel,
    point_in_bbox,
    resized_to_original,
    validate_pixel_point,
)


@pytest.mark.parametrize(
    "point",
    [PixelPoint(10, 20), PixelPoint(30, 40), PixelPoint(20, 30)],
)
def test_point_in_bbox_includes_boundaries(point: PixelPoint) -> None:
    assert point_in_bbox(point, BBox(10, 20, 30, 40))


@pytest.mark.parametrize(
    "point",
    [PixelPoint(9.99, 30), PixelPoint(30.01, 30), PixelPoint(20, 40.01)],
)
def test_point_in_bbox_rejects_outside(point: PixelPoint) -> None:
    assert not point_in_bbox(point, BBox(10, 20, 30, 40))


def test_uground_normalized_coordinate_converts_to_original_pixels() -> None:
    result = normalized_to_pixel(PixelPoint(250, 500), width=3840, height=1080)
    assert result == PixelPoint(960, 540)


def test_normalized_coordinate_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="normalized point"):
        normalized_to_pixel(PixelPoint(1000, 500), width=3840, height=1080)


def test_resized_coordinate_converts_to_original_pixels() -> None:
    result = resized_to_original(
        PixelPoint(500, 250),
        resized_width=1000,
        resized_height=500,
        original_width=3840,
        original_height=1080,
    )
    assert result == PixelPoint(1920, 540)


def test_crop_local_coordinate_converts_to_global_pixels() -> None:
    assert crop_local_to_global(
        PixelPoint(50, 25), BBox(100, 200, 300, 400)
    ) == PixelPoint(150, 225)


def test_validate_pixel_point_rejects_image_boundary_overflow() -> None:
    with pytest.raises(ValueError, match="outside"):
        validate_pixel_point(PixelPoint(100, 20), width=100, height=50)
