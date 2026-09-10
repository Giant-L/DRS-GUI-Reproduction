"""Coordinate conversions and ScreenSpot-style point-in-bbox evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from drsgui.dataset import BBox, GroundingSample


@dataclass(frozen=True, slots=True)
class PixelPoint:
    """A point in original-screenshot pixel coordinates."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError(f"point must be finite, got ({self.x}, {self.y})")

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "PixelPoint":
        if len(values) != 2:
            raise ValueError(f"point must contain two values, got {values!r}")
        return cls(float(values[0]), float(values[1]))

    def to_list(self) -> list[float]:
        return [self.x, self.y]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    sample_id: str
    prediction: PixelPoint
    gt_bbox: BBox
    correct: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prediction": self.prediction.to_list(),
            "gt_bbox": self.gt_bbox.to_list(),
            "correct": self.correct,
        }


def point_in_bbox(point: PixelPoint, bbox: BBox) -> bool:
    """Return whether a point lies inside or on the ground-truth box boundary."""

    return bbox.x1 <= point.x <= bbox.x2 and bbox.y1 <= point.y <= bbox.y2


def evaluate_prediction(
    sample: GroundingSample, prediction: PixelPoint
) -> EvaluationResult:
    validate_pixel_point(prediction, sample.image_width, sample.image_height)
    return EvaluationResult(
        sample_id=sample.sample_id,
        prediction=prediction,
        gt_bbox=sample.gt_bbox,
        correct=point_in_bbox(prediction, sample.gt_bbox),
    )


def validate_pixel_point(point: PixelPoint, width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    if not (0 <= point.x < width and 0 <= point.y < height):
        raise ValueError(
            f"pixel point ({point.x}, {point.y}) lies outside {width}x{height}"
        )


def normalized_to_pixel(
    point: PixelPoint,
    *,
    width: int,
    height: int,
    scale: float = 1000.0,
) -> PixelPoint:
    """Convert a 0..scale model coordinate to original-image pixels."""

    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if not (0 <= point.x < scale and 0 <= point.y < scale):
        raise ValueError(
            f"normalized point must be in [0, {scale}), got ({point.x}, {point.y})"
        )
    converted = PixelPoint(point.x / scale * width, point.y / scale * height)
    validate_pixel_point(converted, width, height)
    return converted


def resized_to_original(
    point: PixelPoint,
    *,
    resized_width: int,
    resized_height: int,
    original_width: int,
    original_height: int,
) -> PixelPoint:
    """Undo a direct image resize and return original-image pixel coordinates."""

    validate_pixel_point(point, resized_width, resized_height)
    converted = PixelPoint(
        point.x * original_width / resized_width,
        point.y * original_height / resized_height,
    )
    validate_pixel_point(converted, original_width, original_height)
    return converted


def crop_local_to_global(point: PixelPoint, crop: BBox) -> PixelPoint:
    """Translate crop-local pixels into original-screenshot pixels."""

    if not (0 <= point.x <= crop.width and 0 <= point.y <= crop.height):
        raise ValueError(
            f"local point ({point.x}, {point.y}) lies outside "
            f"crop size {crop.width}x{crop.height}"
        )
    return PixelPoint(crop.x1 + point.x, crop.y1 + point.y)
