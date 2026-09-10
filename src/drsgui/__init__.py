"""DRS-GUI paper reimplementation."""

from drsgui.dataset import BBox, GroundingSample, ScreenSpotProDataset
from drsgui.evaluator import PixelPoint, point_in_bbox

__all__ = [
    "BBox",
    "GroundingSample",
    "PixelPoint",
    "ScreenSpotProDataset",
    "point_in_bbox",
]

__version__ = "0.1.0"
