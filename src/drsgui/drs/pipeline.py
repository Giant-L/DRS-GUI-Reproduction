"""Crop grounding and restoration to original screenshot coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drsgui.dataset import BBox
from drsgui.drs.geometry import full_image_region, pixel_crop_box
from drsgui.drs.search import RegionSearcher
from drsgui.drs.types import DRSSearchOutput
from drsgui.evaluator import PixelPoint, validate_pixel_point
from drsgui.models.base import (
    GroundingModel,
    GroundingPrediction,
    ImageInput,
    load_image,
)


@dataclass(frozen=True, slots=True)
class DRSGroundingPrediction:
    prediction: GroundingPrediction
    local_prediction: GroundingPrediction
    crop_region: BBox
    search_output: DRSSearchOutput

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction.to_dict(),
            "local_prediction": self.local_prediction.to_dict(),
            "crop_region": self.crop_region.to_list(),
            "search": self.search_output.to_dict(),
        }


@dataclass(slots=True)
class DRSGroundingPipeline:
    searcher: RegionSearcher
    grounding_model: GroundingModel

    def predict(
        self,
        image: ImageInput,
        instruction: str,
        *,
        application: str | None = None,
        platform: str | None = None,
    ) -> DRSGroundingPrediction:
        screenshot = load_image(image)
        full_region = full_image_region(*screenshot.size)
        search_output = self.searcher.search(
            screenshot,
            instruction,
            application=application,
            platform=platform,
        )
        crop_box = pixel_crop_box(search_output.search.best_region, full_region)
        crop = screenshot.crop(crop_box)
        backend_prediction = self.grounding_model.predict(crop, instruction)
        validate_pixel_point(backend_prediction.point, crop.width, crop.height)
        local = GroundingPrediction(
            point=backend_prediction.point,
            backend=backend_prediction.backend,
            model=backend_prediction.model,
            raw_output=backend_prediction.raw_output,
            coordinate_space="crop_local_pixels",
            metadata=dict(backend_prediction.metadata),
        )
        global_point = PixelPoint(
            crop_box[0] + local.point.x,
            crop_box[1] + local.point.y,
        )
        validate_pixel_point(global_point, screenshot.width, screenshot.height)
        crop_region = BBox(*(float(value) for value in crop_box))
        global_prediction = GroundingPrediction(
            point=global_point,
            backend=f"drs+{local.backend}",
            model=local.model,
            raw_output=local.raw_output,
            coordinate_space="original_image_pixels",
            metadata={
                **dict(local.metadata),
                "coordinate_transform": "crop_local_pixels_to_original_image_pixels",
                "crop_region": crop_region.to_list(),
                "local_point": local.point.to_list(),
            },
        )
        return DRSGroundingPrediction(
            prediction=global_prediction,
            local_prediction=local,
            crop_region=crop_region,
            search_output=search_output,
        )
