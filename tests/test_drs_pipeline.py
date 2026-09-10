from __future__ import annotations

from pathlib import Path

from PIL import Image

from drsgui.dataset import BBox
from drsgui.drs.pipeline import DRSGroundingPipeline
from drsgui.drs.types import (
    DRSSearchOutput,
    RewardBreakdown,
    SearchResult,
)
from drsgui.evaluator import PixelPoint
from drsgui.models.base import GroundingModel, GroundingPrediction, ImageInput, load_image


class StaticSearcher:
    def search(
        self,
        image: ImageInput,
        instruction: str,
        *,
        application: str | None = None,
        platform: str | None = None,
    ) -> DRSSearchOutput:
        del image, instruction, application, platform
        reward = RewardBreakdown(0.5, 0.1, 0.8, 0.4, 2)
        search = SearchResult(
            full_region=BBox(0, 0, 100, 80),
            initial_region=BBox(10.2, 20.4, 30.1, 40.2),
            best_region=BBox(10.2, 20.4, 30.1, 40.2),
            initial_reward=reward,
            best_reward=reward,
            action_path=(),
            rollout_budget=8,
            iterations=8,
            nodes=(),
            assumptions={},
        )
        return DRSSearchOutput(search, (), "static", "static")


class CropLocalModel(GroundingModel):
    backend_name = "crop-local-test"
    model_name = "fixed"

    def predict(self, image: ImageInput, instruction: str) -> GroundingPrediction:
        assert load_image(image).size == (21, 21)
        assert instruction == "Click target"
        return GroundingPrediction(
            point=PixelPoint(5, 7),
            backend=self.backend_name,
            model=self.model_name,
            raw_output="(5, 7)",
        )


def test_pipeline_restores_crop_local_coordinate_to_original_pixels(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "screen.png"
    Image.new("RGB", (100, 80)).save(image_path)
    pipeline = DRSGroundingPipeline(StaticSearcher(), CropLocalModel())
    result = pipeline.predict(image_path, "Click target")
    assert result.crop_region == BBox(10, 20, 31, 41)
    assert result.local_prediction.point == PixelPoint(5, 7)
    assert result.local_prediction.coordinate_space == "crop_local_pixels"
    assert result.prediction.point == PixelPoint(15, 27)
    assert result.prediction.coordinate_space == "original_image_pixels"
    assert result.prediction.metadata["coordinate_transform"] == (
        "crop_local_pixels_to_original_image_pixels"
    )
