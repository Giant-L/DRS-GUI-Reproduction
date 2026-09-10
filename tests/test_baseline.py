from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from drsgui.baseline import run_baseline
from drsgui.dataset import BBox, GroundingSample
from drsgui.evaluator import PixelPoint
from drsgui.models.base import GroundingModel, GroundingPrediction, ImageInput


class FixedModel(GroundingModel):
    backend_name = "synthetic-test-only"
    model_name = "fixed-point"

    def predict(self, image: ImageInput, instruction: str) -> GroundingPrediction:
        return GroundingPrediction(
            point=PixelPoint(15, 15),
            backend=self.backend_name,
            model=self.model_name,
            raw_output="(15, 15)",
        )


class FailingModel(GroundingModel):
    backend_name = "synthetic-test-only"
    model_name = "always-fails"

    def predict(self, image: ImageInput, instruction: str) -> GroundingPrediction:
        raise RuntimeError("deliberate test failure")


def test_baseline_persists_config_prediction_and_summary(tmp_path: Path) -> None:
    image_path = tmp_path / "screen.png"
    Image.new("RGB", (100, 50)).save(image_path)
    sample = GroundingSample(
        sample_id="sample_0",
        image_path=image_path,
        image_filename="screen.png",
        instruction="Click target",
        gt_bbox=BBox(10, 10, 20, 20),
        image_width=100,
        image_height=50,
    )
    output_dir = tmp_path / "result"
    summary = run_baseline(
        model=FixedModel(),
        samples=[sample],
        output_dir=output_dir,
        experiment_type="synthetic",
    )
    record = json.loads((output_dir / "predictions.jsonl").read_text())
    config = json.loads((output_dir / "config.json").read_text())
    assert summary.sample_count == 1
    assert summary.correct_count == 1
    assert summary.accuracy == 1.0
    assert record["correct"] is True
    assert record["prediction"]["coordinate_space"] == "original_image_pixels"
    assert config["experiment_type"] == "synthetic"


def test_baseline_keeps_failed_sample_in_denominator(tmp_path: Path) -> None:
    image_path = tmp_path / "screen.png"
    Image.new("RGB", (100, 50)).save(image_path)
    sample = GroundingSample(
        sample_id="sample_0",
        image_path=image_path,
        image_filename="screen.png",
        instruction="Click target",
        gt_bbox=BBox(10, 10, 20, 20),
        image_width=100,
        image_height=50,
    )
    output_dir = tmp_path / "failed-result"
    summary = run_baseline(
        model=FailingModel(),
        samples=[sample],
        output_dir=output_dir,
        experiment_type="synthetic",
    )
    record = json.loads((output_dir / "predictions.jsonl").read_text())
    assert summary.sample_count == 1
    assert summary.correct_count == 0
    assert summary.error_count == 1
    assert summary.accuracy == 0.0
    assert record["correct"] is False
    assert record["error"]["type"] == "RuntimeError"
