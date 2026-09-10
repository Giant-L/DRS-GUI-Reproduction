"""Reusable full-screen grounding baseline runner and result persistence."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from drsgui.dataset import GroundingSample
from drsgui.evaluator import evaluate_prediction
from drsgui.models.base import GroundingModel


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    sample_count: int
    correct_count: int
    error_count: int
    accuracy: float
    output_dir: str


def run_baseline(
    *,
    model: GroundingModel,
    samples: Iterable[GroundingSample],
    output_dir: str | Path,
    experiment_type: str,
    dataset_name: str = "ScreenSpot-Pro",
    extra_config: dict[str, Any] | None = None,
) -> ExperimentSummary:
    """Run full-screen inference and retain every sample in the denominator."""

    if experiment_type not in {"real", "mock", "synthetic"}:
        raise ValueError("experiment_type must be real, mock, or synthetic")
    selected = list(samples)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    predictions_path = destination / "predictions.jsonl"
    started_at = datetime.now(timezone.utc).isoformat()
    config = {
        "experiment_type": experiment_type,
        "stage": "stage_1_full_screen_baseline",
        "timestamp": started_at,
        "backend": model.backend_name,
        "model": model.model_name,
        "dataset": dataset_name,
        "sample_count": len(selected),
        "coordinate_contract": "original_image_pixels",
        **(extra_config or {}),
    }
    _write_json(destination / "config.json", config)

    correct_count = 0
    error_count = 0
    with predictions_path.open("w", encoding="utf-8") as stream:
        for sample in selected:
            start = time.perf_counter()
            prediction_payload: dict[str, Any] | None = None
            correct = False
            error: dict[str, str] | None = None
            try:
                prediction = model.predict(sample.image_path, sample.instruction)
                evaluation = evaluate_prediction(sample, prediction.point)
                prediction_payload = prediction.to_dict()
                correct = evaluation.correct
                correct_count += int(correct)
            except Exception as exc:  # Keep failed samples in the denominator.
                error_count += 1
                error = {"type": type(exc).__name__, "message": str(exc)}
            latency = time.perf_counter() - start
            record = {
                "sample_id": sample.sample_id,
                "instruction": sample.instruction,
                "image_filename": sample.image_filename,
                "image_size": [sample.image_width, sample.image_height],
                "ui_type": sample.ui_type,
                "application": sample.application,
                "platform": sample.platform,
                "group": sample.group,
                "prediction": prediction_payload,
                "gt_bbox": sample.gt_bbox.to_list(),
                "correct": correct,
                "latency_seconds": latency,
                "error": error,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

    sample_count = len(selected)
    summary = ExperimentSummary(
        sample_count=sample_count,
        correct_count=correct_count,
        error_count=error_count,
        accuracy=(correct_count / sample_count if sample_count else 0.0),
        output_dir=str(destination),
    )
    _write_json(destination / "summary.json", asdict(summary))
    return summary


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
