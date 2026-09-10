#!/usr/bin/env python3
"""Run one real DRS crop-grounding sample with an explicitly selected backend."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from drsgui.config import Settings
from drsgui.dataset import ScreenSpotProDataset
from drsgui.drs.config import DRSConfig
from drsgui.drs.mcts import MCTSActionPlanner
from drsgui.drs.perception import CachedUIElementPerceptor, PrecomputedSemanticScorer
from drsgui.drs.pipeline import DRSGroundingPipeline
from drsgui.drs.search import DynamicRegionSearcher
from drsgui.evaluator import evaluate_prediction
from drsgui.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("deepseek", "uground"), required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--elements", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.index < 0:
        raise SystemExit("--index must be non-negative")
    settings = Settings.from_env()
    dataset = ScreenSpotProDataset(args.dataset or settings.dataset_dir)
    try:
        sample = dataset[args.index]
    except IndexError as exc:
        raise SystemExit(f"sample index out of range: {args.index}") from exc
    perceptor = CachedUIElementPerceptor.from_json(args.elements)
    if perceptor.sample_id is not None and perceptor.sample_id != sample.sample_id:
        raise SystemExit(
            f"element cache belongs to {perceptor.sample_id}, not {sample.sample_id}"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output or (
        settings.output_dir / f"{timestamp}_drs_{args.backend}_{sample.sample_id}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    config = DRSConfig()
    started = time.perf_counter()
    try:
        pipeline = DRSGroundingPipeline(
            DynamicRegionSearcher(
                perceptor,
                PrecomputedSemanticScorer(source_name=perceptor.relevance_source),
                MCTSActionPlanner(config),
            ),
            build_model(args.backend, settings),
        )
        result = pipeline.predict(
            sample.image_path,
            sample.instruction,
            application=sample.application,
            platform=sample.platform,
        )
        evaluation = evaluate_prediction(sample, result.prediction.point)
        prediction = result.to_dict()
        correct = evaluation.correct
        error = None
    except Exception as exc:
        prediction = None
        correct = False
        error = {"type": type(exc).__name__, "message": str(exc)}
    latency = time.perf_counter() - started
    payload = {
        "experiment_type": "real_single_sample",
        "sample_id": sample.sample_id,
        "instruction": sample.instruction,
        "backend": args.backend,
        "perceptor": perceptor.source_name,
        "semantic_scorer": perceptor.relevance_source,
        "prediction": prediction,
        "gt_bbox": sample.gt_bbox.to_list(),
        "correct": correct,
        "latency_seconds": latency,
        "error": error,
        "config": config.to_dict(),
    }
    (destination / "prediction.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(destination), **payload}, indent=2))
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
