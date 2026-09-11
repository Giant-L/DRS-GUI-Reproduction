#!/usr/bin/env python3
"""Evaluate DRS Best Region recall on at most ten cached real samples."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from drsgui.config import Settings
from drsgui.dataset import ScreenSpotProDataset
from drsgui.drs.config import DRSConfig
from drsgui.drs.geometry import full_image_region
from drsgui.drs.mcts import MCTSActionPlanner
from drsgui.drs.perception import CachedUIElementPerceptor
from drsgui.evaluator import PixelPoint, point_in_bbox


def parse_case(value: str) -> tuple[int, Path]:
    try:
        raw_index, raw_path = value.split("=", 1)
        index = int(raw_index)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be INDEX=CACHE_JSON") from exc
    path = Path(raw_path)
    if index < 0 or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid case: {value}")
    return index, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--focus-outlier", type=float, required=True)
    parser.add_argument("--focus-target-area-ratio", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.case) > 10:
        raise SystemExit("search diagnostics are intentionally limited to 10 cases")
    indices = [index for index, _ in args.case]
    if len(indices) != len(set(indices)):
        raise SystemExit("case indices must be unique")

    config = DRSConfig(
        focus_outlier_distance_fraction=args.focus_outlier,
        focus_target_area_ratio=args.focus_target_area_ratio,
    )
    dataset = ScreenSpotProDataset(Settings.from_env().dataset_dir)
    planner = MCTSActionPlanner(config)
    rows = []
    for index, cache_path in args.case:
        sample = dataset[index]
        full_region = full_image_region(sample.image_width, sample.image_height)
        perceptor = CachedUIElementPerceptor.from_json(cache_path)
        if perceptor.sample_id and perceptor.sample_id != sample.sample_id:
            raise SystemExit(
                f"cache {cache_path} belongs to {perceptor.sample_id}, "
                f"not {sample.sample_id}"
            )
        elements = perceptor.parse(sample.image_path, full_region)
        result = planner.search(full_region=full_region, elements=elements)
        gt_center = PixelPoint(
            (sample.gt_bbox.x1 + sample.gt_bbox.x2) / 2,
            (sample.gt_bbox.y1 + sample.gt_bbox.y2) / 2,
        )
        rows.append(
            {
                "index": index,
                "sample_id": sample.sample_id,
                "instruction": sample.instruction,
                "best_region": result.best_region.to_list(),
                "best_region_contains_gt_center": point_in_bbox(
                    gt_center, result.best_region
                ),
                "best_region_area_ratio": result.best_region.area / full_region.area,
                "best_reward": result.best_reward.to_dict(),
                "action_path": [action.value for action in result.action_path],
                "search": result.to_dict(),
            }
        )

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    hits = sum(row["best_region_contains_gt_center"] for row in rows)
    summary = {
        "experiment_type": "real_fixed_search_only_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_benchmark": False,
        "grounding_model_called": False,
        "ground_truth_used_during_search": False,
        "dataset": "ScreenSpot-Pro",
        "indices": indices,
        "sample_count": len(rows),
        "best_region_hits": hits,
        "best_region_recall": hits / len(rows),
        "mean_best_region_area_ratio": mean(
            row["best_region_area_ratio"] for row in rows
        ),
        "config": config.to_dict(),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
