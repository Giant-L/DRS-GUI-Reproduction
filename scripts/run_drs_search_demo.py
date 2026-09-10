#!/usr/bin/env python3
"""Run bounded DRS core search on one real screenshot using cached UI elements."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from drsgui.config import Settings
from drsgui.dataset import ScreenSpotProDataset
from drsgui.drs.config import DRSConfig
from drsgui.drs.mcts import MCTSActionPlanner
from drsgui.drs.perception import CachedUIElementPerceptor, PrecomputedSemanticScorer
from drsgui.drs.search import DynamicRegionSearcher
from drsgui.drs.visualization import render_search_visualization
from drsgui.evaluator import PixelPoint, point_in_bbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    config = DRSConfig()
    scorer = PrecomputedSemanticScorer(source_name=perceptor.relevance_source)
    searcher = DynamicRegionSearcher(
        perceptor=perceptor,
        semantic_scorer=scorer,
        planner=MCTSActionPlanner(config),
    )
    search_output = searcher.search(
        sample.image_path,
        sample.instruction,
        application=sample.application,
        platform=sample.platform,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output or (
        settings.output_dir / f"{timestamp}_drs_search_demo_{sample.sample_id}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    gt_center = PixelPoint(
        (sample.gt_bbox.x1 + sample.gt_bbox.x2) / 2,
        (sample.gt_bbox.y1 + sample.gt_bbox.y2) / 2,
    )
    payload = {
        "experiment_type": "real_screenshot_cached_elements_demo",
        "is_benchmark": False,
        "sample_id": sample.sample_id,
        "instruction": sample.instruction,
        "image_filename": sample.image_filename,
        "gt_bbox_diagnostic_only": sample.gt_bbox.to_list(),
        "gt_used_during_search": False,
        "best_region_contains_gt_center": point_in_bbox(
            gt_center, search_output.search.best_region
        ),
        "perceptor": search_output.perceptor,
        "semantic_scorer": search_output.semantic_scorer,
        "config": config.to_dict(),
        "search": search_output.to_dict(),
    }
    (destination / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    visualization = render_search_visualization(
        sample.image_path,
        search_output,
        destination / "visualization.png",
        gt_bbox=sample.gt_bbox,
    )
    print(
        json.dumps(
            {
                "output": str(destination),
                "sample_id": sample.sample_id,
                "best_region": search_output.search.best_region.to_list(),
                "best_reward": search_output.search.best_reward.total,
                "action_path": [
                    action.value for action in search_output.search.action_path
                ],
                "best_region_contains_gt_center": payload[
                    "best_region_contains_gt_center"
                ],
                "visualization": str(visualization),
                "warning": "cached-element demo; not a benchmark or model result",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
