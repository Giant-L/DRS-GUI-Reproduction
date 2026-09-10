#!/usr/bin/env python3
"""Run one real full-screen grounding sample and persist the result."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from drsgui.baseline import run_baseline
from drsgui.config import Settings
from drsgui.dataset import ScreenSpotProDataset
from drsgui.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("deepseek", "uground"), required=True)
    parser.add_argument("--dataset", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-id")
    selection.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--language", choices=("en", "cn"), default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    dataset_path = args.dataset or settings.dataset_dir
    dataset = ScreenSpotProDataset(dataset_path, language=args.language)
    sample = dataset.get(args.sample_id) if args.sample_id else dataset[args.index]
    model = build_model(args.backend, settings)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or settings.output_dir / f"{timestamp}_{args.backend}_{sample.sample_id}"
    summary = run_baseline(
        model=model,
        samples=[sample],
        output_dir=output,
        experiment_type="real",
        extra_config={"language": args.language, "selection": sample.sample_id},
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
