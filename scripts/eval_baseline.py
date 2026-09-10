#!/usr/bin/env python3
"""Run a bounded real ScreenSpot-Pro full-screen baseline evaluation."""

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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("deepseek", "uground"), required=True)
    parser.add_argument(
        "--limit",
        type=positive_int,
        required=True,
        help="explicit sample limit; required to prevent accidental full runs",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--language", choices=("en", "cn"), default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offset < 0:
        raise SystemExit("--offset must be non-negative")
    settings = Settings.from_env()
    dataset_path = args.dataset or settings.dataset_dir
    dataset = ScreenSpotProDataset(dataset_path, language=args.language)
    selected = dataset[args.offset : args.offset + args.limit]
    if len(selected) != args.limit:
        raise SystemExit(
            f"requested {args.limit} samples at offset {args.offset}, "
            f"but dataset only yielded {len(selected)}"
        )
    model = build_model(args.backend, settings)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or settings.output_dir / f"{timestamp}_{args.backend}_screenspot_pro"
    summary = run_baseline(
        model=model,
        samples=selected,
        output_dir=output,
        experiment_type="real",
        extra_config={
            "language": args.language,
            "offset": args.offset,
            "limit": args.limit,
        },
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
