#!/usr/bin/env python3
"""Cache OmniParser V2 + Instructor-large output for one dataset sample."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from drsgui.config import Settings
from drsgui.dataset import ScreenSpotProDataset
from drsgui.drs.remote_perception import RemotePerceptionClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--sample-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    dataset = ScreenSpotProDataset(args.dataset or settings.dataset_dir)
    if args.index is not None:
        if args.index < 0:
            raise SystemExit("sample index must be non-negative")
        try:
            sample = dataset[args.index]
        except IndexError as exc:
            raise SystemExit(f"sample index out of range: {args.index}") from exc
    elif args.sample_id:
        try:
            sample = dataset.get(args.sample_id)
        except KeyError as exc:
            raise SystemExit(f"unknown sample id: {args.sample_id}") from exc
    else:
        raise SystemExit("select one sample")

    client = RemotePerceptionClient(
        settings.perception_api_base,
        api_key=settings.perception_api_key,
        timeout_seconds=settings.perception_timeout_seconds,
    )
    cache = client.perceive(
        sample.image_path,
        sample.instruction,
        sample_id=sample.sample_id,
        application=sample.application,
        platform=sample.platform,
    )
    cache["generated_at"] = datetime.now(timezone.utc).isoformat()
    destination = args.output or (
        settings.perception_cache_dir / f"{sample.sample_id}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SystemExit(f"refusing to overwrite existing cache: {destination}")
    destination.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sample_id": sample.sample_id,
                "element_count": len(cache["elements"]),
                "source": cache["source"],
                "relevance_source": cache["relevance_source"],
                "output": str(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
