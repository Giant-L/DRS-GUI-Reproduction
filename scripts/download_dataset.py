#!/usr/bin/env python3
"""Download and validate the official ScreenSpot-Pro snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from drsgui.dataset import ScreenSpotProDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id", default="likaixin/ScreenSpot-Pro", help="Hugging Face dataset id"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/ScreenSpot-Pro")
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=10,
        help="number of real images to open and size-check after download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_workers <= 0 or args.verify_samples <= 0:
        raise SystemExit("--max-workers and --verify-samples must be positive")
    load_dotenv(override=False)
    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
        max_workers=args.max_workers,
    )
    dataset = ScreenSpotProDataset(local_dir)
    validated = dataset.validate_images(limit=args.verify_samples)
    print(f"Downloaded dataset to: {local_dir}")
    print(f"Loaded annotations: {len(dataset)} samples")
    print(f"Validated real images: {len(validated)}")
    for item in validated:
        print(
            f"  {item['sample_id']}: {item['image_filename']} "
            f"{item['image_size'][0]}x{item['image_size'][1]}"
        )


if __name__ == "__main__":
    main()
