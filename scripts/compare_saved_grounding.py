#!/usr/bin/env python3
"""Compare at most ten new DRS records with persisted paired results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--saved-comparison", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indices = [int(value) for value in args.indices.split(",")]
    if not indices or len(indices) > 10 or len(indices) != len(set(indices)):
        raise SystemExit("provide between 1 and 10 unique indices")
    saved = {row["index"]: row for row in read_jsonl(args.saved_comparison)}
    rows = []
    for index in indices:
        previous = saved[index]
        current = json.loads(
            (args.new_root / f"reproduction_{index}" / "prediction.json").read_text()
        )
        rows.append(
            {
                "index": index,
                "sample_id": current["sample_id"],
                "instruction": current["instruction"],
                "saved_baseline_correct": previous["baseline_correct"],
                "old_reproduction_correct": previous["reproduction_correct"],
                "new_reproduction_correct": current["correct"],
                "new_reproduction_error": current["error"],
                "new_config": current["config"],
            }
        )
    summary = {
        "experiment_type": "fixed_saved_baseline_vs_new_reproduction_diagnostic",
        "sample_count": len(rows),
        "indices": indices,
        "baseline_was_rerun": False,
        "saved_baseline_correct": sum(row["saved_baseline_correct"] for row in rows),
        "old_reproduction_correct": sum(
            row["old_reproduction_correct"] for row in rows
        ),
        "new_reproduction_correct": sum(
            row["new_reproduction_correct"] for row in rows
        ),
        "new_reproduction_errors": sum(
            row["new_reproduction_error"] is not None for row in rows
        ),
        "warning": (
            "Fixed small diagnostic, not benchmark accuracy. Baseline and old DRS "
            "records are persisted historical calls; only new DRS was called now."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    report = [
        "# Saved baseline vs new DRS diagnostic",
        "",
        "> Fixed small diagnostic, not benchmark accuracy. Baseline was not rerun.",
        "",
        f"- Saved baseline: **{summary['saved_baseline_correct']}/{len(rows)}**",
        f"- Old DRS: **{summary['old_reproduction_correct']}/{len(rows)}**",
        f"- New DRS: **{summary['new_reproduction_correct']}/{len(rows)}**",
        f"- New DRS errors: **{summary['new_reproduction_errors']}**",
        "",
        "| Index | Sample | Baseline | Old DRS | New DRS |",
        "|---:|---|:---:|:---:|:---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['index']} | {row['sample_id']} | "
            f"{'✓' if row['saved_baseline_correct'] else '✗'} | "
            f"{'✓' if row['old_reproduction_correct'] else '✗'} | "
            f"{'✓' if row['new_reproduction_correct'] else '✗'} |"
        )
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
