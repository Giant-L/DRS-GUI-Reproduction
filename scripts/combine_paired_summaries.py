#!/usr/bin/env python3
"""Combine disjoint fixed paired-smoke reports using their persisted records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def weighted(values: list[tuple[float, int]]) -> float:
    return sum(value * count for value, count in values) / sum(count for _, count in values)


def main(args: argparse.Namespace) -> None:
    inputs = [Path(value) for value in args.inputs]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = [read_json(path / "summary.json") for path in inputs]
    analyses = [read_json(path / "analysis.json") for path in inputs]
    comparisons = sum((read_jsonl(path / "comparison.jsonl") for path in inputs), [])
    baseline = sum((read_jsonl(path / "baseline.jsonl") for path in inputs), [])
    reproduction = sum((read_jsonl(path / "reproduction.jsonl") for path in inputs), [])
    count = len(comparisons)
    hits = sum(row["best_region_contains_gt_center"] for row in comparisons)
    drs_hit_correct = sum(row["reproduction_correct"] and row["best_region_contains_gt_center"] for row in comparisons)
    base_hit_correct = sum(row["baseline_correct"] and row["best_region_contains_gt_center"] for row in comparisons)
    base_correct = sum(row["baseline_correct"] for row in comparisons)
    drs_correct = sum(row["reproduction_correct"] for row in comparisons)
    base_tokens = sum(row["baseline_tokens"] for row in comparisons)
    drs_tokens = sum(row["reproduction_tokens"] for row in comparisons)
    summary = {
        "experiment_type": "real_fixed_40_sample_paired_smoke_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "ScreenSpot-Pro",
        "indices": [row["index"] for row in comparisons],
        "sample_count": count,
        "grounding_model": "deepseek-flash",
        "baseline_correct": base_correct,
        "baseline_accuracy": base_correct / count,
        "reproduction_correct": drs_correct,
        "reproduction_accuracy": drs_correct / count,
        "accuracy_delta": (drs_correct - base_correct) / count,
        "paired_outcomes": dict(Counter(row["outcome"] for row in comparisons)),
        "best_region_contains_gt_center": hits,
        "reproduction_failure_types": dict(Counter(row["reproduction_failure_type"] for row in comparisons)),
        "baseline_total_tokens": base_tokens,
        "reproduction_total_tokens": drs_tokens,
        "token_reduction_fraction": 1 - drs_tokens / base_tokens,
        "baseline_mean_grounding_latency_seconds": weighted([(item["baseline_mean_grounding_latency_seconds"], item["sample_count"]) for item in summaries]),
        "reproduction_mean_grounding_latency_seconds": weighted([(item["reproduction_mean_grounding_latency_seconds"], item["sample_count"]) for item in summaries]),
        "reproduction_mean_perception_latency_seconds": weighted([(item["reproduction_mean_perception_latency_seconds"], item["sample_count"]) for item in summaries]),
        "baseline_errors": sum(item["baseline_errors"] for item in summaries),
        "reproduction_errors": sum(item["reproduction_errors"] for item in summaries),
        "warning": "Combined fixed non-random smoke subset, not benchmark accuracy. DRS contains documented reproduction assumptions and an OmniParser checkpoint not pinned by the paper.",
    }
    representative = {}
    for key in ("both_correct", "reproduction_only", "baseline_only"):
        match = next((row["index"] for row in comparisons if row["outcome"] == key), None)
        if match is not None:
            representative[key] = match
    analysis = {
        "region_recall": hits / count,
        "reproduction_accuracy_given_region_hit": drs_hit_correct / hits if hits else 0,
        "baseline_accuracy_given_region_hit": base_hit_correct / hits if hits else 0,
        "reproduction_accuracy_given_region_miss": 0.0,
        "region_recall_oracle_upper_bound": hits / count,
        "mean_best_region_area_ratio": weighted([(item["mean_best_region_area_ratio"], summaries[i]["sample_count"]) for i, item in enumerate(analyses)]),
        "mean_area_reduction_fraction": weighted([(item["mean_area_reduction_fraction"], summaries[i]["sample_count"]) for i, item in enumerate(analyses)]),
        "representative_cases": representative,
        "conclusion": "Search-region recall remains the dominant bottleneck in the combined smoke subset.",
    }

    def dump(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for filename, records in (("baseline.jsonl", baseline), ("reproduction.jsonl", reproduction), ("comparison.jsonl", comparisons)):
        (output / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in comparisons for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparisons)
    dump(output / "summary.json", summary)
    dump(output / "analysis.json", analysis)
    dump(output / "config.json", {"source_reports": [str(path) for path in inputs], "indices": summary["indices"], "all_failures_retained": True})
    lines = [
        "# Combined fixed 40-sample baseline vs DRS smoke comparison", "",
        "> Real ScreenSpot-Pro samples and real model calls. This fixed, non-random subset is not benchmark accuracy.", "",
        f"- Baseline: **{base_correct}/{count} = {summary['baseline_accuracy']:.1%}**",
        f"- Reproduction: **{drs_correct}/{count} = {summary['reproduction_accuracy']:.1%}**",
        f"- Accuracy delta: **{summary['accuracy_delta']:+.1%}**",
        f"- Best Region contains GT center: **{hits}/{count} = {analysis['region_recall']:.1%}**",
        f"- Grounding tokens: **{base_tokens} -> {drs_tokens} ({summary['token_reduction_fraction']:.1%} reduction)**",
        f"- Mean grounding latency: **{summary['baseline_mean_grounding_latency_seconds']:.2f}s -> {summary['reproduction_mean_grounding_latency_seconds']:.2f}s** (DRS excludes perception)",
        f"- Mean remote perception latency: **{summary['reproduction_mean_perception_latency_seconds']:.2f}s**",
        f"- Outcomes: `{summary['paired_outcomes']}`",
        f"- DRS failure types: `{summary['reproduction_failure_types']}`",
        f"- API/validation errors: baseline **{summary['baseline_errors']}**, reproduction **{summary['reproduction_errors']}**", "",
        "| Index | Sample | Base | DRS | Region GT | Outcome |", "|---:|---|:---:|:---:|:---:|---|",
    ]
    for row in comparisons:
        lines.append(f"| {row['index']} | {row['sample_id']} | {'✓' if row['baseline_correct'] else '✗'} | {'✓' if row['reproduction_correct'] else '✗'} | {'✓' if row['best_region_contains_gt_center'] else '✗'} | {row['outcome']} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    badcases = [
        "# Combined badcase analysis", "",
        f"- Best Region recall: **{hits}/{count} = {analysis['region_recall']:.1%}**",
        f"- DRS accuracy given region hit: **{drs_hit_correct}/{hits} = {analysis['reproduction_accuracy_given_region_hit']:.1%}**" if hits else "- DRS accuracy given region hit: **N/A**",
        f"- Mean Best Region area: **{analysis['mean_best_region_area_ratio']:.1%}** of full screenshot",
        f"- Mean image-area reduction: **{analysis['mean_area_reduction_fraction']:.1%}**", "",
        analysis["conclusion"],
    ]
    (output / "BADCASE_REPORT.md").write_text("\n".join(badcases) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "analysis": analysis}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
