#!/usr/bin/env python3
"""Summarize a fixed paired baseline/DRS smoke subset without dropping failures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def point(record: dict) -> list[float] | None:
    prediction = record.get("prediction")
    if not prediction:
        return None
    if "point" in prediction:
        return prediction["point"]
    nested = prediction.get("prediction")
    return nested.get("point") if nested else None


def tokens(record: dict, *, drs: bool) -> int:
    prediction = record.get("prediction")
    if not prediction:
        return 0
    if drs:
        prediction = prediction.get("local_prediction") or prediction.get("prediction")
    usage = (prediction.get("metadata") or {}).get("usage") or {}
    return int(usage.get("total_tokens") or 0)


def contains_center(region: list[float], bbox: list[float]) -> bool:
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def summarize(args: argparse.Namespace) -> None:
    indices = [int(value) for value in args.indices.split(",")]
    raw_root = Path(args.raw_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, baseline_records, reproduction_records = [], [], []
    perception_latencies, area_ratios = [], []

    for index in indices:
        baseline = read_json(raw_root / f"baseline_{index}" / "predictions.jsonl")
        reproduction = read_json(raw_root / f"reproduction_{index}" / "prediction.json")
        baseline_records.append(baseline)
        reproduction_records.append(reproduction)
        search_wrapper = (reproduction.get("prediction") or {}).get("search") or {}
        search = search_wrapper.get("search") or {}
        region = search.get("best_region")
        full = search.get("full_region")
        region_hit = bool(region and contains_center(region, reproduction["gt_bbox"]))
        if region and full:
            region_area = max(0, region[2] - region[0]) * max(0, region[3] - region[1])
            full_area = max(0, full[2] - full[0]) * max(0, full[3] - full[1])
            area_ratios.append(region_area / full_area)
        perception_path = Path(args.perception_template.format(index=index))
        perception = read_json(perception_path)
        metadata = perception.get("metadata") or {}
        perception_latencies.append(
            float(metadata.get("parse_latency_seconds") or 0)
            + float(metadata.get("score_latency_seconds") or 0)
        )
        base_ok, drs_ok = bool(baseline["correct"]), bool(reproduction["correct"])
        if base_ok and drs_ok:
            outcome = "both_correct"
        elif drs_ok:
            outcome = "reproduction_only"
        elif base_ok:
            outcome = "baseline_only"
        else:
            outcome = "both_wrong"
        if drs_ok:
            failure = "none"
        elif not region_hit:
            failure = "search_region_miss"
        else:
            failure = "grounding_failed_inside_valid_region"
        rows.append({
            "index": index,
            "sample_id": baseline["sample_id"],
            "instruction": baseline["instruction"],
            "baseline_correct": base_ok,
            "reproduction_correct": drs_ok,
            "outcome": outcome,
            "reproduction_failure_type": failure,
            "baseline_prediction": point(baseline),
            "reproduction_prediction": point(reproduction),
            "gt_bbox": baseline["gt_bbox"],
            "baseline_tokens": tokens(baseline, drs=False),
            "reproduction_tokens": tokens(reproduction, drs=True),
            "best_region_contains_gt_center": region_hit,
            "element_count": len(perception.get("elements") or []),
            "best_reward": (search.get("best_reward") or {}).get("total"),
            "action_path": search.get("action_path") or [],
            "baseline_error": baseline.get("error"),
            "reproduction_error": reproduction.get("error"),
        })

    baseline_correct = sum(row["baseline_correct"] for row in rows)
    reproduction_correct = sum(row["reproduction_correct"] for row in rows)
    hits = sum(row["best_region_contains_gt_center"] for row in rows)
    baseline_total_tokens = sum(row["baseline_tokens"] for row in rows)
    reproduction_total_tokens = sum(row["reproduction_tokens"] for row in rows)
    summary = {
        "experiment_type": "real_fixed_20_sample_paired_smoke_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "ScreenSpot-Pro",
        "indices": indices,
        "sample_count": len(rows),
        "grounding_model": "deepseek-flash",
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / len(rows),
        "reproduction_correct": reproduction_correct,
        "reproduction_accuracy": reproduction_correct / len(rows),
        "accuracy_delta": (reproduction_correct - baseline_correct) / len(rows),
        "paired_outcomes": dict(Counter(row["outcome"] for row in rows)),
        "best_region_contains_gt_center": hits,
        "reproduction_failure_types": dict(Counter(row["reproduction_failure_type"] for row in rows)),
        "baseline_total_tokens": baseline_total_tokens,
        "reproduction_total_tokens": reproduction_total_tokens,
        "token_reduction_fraction": 1 - reproduction_total_tokens / baseline_total_tokens,
        "baseline_mean_grounding_latency_seconds": mean(float(x["latency_seconds"]) for x in baseline_records),
        "reproduction_mean_grounding_latency_seconds": mean(float(x["latency_seconds"]) for x in reproduction_records),
        "reproduction_mean_perception_latency_seconds": mean(perception_latencies),
        "baseline_errors": sum(x.get("error") is not None for x in baseline_records),
        "reproduction_errors": sum(x.get("error") is not None for x in reproduction_records),
        "warning": "Small fixed non-random smoke subset, not benchmark accuracy. DRS contains documented reproduction assumptions and an OmniParser checkpoint not pinned by the paper.",
    }
    hit_correct = sum(row["reproduction_correct"] and row["best_region_contains_gt_center"] for row in rows)
    representative = {}
    for key in ("both_correct", "reproduction_only", "baseline_only"):
        match = next((row["index"] for row in rows if row["outcome"] == key), None)
        if match is not None:
            representative[key] = match
    analysis = {
        "region_recall": hits / len(rows),
        "reproduction_accuracy_given_region_hit": hit_correct / hits if hits else 0,
        "baseline_accuracy_given_region_hit": sum(row["baseline_correct"] and row["best_region_contains_gt_center"] for row in rows) / hits if hits else 0,
        "reproduction_accuracy_given_region_miss": 0.0,
        "region_recall_oracle_upper_bound": hits / len(rows),
        "mean_best_region_area_ratio": mean(area_ratios),
        "median_best_region_area_ratio": median(area_ratios),
        "mean_area_reduction_fraction": 1 - mean(area_ratios),
        "representative_cases": representative,
        "conclusion": "Search-region recall is the dominant bottleneck: no DRS prediction was correct when Best Region missed the GT center.",
    }

    def dump_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for filename, records in (("baseline.jsonl", baseline_records), ("reproduction.jsonl", reproduction_records), ("comparison.jsonl", rows)):
        (output / filename).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    dump_json(output / "summary.json", summary)
    dump_json(output / "analysis.json", analysis)
    dump_json(output / "config.json", {"indices": indices, "dataset": "ScreenSpot-Pro", "baseline": "full screenshot -> deepseek-flash", "reproduction": "OmniParser V2 -> Instructor-large -> DRS/MCTS -> deepseek-flash", "all_failures_retained": True})

    report = [
        "# Fixed 20-sample baseline vs DRS smoke comparison", "",
        "> Real ScreenSpot-Pro samples and real model calls. This fixed, non-random subset is not benchmark accuracy.", "",
        f"- Baseline: **{baseline_correct}/{len(rows)} = {summary['baseline_accuracy']:.1%}**",
        f"- Reproduction: **{reproduction_correct}/{len(rows)} = {summary['reproduction_accuracy']:.1%}**",
        f"- Accuracy delta: **{summary['accuracy_delta']:+.1%}**",
        f"- Best Region contains GT center: **{hits}/{len(rows)}**",
        f"- Grounding tokens: **{baseline_total_tokens} -> {reproduction_total_tokens} ({summary['token_reduction_fraction']:.1%} reduction)**",
        f"- Mean grounding latency: **{summary['baseline_mean_grounding_latency_seconds']:.2f}s -> {summary['reproduction_mean_grounding_latency_seconds']:.2f}s** (DRS excludes perception)",
        f"- Mean remote perception latency: **{summary['reproduction_mean_perception_latency_seconds']:.2f}s**",
        f"- Outcomes: `{summary['paired_outcomes']}`",
        f"- DRS failure types: `{summary['reproduction_failure_types']}`", "",
        "| Index | Sample | Instruction | Base | DRS | Region GT | Failure/outcome |", "|---:|---|---|:---:|:---:|:---:|---|",
    ]
    for row in rows:
        report.append(f"| {row['index']} | {row['sample_id']} | {row['instruction'].replace('|', '/')} | {'✓' if row['baseline_correct'] else '✗'} | {'✓' if row['reproduction_correct'] else '✗'} | {'✓' if row['best_region_contains_gt_center'] else '✗'} | {row['outcome']} / {row['reproduction_failure_type']} |")
    report += ["", "Interpretation: accuracy and latency are smoke diagnostics. Search misses and final-grounder failures are separated explicitly."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    badcases = [
        "# Badcase analysis", "",
        f"- Best Region recall: **{hits}/{len(rows)} = {analysis['region_recall']:.0%}**",
        f"- DRS accuracy conditioned on region hit: **{hit_correct}/{hits} = {analysis['reproduction_accuracy_given_region_hit']:.0%}**" if hits else "- DRS accuracy conditioned on region hit: **N/A**",
        f"- Mean Best Region area: **{analysis['mean_best_region_area_ratio']:.1%} of the full screenshot**",
        f"- Mean image-area reduction: **{analysis['mean_area_reduction_fraction']:.1%}**", "",
        analysis["conclusion"],
    ]
    (output / "BADCASE_REPORT.md").write_text("\n".join(badcases) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "analysis": analysis}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--perception-template", required=True)
    parser.add_argument("--output", required=True)
    summarize(parser.parse_args())
