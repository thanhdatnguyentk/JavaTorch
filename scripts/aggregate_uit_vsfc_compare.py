#!/usr/bin/env python3
"""Aggregate UIT-VSFC run summaries across frameworks into CSV and Markdown report."""

import argparse
import csv
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def to_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def candidate_result_roots(primary: Path) -> List[Path]:
    candidates: List[Path] = [primary]
    if primary.as_posix() == "benchmark/results":
        candidates.append(Path("examples") / primary)

    unique: List[Path] = []
    seen = set()
    for p in candidates:
        key = p.resolve().as_posix() if p.exists() else p.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def load_rows(base: Path, framework: str) -> List[Dict[str, str]]:
    root = base / framework / "uit_vsfc_multitask"
    if not root.exists():
        return []

    rows: List[Dict[str, str]] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = run_dir / "run_summary.csv"
        if not summary.exists():
            continue
        try:
            with summary.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["framework"] = framework
                    row["_run_summary_path"] = str(summary.resolve())
                    rows.append(row)
        except Exception:
            continue
    return rows


def weighted_score(row: Dict[str, str]) -> float:
    sent = to_float(row.get("test_sent_macro_f1", "nan"))
    topic = to_float(row.get("test_topic_macro_f1", "nan"))
    alpha = to_float(row.get("alpha", "1"), 1.0)
    beta = to_float(row.get("beta", "1"), 1.0)
    denom = alpha + beta
    if denom <= 0:
        return (sent + topic) / 2.0
    return (alpha / denom) * sent + (beta / denom) * topic


def score_for_sort(row: Dict[str, str]) -> float:
    s = weighted_score(row)
    return s if math.isfinite(s) else float("-inf")


def fmt_float(value: float, digits: int) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _parse_timestamp(value: str) -> float:
    if not value:
        return float("-inf")

    text = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return float("-inf")


def _extract_run_id_ts(run_id: str) -> float:
    if not run_id:
        return float("-inf")
    matches = re.findall(r"(\d{8}_\d{6})", run_id)
    if not matches:
        return float("-inf")
    try:
        return datetime.strptime(matches[-1], "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return float("-inf")


def _recency_key(row: Dict[str, str]) -> float:
    ts = _parse_timestamp(row.get("timestamp", ""))
    if math.isfinite(ts):
        return ts
    return _extract_run_id_ts(row.get("run_id", ""))


def _is_newer(candidate: Dict[str, str], current: Dict[str, str]) -> bool:
    c_ts = _recency_key(candidate)
    cur_ts = _recency_key(current)
    if c_ts != cur_ts:
        return c_ts > cur_ts
    return score_for_sort(candidate) > score_for_sort(current)


def filter_latest_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    latest: Dict[tuple, Dict[str, str]] = {}
    for r in rows:
        key = (
            r.get("framework", ""),
            r.get("model", ""),
            r.get("device", ""),
            r.get("seed", ""),
        )
        current = latest.get(key)
        if current is None or _is_newer(r, current):
            latest[key] = r

    return sorted(
        latest.values(),
        key=lambda x: (x.get("framework", ""), x.get("model", ""), x.get("device", ""), x.get("seed", "")),
    )


def has_required_metrics(row: Dict[str, str]) -> bool:
    sent = to_float(row.get("test_sent_macro_f1", "nan"))
    topic = to_float(row.get("test_topic_macro_f1", "nan"))
    return math.isfinite(sent) and math.isfinite(topic)


def validate_row(row: Dict[str, str]) -> List[str]:
    issues: List[str] = []
    required_text = ["framework", "run_id", "model", "device", "seed"]
    for key in required_text:
        if not str(row.get(key, "")).strip():
            issues.append(f"missing:{key}")

    try:
        int(str(row.get("seed", "")).strip())
    except Exception:
        issues.append("invalid:seed")

    sent = to_float(row.get("test_sent_macro_f1", "nan"))
    topic = to_float(row.get("test_topic_macro_f1", "nan"))
    joint = to_float(row.get("test_joint_exact_match", "nan"))
    p50 = to_float(row.get("inference_p50_ms", "nan"))
    p95 = to_float(row.get("inference_p95_ms", "nan"))
    throughput = to_float(row.get("inference_throughput_sps", "nan"))
    total_train_ms = to_float(row.get("total_train_time_ms", "nan"))

    for name, value in (("test_sent_macro_f1", sent), ("test_topic_macro_f1", topic), ("test_joint_exact_match", joint)):
        if not math.isfinite(value):
            issues.append(f"nan:{name}")
        elif value < 0.0 or value > 1.0:
            issues.append(f"range:{name}")

    if not math.isfinite(p50) or p50 <= 0:
        issues.append("invalid:inference_p50_ms")
    if not math.isfinite(p95) or p95 <= 0:
        issues.append("invalid:inference_p95_ms")
    if math.isfinite(p50) and math.isfinite(p95) and p95 < p50:
        issues.append("invalid:p95_lt_p50")
    if not math.isfinite(throughput) or throughput <= 0:
        issues.append("invalid:inference_throughput_sps")
    if math.isfinite(total_train_ms) and total_train_ms <= 0:
        issues.append("invalid:total_train_time_ms")

    return issues


def normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def apply_composite_score(
    rows: List[Dict[str, str]],
    accuracy_weight: float,
    throughput_weight: float,
    latency_weight: float,
) -> List[Dict[str, str]]:
    weighted_vals = [weighted_score(r) for r in rows]
    throughput_vals = [to_float(r.get("inference_throughput_sps", "nan")) for r in rows]
    latency_vals = [to_float(r.get("inference_p50_ms", "nan")) for r in rows]

    weighted_norm = normalize(weighted_vals)
    throughput_norm = normalize(throughput_vals)
    latency_norm = normalize(latency_vals)

    out_rows: List[Dict[str, str]] = []
    for i, r in enumerate(rows):
        latency_inv = 1.0 - latency_norm[i]
        overall = (
            accuracy_weight * weighted_norm[i]
            + throughput_weight * throughput_norm[i]
            + latency_weight * latency_inv
        )
        row = dict(r)
        row["weighted_test_f1"] = fmt_float(weighted_vals[i], 6)
        row["score_accuracy_norm"] = fmt_float(weighted_norm[i], 6)
        row["score_throughput_norm"] = fmt_float(throughput_norm[i], 6)
        row["score_latency_inv_norm"] = fmt_float(latency_inv, 6)
        row["score_overall"] = fmt_float(overall, 6)
        out_rows.append(row)

    return sorted(out_rows, key=lambda x: to_float(x.get("score_overall", "0"), 0.0), reverse=True)


def write_invalid_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "framework",
        "run_id",
        "model",
        "device",
        "seed",
        "issues",
        "run_summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "framework": r.get("framework", ""),
                    "run_id": r.get("run_id", ""),
                    "model": r.get("model", ""),
                    "device": r.get("device", ""),
                    "seed": r.get("seed", ""),
                    "issues": r.get("_issues", ""),
                    "run_summary_path": r.get("_run_summary_path", ""),
                }
            )


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "framework",
        "run_id",
        "model",
        "device",
        "seed",
        "best_epoch",
        "test_sent_macro_f1",
        "test_topic_macro_f1",
        "test_joint_exact_match",
        "inference_p50_ms",
        "inference_p95_ms",
        "inference_throughput_sps",
        "weighted_test_f1",
        "score_accuracy_norm",
        "score_throughput_norm",
        "score_latency_inv_norm",
        "score_overall",
        "final_model_path",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in headers}
            writer.writerow(out)


def write_markdown(
    path: Path,
    rows: List[Dict[str, str]],
    accuracy_weight: float,
    throughput_weight: float,
    latency_weight: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("## UIT-VSFC Comparison Report\n")

    if not rows:
        lines.append("No valid UIT run_summary artifacts found for frameworks `JavaTorch` or `pytorch`.\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("| Rank | Framework | Run ID | Model | Device | Seed | Weighted Test F1 | p50 ms | Throughput sps | Overall Score |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|")

    for i, r in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {r.get('framework', '')} | {r.get('run_id', '')} | {r.get('model', '')} | {r.get('device', '')} | {r.get('seed', '')} | "
            f"{r.get('weighted_test_f1', '')} | {fmt_float(to_float(r.get('inference_p50_ms', 'nan')), 3)} | "
            f"{fmt_float(to_float(r.get('inference_throughput_sps', 'nan')), 2)} | {r.get('score_overall', '')} |"
        )

    lines.append("")
    lines.append(
        "Scoring rule: "
        f"overall = {accuracy_weight:.3f}*accuracy_norm + {throughput_weight:.3f}*throughput_norm + {latency_weight:.3f}*latency_inv_norm"
    )
    lines.append(
        "Accuracy proxy: weighted_test_f1 = (alpha/(alpha+beta))*test_sent_macro_f1 + (beta/(alpha+beta))*test_topic_macro_f1"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def split_valid_invalid(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    valid: List[Dict[str, str]] = []
    invalid: List[Dict[str, str]] = []
    for r in rows:
        issues = validate_row(r)
        if issues:
            bad = dict(r)
            bad["_issues"] = ";".join(issues)
            invalid.append(bad)
        else:
            valid.append(r)
    return valid, invalid


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate UIT-VSFC benchmark comparisons")
    parser.add_argument("--resultsDir", default="benchmark/results")
    parser.add_argument("--outputDir", default="benchmark/results/compare/uit_vsfc_multitask")
    parser.add_argument(
        "--latestOnly",
        action="store_true",
        help="Keep only the most recent run per (framework, model, device, seed)",
    )
    parser.add_argument(
        "--dropMissingMetrics",
        action="store_true",
        help="Drop rows missing test_sent_macro_f1 or test_topic_macro_f1",
    )
    parser.add_argument(
        "--rejectInvalid",
        action="store_true",
        help="Reject rows that fail sanity checks (recommended for Phase 4).",
    )
    parser.add_argument("--accuracyWeight", type=float, default=0.5)
    parser.add_argument("--throughputWeight", type=float, default=0.3)
    parser.add_argument("--latencyWeight", type=float, default=0.2)
    args = parser.parse_args()

    base = Path(args.resultsDir)
    out = Path(args.outputDir)
    out.mkdir(parents=True, exist_ok=True)

    weights_sum = args.accuracyWeight + args.throughputWeight + args.latencyWeight
    if weights_sum <= 0:
        raise RuntimeError("Weights must sum to > 0")

    accuracy_weight = args.accuracyWeight / weights_sum
    throughput_weight = args.throughputWeight / weights_sum
    latency_weight = args.latencyWeight / weights_sum

    rows: List[Dict[str, str]] = []
    for root in candidate_result_roots(base):
        rows.extend(load_rows(root, "JavaTorch"))
        rows.extend(load_rows(root, "pytorch"))

    if args.latestOnly:
        before = len(rows)
        rows = filter_latest_rows(rows)
        print(f"Filtered latest rows: {before} -> {len(rows)}")

    if args.dropMissingMetrics:
        before = len(rows)
        rows = [r for r in rows if has_required_metrics(r)]
        print(f"Dropped rows with missing metrics: {before} -> {len(rows)}")

    invalid_rows: List[Dict[str, str]] = []
    if args.rejectInvalid:
        before = len(rows)
        rows, invalid_rows = split_valid_invalid(rows)
        print(f"Rejected invalid rows: {before - len(rows)}")
        invalid_csv = out / "invalid_rows.csv"
        write_invalid_rows(invalid_csv, invalid_rows)
        print(f"Wrote: {invalid_csv}")

    scored_rows = apply_composite_score(rows, accuracy_weight, throughput_weight, latency_weight)

    csv_path = out / "comparison.csv"
    md_path = out / "comparison.md"

    write_csv(csv_path, scored_rows)
    write_markdown(md_path, scored_rows, accuracy_weight, throughput_weight, latency_weight)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
