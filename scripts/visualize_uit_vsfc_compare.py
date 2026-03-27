#!/usr/bin/env python3
"""Visualize UIT-VSFC benchmark comparison CSV into PNG charts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def _to_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = dict(r)
            row["weighted_test_f1"] = _to_float(row.get("weighted_test_f1", ""))
            row["test_sent_macro_f1"] = _to_float(row.get("test_sent_macro_f1", ""))
            row["test_topic_macro_f1"] = _to_float(row.get("test_topic_macro_f1", ""))
            row["test_joint_exact_match"] = _to_float(row.get("test_joint_exact_match", ""))
            row["inference_p50_ms"] = _to_float(row.get("inference_p50_ms", ""))
            row["inference_throughput_sps"] = _to_float(row.get("inference_throughput_sps", ""))
            rows.append(row)
    return rows


def filter_rows(rows: List[Dict[str, Any]], seed: str | None) -> List[Dict[str, Any]]:
    if seed is None:
        return rows
    return [r for r in rows if str(r.get("seed", "")).strip() == seed]


def run_label(row: Dict[str, Any]) -> str:
    fw = row.get("framework", "")
    model = row.get("model", "")
    dev = row.get("device", "")
    seed = row.get("seed", "")
    return f"{fw}|{model}|{dev}|s{seed}"


def framework_model_label(row: Dict[str, Any]) -> str:
    return f"{row.get('framework', '')}-{row.get('model', '')}"


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def plot_weighted_f1(rows: List[Dict[str, Any]], out_path: Path) -> None:
    ordered = sorted(rows, key=lambda r: r["weighted_test_f1"], reverse=True)
    labels = [run_label(r) for r in ordered]
    values = [r["weighted_test_f1"] for r in ordered]
    colors = ["#1f77b4" if r.get("framework") == "pytorch" else "#ff7f0e" for r in ordered]

    plt.figure(figsize=(14, 6))
    plt.bar(labels, values, color=colors)
    plt.title("UIT-VSFC Weighted Test F1 (higher is better)")
    plt.ylabel("weighted_test_f1")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, min(1.0, max(values) * 1.12 if values else 1.0))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_quality_breakdown(rows: List[Dict[str, Any]], out_path: Path) -> None:
    ordered = sorted(rows, key=lambda r: r["weighted_test_f1"], reverse=True)
    x = list(range(len(ordered)))
    labels = [run_label(r) for r in ordered]

    sent = [r["test_sent_macro_f1"] for r in ordered]
    topic = [r["test_topic_macro_f1"] for r in ordered]
    joint = [r["test_joint_exact_match"] for r in ordered]

    plt.figure(figsize=(14, 6))
    plt.plot(x, sent, marker="o", label="sent_macro_f1")
    plt.plot(x, topic, marker="o", label="topic_macro_f1")
    plt.plot(x, joint, marker="o", label="joint_exact_match")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("score")
    plt.title("UIT-VSFC Quality Breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_speed_vs_quality(rows: List[Dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[framework_model_label(r)].append(r)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    keys = sorted(grouped.keys())
    for idx, key in enumerate(keys):
        pts = grouped[key]
        xs = [p["inference_p50_ms"] for p in pts]
        ys = [p["weighted_test_f1"] for p in pts]
        plt.scatter(xs, ys, s=85, alpha=0.9, label=key, color=palette[idx % len(palette)])

    plt.xlabel("inference_p50_ms (lower is better)")
    plt.ylabel("weighted_test_f1 (higher is better)")
    plt.title("UIT-VSFC Speed vs Quality")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_throughput(rows: List[Dict[str, Any]], out_path: Path) -> None:
    ordered = sorted(rows, key=lambda r: r["inference_throughput_sps"], reverse=True)
    labels = [run_label(r) for r in ordered]
    values = [r["inference_throughput_sps"] for r in ordered]
    colors = ["#1f77b4" if r.get("framework") == "pytorch" else "#ff7f0e" for r in ordered]

    plt.figure(figsize=(14, 6))
    plt.bar(labels, values, color=colors)
    plt.title("UIT-VSFC Inference Throughput (samples/s, higher is better)")
    plt.ylabel("throughput_sps")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def compute_normalized_scores(
    rows: List[Dict[str, Any]],
    quality_weight: float,
    latency_weight: float,
    throughput_weight: float,
) -> List[Dict[str, Any]]:
    quality_vals = [r["weighted_test_f1"] for r in rows]
    latency_vals = [r["inference_p50_ms"] for r in rows]
    throughput_vals = [r["inference_throughput_sps"] for r in rows]

    qn = _normalize(quality_vals)
    ln = _normalize(latency_vals)
    tn = _normalize(throughput_vals)

    results: List[Dict[str, Any]] = []
    for idx, r in enumerate(rows):
        # Lower latency is better, so invert normalized latency.
        latency_inv = 1.0 - ln[idx]
        overall = quality_weight * qn[idx] + latency_weight * latency_inv + throughput_weight * tn[idx]
        results.append(
            {
                **r,
                "quality_norm": qn[idx],
                "latency_inv_norm": latency_inv,
                "throughput_norm": tn[idx],
                "normalized_overall": overall,
            }
        )

    return sorted(results, key=lambda r: r["normalized_overall"], reverse=True)


def write_normalized_ranking_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    headers = [
        "rank",
        "framework",
        "model",
        "device",
        "seed",
        "run_id",
        "weighted_test_f1",
        "inference_p50_ms",
        "inference_throughput_sps",
        "quality_norm",
        "latency_inv_norm",
        "throughput_norm",
        "normalized_overall",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "framework": row.get("framework", ""),
                    "model": row.get("model", ""),
                    "device": row.get("device", ""),
                    "seed": row.get("seed", ""),
                    "run_id": row.get("run_id", ""),
                    "weighted_test_f1": f"{row['weighted_test_f1']:.6f}",
                    "inference_p50_ms": f"{row['inference_p50_ms']:.6f}",
                    "inference_throughput_sps": f"{row['inference_throughput_sps']:.6f}",
                    "quality_norm": f"{row['quality_norm']:.6f}",
                    "latency_inv_norm": f"{row['latency_inv_norm']:.6f}",
                    "throughput_norm": f"{row['throughput_norm']:.6f}",
                    "normalized_overall": f"{row['normalized_overall']:.6f}",
                }
            )


def plot_normalized_overall(rows: List[Dict[str, Any]], out_path: Path) -> None:
    labels = [run_label(r) for r in rows]
    values = [r["normalized_overall"] for r in rows]
    colors = ["#1f77b4" if r.get("framework") == "pytorch" else "#ff7f0e" for r in rows]

    plt.figure(figsize=(14, 6))
    plt.bar(labels, values, color=colors)
    plt.title("UIT-VSFC Normalized Overall Score (higher is better)")
    plt.ylabel("normalized_overall")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize UIT-VSFC comparison CSV")
    parser.add_argument(
        "--csv",
        default="benchmark/results/compare/uit_vsfc_multitask/comparison.csv",
        help="Path to comparison CSV",
    )
    parser.add_argument(
        "--outDir",
        default="benchmark/results/compare/uit_vsfc_multitask",
        help="Directory to write PNG files",
    )
    parser.add_argument(
        "--seed",
        default="42",
        help="Optional seed filter (set to empty to keep all seeds)",
    )
    parser.add_argument("--qualityWeight", type=float, default=0.6, help="Weight for quality metric")
    parser.add_argument("--latencyWeight", type=float, default=0.3, help="Weight for inverse latency")
    parser.add_argument("--throughputWeight", type=float, default=0.1, help="Weight for throughput")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    seed_filter = args.seed.strip() if args.seed is not None else None
    if seed_filter == "":
        seed_filter = None
    rows = filter_rows(rows, seed_filter)

    if not rows:
        raise RuntimeError("No rows found after filtering. Try passing --seed ''")

    weight_sum = args.qualityWeight + args.latencyWeight + args.throughputWeight
    if weight_sum <= 0:
        raise RuntimeError("Weights must sum to > 0")

    q_w = args.qualityWeight / weight_sum
    l_w = args.latencyWeight / weight_sum
    t_w = args.throughputWeight / weight_sum

    plot_weighted_f1(rows, out_dir / "weighted_test_f1.png")
    plot_quality_breakdown(rows, out_dir / "quality_breakdown.png")
    plot_speed_vs_quality(rows, out_dir / "speed_vs_quality.png")
    plot_throughput(rows, out_dir / "throughput.png")
    normalized_rows = compute_normalized_scores(rows, q_w, l_w, t_w)
    write_normalized_ranking_csv(normalized_rows, out_dir / "normalized_ranking.csv")
    plot_normalized_overall(normalized_rows, out_dir / "normalized_overall.png")

    print(f"Input rows: {len(rows)}")
    print(f"Wrote: {out_dir / 'weighted_test_f1.png'}")
    print(f"Wrote: {out_dir / 'quality_breakdown.png'}")
    print(f"Wrote: {out_dir / 'speed_vs_quality.png'}")
    print(f"Wrote: {out_dir / 'throughput.png'}")
    print(f"Wrote: {out_dir / 'normalized_ranking.csv'}")
    print(f"Wrote: {out_dir / 'normalized_overall.png'}")


if __name__ == "__main__":
    main()
