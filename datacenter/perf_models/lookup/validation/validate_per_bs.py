"""Validate simulator latency lookup against raw benchmark data per batch size.

For each model, loads the benchmark CSV and the LatencyLookup, then compares:
  - Benchmark: actual TTFT, TPOT, inf_latency per sample
  - Simulator: lookup.ttft(bs, in_tok) + lookup.tpot(bs) * out_tok

This validates the interpolation/averaging logic without running the full simulator.
"""

import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path for datacenter imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datacenter.perf_models.lookup.latency import LatencyLookup

BASE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "LLaMA 3.1-8B (TP=1)": {
        "csv": os.path.join(PROJECT_ROOT, "data", "measurements",
                            "llama3.1-8B", "GovReport", "benchmarks", "merged_benchmark.csv"),
        "power": os.path.join(PROJECT_ROOT, "data", "measurements",
                              "llama3.1-8B", "GovReport", "benchmarks", "power.csv"),
    },
    "Mixtral 8x7B (TP=2)": {
        "csv": os.path.join(PROJECT_ROOT, "data", "measurements",
                            "mixtral-8x7B", "GovReport", "benchmarks", "merged_benchmark.csv"),
        "power": os.path.join(PROJECT_ROOT, "data", "measurements",
                              "mixtral-8x7B", "GovReport", "benchmarks", "power.csv"),
    },
}


def load_benchmark(csv_path):
    """Load raw benchmark samples grouped by batch size."""
    by_bs = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bs = int(row["batch_size"])
            sample = {
                "in_tokens": int(row["in_tokens"]),
                "out_tokens": int(row["out_tokens"]),
                "ttft": float(row["ttft"]),
                "tpot": float(row["tpot"]),
                "inf_latency": float(row["inf_latency"]),
            }
            if bs not in by_bs:
                by_bs[bs] = []
            by_bs[bs].append(sample)
    return by_bs


for model_name, cfg in MODELS.items():
    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"{'='*70}")

    # Load lookup (what the simulator uses)
    lookup = LatencyLookup(cfg["csv"], cfg["power"])
    print(lookup.summary())
    print()

    # Load raw benchmark data
    bench = load_benchmark(cfg["csv"])
    batch_sizes = sorted(bench.keys())

    # Per-BS comparison
    all_bs = []
    all_actual = []
    all_predicted = []
    all_ttft_actual = []
    all_ttft_predicted = []
    all_tpot_actual = []
    all_tpot_predicted = []

    print(f"{'BS':>5}  {'N':>5}  {'Actual(s)':>10}  {'Predicted(s)':>12}  "
          f"{'Error%':>8}  {'TTFT_err%':>10}  {'TPOT_err%':>10}")
    print("-" * 75)

    for bs in batch_sizes:
        samples = bench[bs]
        n = len(samples)

        actual_lats = []
        predicted_lats = []
        ttft_errors = []
        tpot_errors = []

        for s in samples:
            # Actual from benchmark
            actual = s["inf_latency"]

            # Predicted using simulator lookup
            pred_ttft = lookup.ttft(bs, s["in_tokens"])
            pred_tpot = lookup.tpot(bs)
            predicted = pred_ttft + pred_tpot * s["out_tokens"]

            actual_lats.append(actual)
            predicted_lats.append(predicted)

            all_bs.append(bs)
            all_actual.append(actual)
            all_predicted.append(predicted)

            # TTFT comparison
            all_ttft_actual.append(s["ttft"])
            all_ttft_predicted.append(pred_ttft)
            if s["ttft"] > 0:
                ttft_errors.append((pred_ttft - s["ttft"]) / s["ttft"] * 100)

            # TPOT comparison
            all_tpot_actual.append(s["tpot"])
            all_tpot_predicted.append(pred_tpot)
            if s["tpot"] > 0:
                tpot_errors.append((pred_tpot - s["tpot"]) / s["tpot"] * 100)

        actual_med = np.median(actual_lats)
        pred_med = np.median(predicted_lats)
        error_pct = (pred_med - actual_med) / actual_med * 100
        ttft_err = np.median(ttft_errors) if ttft_errors else 0
        tpot_err = np.median(tpot_errors) if tpot_errors else 0

        print(f"{bs:>5}  {n:>5}  {actual_med:>10.3f}  {pred_med:>12.3f}  "
              f"{error_pct:>7.1f}%  {ttft_err:>9.1f}%  {tpot_err:>9.1f}%")

    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    all_bs = np.array(all_bs)
    all_ttft_actual = np.array(all_ttft_actual)
    all_ttft_predicted = np.array(all_ttft_predicted)
    all_tpot_actual = np.array(all_tpot_actual)
    all_tpot_predicted = np.array(all_tpot_predicted)

    # Overall stats
    ratio = all_predicted / all_actual
    print(f"\nOverall Predicted/Actual ratio:")
    print(f"  Mean:   {np.mean(ratio):.4f}")
    print(f"  Median: {np.median(ratio):.4f}")
    print(f"  Std:    {np.std(ratio):.4f}")
    pct10 = np.mean(np.abs(ratio - 1.0) < 0.10) * 100
    pct20 = np.mean(np.abs(ratio - 1.0) < 0.20) * 100
    print(f"  Within 10%: {pct10:.1f}%")
    print(f"  Within 20%: {pct20:.1f}%")

    # ── Plots ──
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: Total latency
    # Panel 1: Scatter actual vs predicted
    ax = axes[0, 0]
    scatter_colors = plt.cm.viridis(np.log2(all_bs) / np.log2(all_bs.max()))
    ax.scatter(all_actual, all_predicted, alpha=0.3, s=8, c=scatter_colors)
    max_val = max(all_actual.max(), all_predicted.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="y=x")
    ax.set_xlabel("Actual Latency (s)", fontsize=12)
    ax.set_ylabel("Predicted Latency (s)", fontsize=12)
    ax.set_title("Total Latency: Actual vs Predicted", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Error % per BS (box plot)
    ax = axes[0, 1]
    error_by_bs = {}
    for i, bs in enumerate(all_bs):
        if bs not in error_by_bs:
            error_by_bs[bs] = []
        error_by_bs[bs].append((all_predicted[i] / all_actual[i] - 1) * 100)

    bs_labels = sorted(error_by_bs.keys())
    bp_data = [error_by_bs[bs] for bs in bs_labels]
    ax.boxplot(bp_data, labels=[str(b) for b in bs_labels], showfliers=False)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Error %", fontsize=12)
    ax.set_title("Prediction Error by Batch Size", fontsize=13)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Panel 3: Ratio histogram
    ax = axes[0, 2]
    ax.hist(ratio, bins=80, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Perfect")
    ax.axvline(np.median(ratio), color="orange", linewidth=1.5,
               label=f"Median={np.median(ratio):.3f}")
    ax.set_xlabel("Predicted / Actual", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Ratio Distribution", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Row 2: TTFT and TPOT
    # Panel 4: TTFT scatter
    ax = axes[1, 0]
    ax.scatter(all_ttft_actual, all_ttft_predicted, alpha=0.3, s=8, c=scatter_colors)
    max_ttft = max(all_ttft_actual.max(), all_ttft_predicted.max()) * 1.05
    ax.plot([0, max_ttft], [0, max_ttft], "r--", linewidth=1.5, label="y=x")
    ax.set_xlabel("Actual TTFT (s)", fontsize=12)
    ax.set_ylabel("Predicted TTFT (s)", fontsize=12)
    ax.set_title("TTFT: Actual vs Predicted", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 5: TPOT scatter
    ax = axes[1, 1]
    ax.scatter(all_tpot_actual, all_tpot_predicted, alpha=0.3, s=8, c=scatter_colors)
    max_tpot = max(all_tpot_actual.max(), all_tpot_predicted.max()) * 1.05
    ax.plot([0, max_tpot], [0, max_tpot], "r--", linewidth=1.5, label="y=x")
    ax.set_xlabel("Actual TPOT (s)", fontsize=12)
    ax.set_ylabel("Predicted TPOT (s)", fontsize=12)
    ax.set_title("TPOT: Actual vs Predicted", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 6: Median latency per BS comparison
    ax = axes[1, 2]
    med_actual = [np.median([s["inf_latency"] for s in bench[bs]]) for bs in batch_sizes]
    med_predicted = []
    for bs in batch_sizes:
        preds = []
        for s in bench[bs]:
            p = lookup.ttft(bs, s["in_tokens"]) + lookup.tpot(bs) * s["out_tokens"]
            preds.append(p)
        med_predicted.append(np.median(preds))

    x_pos = np.arange(len(batch_sizes))
    width = 0.35
    ax.bar(x_pos - width/2, med_actual, width, label="Actual", color="tab:blue", alpha=0.8)
    ax.bar(x_pos + width/2, med_predicted, width, label="Predicted", color="tab:orange", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(bs) for bs in batch_sizes], rotation=45)
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Median Latency (s)", fontsize=12)
    ax.set_title("Median Latency per Batch Size", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Latency Validation: {model_name}", fontsize=15, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(BASE, f"validation_{safe_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")
