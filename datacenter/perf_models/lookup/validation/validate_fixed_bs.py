"""Validate simulator latency lookup against real benchmarks at fixed batch sizes.

For each BS, compares:
  - Simulator prediction: TTFT(bs, in_tok) + TPOT(bs) × out_tok
  - Actual benchmark: inf_latency from real A100 runs

This validates the core latency model without time.sleep overhead.
"""

import csv
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datacenter.perf_models.lookup.latency import LatencyLookup

BASE = os.path.dirname(os.path.abspath(__file__))

# Generate MBPP merged if needed
mbpp_merged = os.path.join(PROJECT_ROOT, "data", "measurements",
                           "llama3.1-8B", "MBPP", "benchmarks", "merged_benchmark.csv")
if not os.path.exists(mbpp_merged):
    raw_dir = os.path.join(PROJECT_ROOT, "data", "measurements", "llama3.1-8B", "MBPP", "benchmarks", "raw_data")
    files = sorted(glob.glob(os.path.join(raw_dir, "benchmark_mbpp_bs_*.csv")),
                   key=lambda x: int(x.split("_bs_")[1].split(".")[0]))
    all_rows = []
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                all_rows.append(row)
    with open(mbpp_merged, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample_id", "batch_size", "in_tokens",
                                                 "out_tokens", "ttft", "tpot", "inf_latency"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Generated {mbpp_merged} ({len(all_rows)} rows)")

MODELS = {
    "Mixtral 8x7B GovReport (TP=2)": {
        "csv": os.path.join(PROJECT_ROOT, "data", "measurements",
                            "mixtral-8x7B", "GovReport", "benchmarks", "merged_benchmark.csv"),
        "power": os.path.join(PROJECT_ROOT, "data", "measurements",
                              "mixtral-8x7B", "GovReport", "benchmarks", "power.csv"),
    },
    "LLaMA 3.1-8B GovReport (TP=1)": {
        "csv": os.path.join(PROJECT_ROOT, "data", "measurements",
                            "llama3.1-8B", "GovReport", "benchmarks", "merged_benchmark.csv"),
        "power": os.path.join(PROJECT_ROOT, "data", "measurements",
                              "llama3.1-8B", "GovReport", "benchmarks", "power.csv"),
    },
    "LLaMA 3.1-8B MBPP (TP=1)": {
        "csv": mbpp_merged,
        "power": os.path.join(PROJECT_ROOT, "data", "measurements",
                              "llama3.1-8B", "MBPP", "benchmarks", "power.csv"),
    },
}

CLEAN_BS = {1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768}


def load_benchmark_by_bs(csv_path):
    by_bs = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            bs = int(row["batch_size"])
            if bs not in CLEAN_BS:
                continue
            if bs not in by_bs:
                by_bs[bs] = []
            by_bs[bs].append({
                "in_tokens": int(row["in_tokens"]),
                "out_tokens": int(row["out_tokens"]),
                "ttft": float(row["ttft"]),
                "tpot": float(row["tpot"]),
                "inf_latency": float(row["inf_latency"]),
            })
    return by_bs


for model_name, cfg in MODELS.items():
    print(f"\n{'='*80}")
    print(f"  {model_name}")
    print(f"{'='*80}")

    if not os.path.exists(cfg["csv"]):
        print(f"  Skipping: {cfg['csv']} not found")
        continue

    lookup = LatencyLookup(cfg["csv"], cfg["power"])
    bench = load_benchmark_by_bs(cfg["csv"])

    all_bs_list = []
    all_sim = []
    all_actual = []
    all_ttft_sim = []
    all_ttft_actual = []
    all_tpot_sim = []
    all_tpot_actual = []

    print(f"\n{'BS':>4}  {'N':>5}  {'Sim(s)':>10}  {'Actual(s)':>10}  {'Ratio':>7}  "
          f"{'Err%':>7}  {'TTFT_sim':>9}  {'TTFT_act':>9}  {'TTFT_e%':>8}  "
          f"{'TPOT_sim':>9}  {'TPOT_act':>9}  {'TPOT_e%':>8}")
    print("-" * 115)

    for bs in sorted(CLEAN_BS):
        if bs not in bench:
            continue

        samples = bench[bs]
        sim_lats = []
        act_lats = []
        ttft_sims = []
        ttft_acts = []
        tpot_sims = []
        tpot_acts = []

        for s in samples:
            # Simulator prediction
            pred_ttft = lookup.ttft(bs, s["in_tokens"])
            pred_tpot = lookup.tpot(bs)
            pred_lat = pred_ttft + pred_tpot * s["out_tokens"]

            sim_lats.append(pred_lat)
            act_lats.append(s["inf_latency"])
            ttft_sims.append(pred_ttft)
            ttft_acts.append(s["ttft"])
            tpot_sims.append(pred_tpot)
            tpot_acts.append(s["tpot"])

            all_bs_list.append(bs)
            all_sim.append(pred_lat)
            all_actual.append(s["inf_latency"])
            all_ttft_sim.append(pred_ttft)
            all_ttft_actual.append(s["ttft"])
            all_tpot_sim.append(pred_tpot)
            all_tpot_actual.append(s["tpot"])

        sim_med = np.median(sim_lats)
        act_med = np.median(act_lats)
        ratio = sim_med / act_med
        err = (ratio - 1) * 100
        ttft_s_med = np.median(ttft_sims)
        ttft_a_med = np.median(ttft_acts)
        ttft_err = (ttft_s_med / ttft_a_med - 1) * 100 if ttft_a_med > 0 else 0
        tpot_s_med = np.median(tpot_sims)
        tpot_a_med = np.median(tpot_acts)
        tpot_err = (tpot_s_med / tpot_a_med - 1) * 100 if tpot_a_med > 0 else 0

        print(f"{bs:>4}  {len(samples):>5}  {sim_med:>10.3f}  {act_med:>10.3f}  "
              f"{ratio:>7.3f}  {err:>6.1f}%  {ttft_s_med:>9.4f}  {ttft_a_med:>9.4f}  "
              f"{ttft_err:>7.1f}%  {tpot_s_med:>9.6f}  {tpot_a_med:>9.6f}  {tpot_err:>7.1f}%")

    all_sim = np.array(all_sim)
    all_actual = np.array(all_actual)
    all_bs_list = np.array(all_bs_list)
    ratio_all = all_sim / all_actual

    print(f"\nOverall:")
    print(f"  Median ratio:  {np.median(ratio_all):.4f}")
    print(f"  Mean ratio:    {np.mean(ratio_all):.4f}")
    print(f"  Within  5%:    {np.mean(np.abs(ratio_all - 1.0) < 0.05) * 100:.1f}%")
    print(f"  Within 10%:    {np.mean(np.abs(ratio_all - 1.0) < 0.10) * 100:.1f}%")
    print(f"  Within 20%:    {np.mean(np.abs(ratio_all - 1.0) < 0.20) * 100:.1f}%")

    # ── Plots ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: Scatter actual vs sim (colored by BS)
    ax = axes[0, 0]
    unique_bs = sorted(set(all_bs_list))
    cmap = plt.cm.viridis
    for i, bs in enumerate(unique_bs):
        mask = all_bs_list == bs
        c = cmap(i / max(len(unique_bs) - 1, 1))
        ax.scatter(all_actual[mask], all_sim[mask], alpha=0.3, s=10, color=c, label=f"BS={bs}")
    max_val = max(all_actual.max(), all_sim.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="y=x")
    ax.set_xlabel("Actual Latency (s)", fontsize=12)
    ax.set_ylabel("Predicted Latency (s)", fontsize=12)
    ax.set_title("Total Latency: Actual vs Predicted", fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 2: Box plot of ratio per BS
    ax = axes[0, 1]
    ratio_by_bs = {}
    for i, bs in enumerate(all_bs_list):
        if bs not in ratio_by_bs:
            ratio_by_bs[bs] = []
        ratio_by_bs[bs].append(ratio_all[i])
    bs_labels = sorted(ratio_by_bs.keys())
    bp_data = [ratio_by_bs[bs] for bs in bs_labels]
    ax.boxplot(bp_data, tick_labels=[str(int(b)) for b in bs_labels], showfliers=False)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Predicted / Actual", fontsize=12)
    ax.set_title("Prediction Ratio by Batch Size", fontsize=13)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Panel 3: Ratio histogram
    ax = axes[0, 2]
    ax.hist(ratio_all, bins=80, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Perfect")
    ax.axvline(np.median(ratio_all), color="orange", linewidth=1.5,
               label=f"Median={np.median(ratio_all):.3f}")
    ax.set_xlabel("Predicted / Actual", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Ratio Distribution", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 4: TTFT scatter
    ax = axes[1, 0]
    all_ttft_sim_arr = np.array(all_ttft_sim)
    all_ttft_actual_arr = np.array(all_ttft_actual)
    for i, bs in enumerate(unique_bs):
        mask = all_bs_list == bs
        c = cmap(i / max(len(unique_bs) - 1, 1))
        ax.scatter(all_ttft_actual_arr[mask], all_ttft_sim_arr[mask],
                   alpha=0.3, s=10, color=c, label=f"BS={bs}")
    max_ttft = max(all_ttft_actual_arr.max(), all_ttft_sim_arr.max()) * 1.05
    ax.plot([0, max_ttft], [0, max_ttft], "r--", linewidth=1.5)
    ax.set_xlabel("Actual TTFT (s)", fontsize=12)
    ax.set_ylabel("Predicted TTFT (s)", fontsize=12)
    ax.set_title("TTFT: Actual vs Predicted", fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 5: TPOT scatter
    ax = axes[1, 1]
    all_tpot_sim_arr = np.array(all_tpot_sim)
    all_tpot_actual_arr = np.array(all_tpot_actual)
    for i, bs in enumerate(unique_bs):
        mask = all_bs_list == bs
        c = cmap(i / max(len(unique_bs) - 1, 1))
        ax.scatter(all_tpot_actual_arr[mask], all_tpot_sim_arr[mask],
                   alpha=0.3, s=10, color=c, label=f"BS={bs}")
    max_tpot = max(all_tpot_actual_arr.max(), all_tpot_sim_arr.max()) * 1.05
    ax.plot([0, max_tpot], [0, max_tpot], "r--", linewidth=1.5)
    ax.set_xlabel("Actual TPOT (s)", fontsize=12)
    ax.set_ylabel("Predicted TPOT (s)", fontsize=12)
    ax.set_title("TPOT: Actual vs Predicted", fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 6: Median bar chart
    ax = axes[1, 2]
    med_actual = [np.median(all_actual[all_bs_list == bs]) for bs in unique_bs]
    med_sim = [np.median(all_sim[all_bs_list == bs]) for bs in unique_bs]
    x_pos = np.arange(len(unique_bs))
    width = 0.35
    ax.bar(x_pos - width / 2, med_actual, width, label="Actual", color="tab:blue", alpha=0.8)
    ax.bar(x_pos + width / 2, med_sim, width, label="Predicted", color="tab:orange", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(b)) for b in unique_bs], rotation=45)
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Median Latency (s)", fontsize=12)
    ax.set_title("Median Latency per Batch Size", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    safe = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    plt.suptitle(f"Fixed-BS Latency Validation: {model_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(BASE, f"validation_fixed_bs_{safe}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")
