"""Validate simulator latency with continuous batching.

BS=1: Analytical (no sleep needed, no batching effect)
BS>=2: Full GPU sim with time.sleep, parallelized across multiple GPU instances.
       Each GPU runs one batch of BS requests concurrently.
       Continuous batching: earlier-finishing requests leave, shrinking the batch.

Tests all samples from benchmark CSV at each batch size.
"""

import csv
import glob
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datacenter.inference_engine import GPU
from datacenter.perf_models.lookup.latency import LatencyLookup

BASE = os.path.dirname(os.path.abspath(__file__))

# Ensure merged benchmarks exist
for subdir, pattern in [(os.path.join("llama3.1-8B", "MBPP"), "benchmark_mbpp_bs_*.csv")]:
    merged = os.path.join(PROJECT_ROOT, "data", "measurements", subdir, "benchmarks", "merged_benchmark.csv")
    if not os.path.exists(merged):
        raw_dir = os.path.join(PROJECT_ROOT, "data", "measurements", subdir, "benchmarks", "raw_data")
        files = sorted(glob.glob(os.path.join(raw_dir, pattern)),
                       key=lambda x: int(x.split("_bs_")[1].split(".")[0]))
        rows = []
        for f in files:
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    rows.append(row)
        with open(merged, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["sample_id", "batch_size", "in_tokens",
                                                "out_tokens", "ttft", "tpot", "inf_latency"])
            w.writeheader()
            w.writerows(rows)
        print(f"Generated {merged} ({len(rows)} rows)")

MODELS = {
    "LLaMA 3.1-8B MBPP (TP=1)": {
        "csv": os.path.join(PROJECT_ROOT, "data", "measurements",
                            "llama3.1-8B", "MBPP", "benchmarks", "merged_benchmark.csv"),
        "power": os.path.join(PROJECT_ROOT, "data", "measurements",
                              "llama3.1-8B", "MBPP", "benchmarks", "power.csv"),
    },
    "LLaMA 3.1-8B GovReport (TP=1)": {
        "csv": os.path.join(PROJECT_ROOT, "data", "measurements",
                            "llama3.1-8B", "GovReport", "benchmarks", "merged_benchmark.csv"),
        "power": os.path.join(PROJECT_ROOT, "data", "measurements",
                              "llama3.1-8B", "GovReport", "benchmarks", "power.csv"),
    },
    "Mixtral 8x7B GovReport (TP=2)": {
        "csv": os.path.join(PROJECT_ROOT, "data", "measurements",
                            "mixtral-8x7B", "GovReport", "benchmarks", "merged_benchmark.csv"),
        "power": os.path.join(PROJECT_ROOT, "data", "measurements",
                              "mixtral-8x7B", "GovReport", "benchmarks", "power.csv"),
    },
}

CLEAN_BS = {1, 2, 4, 8, 16, 32, 64, 128, 256}
NUM_PARALLEL_GPUS = 32  # run this many batches concurrently


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


def run_batch_on_gpu(lookup, batch, bs):
    """Run one batch of BS requests on a fresh GPU. Returns list of (sim_svc_s, actual_lat, prefill_bs)."""
    gpu = GPU(gpu_id="val", lookup=lookup, max_batch_size=bs * 2)
    results = [None] * len(batch)

    def worker(idx, s):
        r = gpu.process(s["in_tokens"], s["out_tokens"])
        results[idx] = (r["service_ms"] / 1000.0, s["inf_latency"], r.get("prefill_bs", 0))

    threads = [threading.Thread(target=worker, args=(i, s)) for i, s in enumerate(batch)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def analytical_bs1(lookup, samples):
    """BS=1: no continuous batching, compute analytically."""
    results = []
    for s in samples:
        pred = lookup.ttft(1, s["in_tokens"]) + lookup.tpot(1) * s["out_tokens"]
        results.append((pred, s["inf_latency"], 1))
    return results


total_start = time.time()

for model_name, cfg in MODELS.items():
    print(f"\n{'='*80}")
    print(f"  {model_name}")
    print(f"{'='*80}", flush=True)

    if not os.path.exists(cfg["csv"]):
        print(f"  Skipping: {cfg['csv']} not found")
        continue

    lookup = LatencyLookup(cfg["csv"], cfg["power"])
    bench = load_benchmark_by_bs(cfg["csv"])

    all_bs_list = []
    all_sim = []
    all_actual = []

    print(f"\n{'BS':>4}  {'N':>5}  {'Batches':>8}  {'Sim_med(s)':>11}  {'Act_med(s)':>11}  "
          f"{'Ratio':>7}  {'Err%':>7}  {'Elapsed':>8}")
    print("-" * 80, flush=True)

    for bs in sorted(CLEAN_BS):
        if bs not in bench:
            continue

        samples = bench[bs]
        t0 = time.time()

        if bs == 1:
            # Analytical, no sleep needed
            results = analytical_bs1(lookup, samples)
            n_batches = len(samples)
        else:
            # Split samples into batches of size BS
            batches = []
            for i in range(0, len(samples), bs):
                batch = samples[i:i + bs]
                if len(batch) == bs:
                    batches.append(batch)
            n_batches = len(batches)

            # Run batches in parallel across multiple GPUs
            results = []
            with ThreadPoolExecutor(max_workers=NUM_PARALLEL_GPUS) as executor:
                futures = {executor.submit(run_batch_on_gpu, lookup, batch, bs): batch
                           for batch in batches}
                for future in as_completed(futures):
                    batch_results = future.result()
                    results.extend(batch_results)

        elapsed = time.time() - t0

        sim_vals = [r[0] for r in results]
        act_vals = [r[1] for r in results]

        for s, a in zip(sim_vals, act_vals):
            all_bs_list.append(bs)
            all_sim.append(s)
            all_actual.append(a)

        sim_med = np.median(sim_vals)
        act_med = np.median(act_vals)
        ratio = sim_med / act_med
        err = (ratio - 1) * 100

        print(f"{bs:>4}  {len(results):>5}  {n_batches:>8}  {sim_med:>11.3f}  {act_med:>11.3f}  "
              f"{ratio:>7.3f}  {err:>6.1f}%  {elapsed:>7.1f}s", flush=True)

    if not all_sim:
        continue

    all_sim = np.array(all_sim)
    all_actual = np.array(all_actual)
    all_bs_list = np.array(all_bs_list)
    ratio_all = all_sim / all_actual

    print(f"\nOverall:")
    print(f"  Median ratio:  {np.median(ratio_all):.4f}")
    print(f"  Within  5%:    {np.mean(np.abs(ratio_all - 1.0) < 0.05) * 100:.1f}%")
    print(f"  Within 10%:    {np.mean(np.abs(ratio_all - 1.0) < 0.10) * 100:.1f}%")
    print(f"  Within 20%:    {np.mean(np.abs(ratio_all - 1.0) < 0.20) * 100:.1f}%", flush=True)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    unique_bs = sorted(set(all_bs_list))
    cmap = plt.cm.viridis

    # Panel 1: Scatter
    ax = axes[0]
    for i, bs in enumerate(unique_bs):
        mask = all_bs_list == bs
        c = cmap(i / max(len(unique_bs) - 1, 1))
        ax.scatter(all_actual[mask], all_sim[mask], alpha=0.3, s=10, color=c, label=f"BS={bs}")
    max_val = max(all_actual.max(), all_sim.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="y=x")
    ax.set_xlabel("Actual Latency (s)", fontsize=12)
    ax.set_ylabel("Simulator Latency (s)", fontsize=12)
    ax.set_title("Simulator vs Actual", fontsize=13)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Box plot
    ax = axes[1]
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
    ax.set_ylabel("Simulator / Actual", fontsize=12)
    ax.set_title("Latency Ratio by Batch Size", fontsize=13)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Panel 3: Median bar chart
    ax = axes[2]
    med_act = [np.median(all_actual[all_bs_list == bs]) for bs in unique_bs]
    med_sim = [np.median(all_sim[all_bs_list == bs]) for bs in unique_bs]
    x_pos = np.arange(len(unique_bs))
    width = 0.35
    ax.bar(x_pos - width / 2, med_act, width, label="Actual", color="tab:blue", alpha=0.8)
    ax.bar(x_pos + width / 2, med_sim, width, label="Simulator", color="tab:orange", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(b)) for b in unique_bs], rotation=45)
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Median Latency (s)", fontsize=12)
    ax.set_title("Median Latency: Actual vs Simulator", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    safe = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    plt.suptitle(f"Simulation Validation (Continuous Batching): {model_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(BASE, f"validation_sim_{safe}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}", flush=True)

print(f"\nTotal elapsed: {time.time() - total_start:.0f}s")
