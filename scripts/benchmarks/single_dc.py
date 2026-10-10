"""Local single-datacenter inference simulation test.

Creates one datacenter with a latency backend loaded from benchmark CSV,
sends concurrent requests through it, and compares:
  1. Simulated latency (from inference_engine dispatch, which uses time.sleep)
  2. Analytical prediction (ttft + tpot * out_tokens from the model)
  3. Actual benchmark measurement (from the CSV)

This validates the full pipeline: latency model → GPU sim → dispatch timing.

Usage:
    python -m tests.test_single_dc \
        --csv datacenter/perf_models/lookup/measured/summaries/Llama3.1-8B_GovReport/merged_benchmark.csv \
        --num-requests 20 --concurrency 4

    # With predictor (curve fit) instead of lookup
    python -m tests.test_single_dc \
        --csv datacenter/perf_models/lookup/measured/summaries/Llama3.1-8B_GovReport/merged_benchmark.csv \
        --mode predictor --num-requests 20 --concurrency 4

    # Sweep batch sizes
    python -m tests.test_single_dc \
        --csv datacenter/perf_models/lookup/measured/summaries/Llama3.1-8B_GovReport/merged_benchmark.csv \
        --sweep-bs --num-requests 10
"""

import argparse
import csv
import logging
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, ".")

from datacenter.perf_models.factory import create_latency_backend
from datacenter.meta_dc import Datacenter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
)
log = logging.getLogger("test_single_dc")


def load_samples(csv_path, batch_size=None, max_samples=100):
    """Load samples from benchmark CSV, optionally filtered by batch_size."""
    samples = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            bs = int(row["batch_size"])
            if batch_size is not None and bs != batch_size:
                continue
            samples.append({
                "sample_id": row["sample_id"],
                "batch_size": bs,
                "in_tokens": int(row["in_tokens"]),
                "out_tokens": int(row["out_tokens"]),
                "ttft": float(row["ttft"]),
                "tpot": float(row["tpot"]),
                "inf_latency": float(row["inf_latency"]),
            })
            if len(samples) >= max_samples:
                break
    return samples


def run_concurrent_dispatch(dc, samples, concurrency):
    """Dispatch samples concurrently and collect results."""
    results = []
    lock = threading.Lock()

    def dispatch_one(sample):
        t0 = time.monotonic()
        result = dc.dispatch(sample["in_tokens"], sample["out_tokens"])
        elapsed = time.monotonic() - t0
        with lock:
            results.append({
                "sample_id": sample["sample_id"],
                "in_tokens": sample["in_tokens"],
                "out_tokens": sample["out_tokens"],
                "actual_ttft": sample["ttft"],
                "actual_tpot": sample["tpot"],
                "actual_latency": sample["inf_latency"],
                "sim_service_ms": result["service_ms"],
                "sim_elapsed_s": elapsed,
                "sim_gpu_id": result["gpu_id"],
                "sim_prefill_bs": result.get("prefill_bs", 0),
                "sim_ttft_s": result.get("ttft_s", 0),
            })

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(dispatch_one, s) for s in samples]
        for f in as_completed(futures):
            f.result()  # raise exceptions

    return results


def run_sequential_dispatch(dc, samples):
    """Dispatch samples one at a time (BS=1 validation)."""
    results = []
    for sample in samples:
        t0 = time.monotonic()
        result = dc.dispatch(sample["in_tokens"], sample["out_tokens"])
        elapsed = time.monotonic() - t0
        results.append({
            "sample_id": sample["sample_id"],
            "in_tokens": sample["in_tokens"],
            "out_tokens": sample["out_tokens"],
            "actual_ttft": sample["ttft"],
            "actual_tpot": sample["tpot"],
            "actual_latency": sample["inf_latency"],
            "sim_service_ms": result["service_ms"],
            "sim_elapsed_s": elapsed,
            "sim_gpu_id": result["gpu_id"],
            "sim_prefill_bs": result.get("prefill_bs", 0),
            "sim_ttft_s": result.get("ttft_s", 0),
        })
    return results


def analyze_results(results, backend, label=""):
    """Print comparison of simulated vs actual vs analytical predictions."""
    if not results:
        log.info("No results to analyze")
        return

    log.info("\n%s Results (%d requests):", label, len(results))
    log.info("  %6s  %6s  %6s  | %10s  %10s  %10s  | %10s  %10s",
             "in_tok", "out_tok", "pf_bs",
             "actual(s)", "sim(s)", "analytic(s)",
             "sim_err%", "ana_err%")

    actual_lats = []
    sim_lats = []
    analytic_lats = []

    sim_ttfts = []
    actual_ttfts = []
    analytic_ttfts = []

    for r in sorted(results, key=lambda x: x["in_tokens"]):
        actual_lat = r["actual_latency"]
        sim_lat = r["sim_elapsed_s"]

        # Analytical prediction: ttft(bs=1, in_tokens) + tpot(bs=1) * out_tokens
        # Use prefill_bs from simulation for fair comparison
        pf_bs = max(1, r["sim_prefill_bs"])
        ana_ttft = backend.ttft(pf_bs, r["in_tokens"])
        ana_tpot = backend.tpot(pf_bs)
        ana_lat = ana_ttft + ana_tpot * r["out_tokens"]

        actual_lats.append(actual_lat)
        sim_lats.append(sim_lat)
        analytic_lats.append(ana_lat)

        sim_ttfts.append(r["sim_ttft_s"])
        actual_ttfts.append(r["actual_ttft"])
        analytic_ttfts.append(ana_ttft)

        sim_err = (sim_lat - actual_lat) / actual_lat * 100 if actual_lat > 0 else 0
        ana_err = (ana_lat - actual_lat) / actual_lat * 100 if actual_lat > 0 else 0

        log.info("  %6d  %6d  %6d  | %10.4f  %10.4f  %10.4f  | %9.1f%%  %9.1f%%",
                 r["in_tokens"], r["out_tokens"], pf_bs,
                 actual_lat, sim_lat, ana_lat,
                 sim_err, ana_err)

    # Summary statistics
    actual_lats = np.array(actual_lats)
    sim_lats = np.array(sim_lats)
    analytic_lats = np.array(analytic_lats)

    def report(name, predicted, actual):
        ae = np.abs(predicted - actual)
        nonzero = actual > 0
        mape = np.mean(ae[nonzero] / actual[nonzero]) * 100
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        log.info("  %s:  MAE=%.4f  MAPE=%.1f%%  R²=%.4f",
                 name, ae.mean(), mape, r2)

    log.info("\n--- Summary ---")
    log.info("Total Latency:")
    report("Simulation ", sim_lats, actual_lats)
    report("Analytical ", analytic_lats, actual_lats)

    if actual_ttfts and sim_ttfts:
        log.info("TTFT:")
        report("Simulation ", np.array(sim_ttfts), np.array(actual_ttfts))
        report("Analytical ", np.array(analytic_ttfts), np.array(actual_ttfts))


def main():
    parser = argparse.ArgumentParser(description="Single DC inference simulation test")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to merged_benchmark.csv")
    parser.add_argument("--mode", choices=["lookup", "predictor"], default="lookup",
                        help="Latency backend mode")
    parser.add_argument("--num-requests", type=int, default=20,
                        help="Number of requests to dispatch")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent requests (simulates batch size)")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=2,
                        help="GPUs per node")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--max-bs", type=int, default=0,
                        help="Max batch size per GPU (0=unlimited)")
    parser.add_argument("--filter-bs", type=int, default=None,
                        help="Only use samples from this batch_size in CSV")
    parser.add_argument("--sweep-bs", action="store_true",
                        help="Sweep concurrency levels 1,2,4,8,16,32")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Load latency backend
    log.info("Loading latency backend (%s) from %s", args.mode, args.csv)
    backend = create_latency_backend(
        args.mode, csv_path=args.csv,
    )
    log.info("\n%s", backend.summary())

    # Load samples
    samples = load_samples(args.csv, batch_size=args.filter_bs,
                           max_samples=args.num_requests * 10)
    if not samples:
        log.error("No samples loaded!")
        return

    np.random.shuffle(samples)

    if args.sweep_bs:
        # Sweep different concurrency levels
        for conc in [1, 2, 4, 8, 16, 32]:
            n = min(args.num_requests, len(samples))
            batch = samples[:n]

            dc = Datacenter(
                f"dc-test-c{conc}", args.num_nodes, args.num_gpus,
                backend, tp_size=args.tp_size, max_batch_size=args.max_bs,
            )

            log.info("\n" + "=" * 70)
            log.info("CONCURRENCY=%d  (simulated batch size)", conc)
            log.info("=" * 70)

            if conc == 1:
                results = run_sequential_dispatch(dc, batch)
            else:
                results = run_concurrent_dispatch(dc, batch, conc)

            analyze_results(results, backend, label=f"Concurrency={conc}")
    else:
        # Single run
        n = min(args.num_requests, len(samples))
        batch = samples[:n]

        dc = Datacenter(
            "dc-test", args.num_nodes, args.num_gpus,
            backend, tp_size=args.tp_size, max_batch_size=args.max_bs,
        )

        log.info("DC: nodes=%d  gpus/node=%d  tp=%d  gpu_pairs=%d  max_bs=%s",
                 args.num_nodes, args.num_gpus, args.tp_size,
                 args.num_gpus // args.tp_size,
                 args.max_bs if args.max_bs > 0 else "unlimited")
        log.info("Dispatching %d requests with concurrency=%d\n",
                 n, args.concurrency)

        if args.concurrency == 1:
            results = run_sequential_dispatch(dc, batch)
        else:
            results = run_concurrent_dispatch(dc, batch, args.concurrency)

        analyze_results(results, backend,
                        label=f"Concurrency={args.concurrency}")

        # Energy report
        stats = dc.stats()
        log.info("\n--- Energy ---")
        log.info("Total energy: %.2f J  (%.6f kWh)",
                 stats["total_energy_j"], stats["total_energy_kwh"])


if __name__ == "__main__":
    main()
