"""In-process multi-datacenter test with load-aware routing.

Compares two scheduling strategies:
  1. Round-robin across datacenters (baseline)
  2. Least-loaded: pick the DC with lowest avg_batch_size (latency-critical)

Usage:
    python -m tests.test_multi_dc
"""

import time
import random
import threading
import sys
from pathlib import Path

# Run directly (`python scripts/benchmarks/<name>.py`), so the repo root
# has to be on the path before the first datacenter import. Two of these
# files did it and three did not, which meant the same command worked for
# some and raised ModuleNotFoundError for the others.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datacenter.perf_models.lookup.latency import LatencyLookup
from datacenter.meta_dc import Datacenter

# ── Scheduler policies ──────────────────────────────────────────────────
class RoundRobinScheduler:
    """Evenly distribute requests across datacenters."""
    def __init__(self, datacenters: list[Datacenter]):
        self.dcs = datacenters
        self._idx = 0
        self._lock = threading.Lock()

    def pick(self) -> Datacenter:
        with self._lock:
            dc = self.dcs[self._idx % len(self.dcs)]
            self._idx += 1
        return dc


class LeastLoadedScheduler:
    """Pick the datacenter with the lowest avg_batch_size (load proxy)."""
    def __init__(self, datacenters: list[Datacenter]):
        self.dcs = datacenters

    def pick(self) -> Datacenter:
        best = None
        best_load = float("inf")
        for dc in self.dcs:
            load = dc.stats()["avg_batch_size"]
            if load < best_load:
                best_load = load
                best = dc
        return best


# ── Test runner ─────────────────────────────────────────────────────────
def run_experiment(
    scheduler,
    scheduler_name: str,
    num_requests: int,
    arrival_rate: float,
    seed: int = 42,
):
    random.seed(seed)
    results = []
    lock = threading.Lock()
    t0 = time.monotonic()

    def submit(req_id, in_tok, out_tok):
        dc = scheduler.pick()
        arrival = time.monotonic() - t0
        result = dc.dispatch(in_tok, out_tok)
        finish = time.monotonic() - t0
        with lock:
            results.append({
                "req_id": req_id,
                "dc_id": result["dc_id"],
                "gpu": result["gpu_id"],
                "arrival_s": round(arrival, 3),
                "finish_s": round(finish, 3),
                "service_ms": round(result["service_ms"], 1),
                "in_tok": in_tok,
                "out_tok": out_tok,
            })

    # Generate requests
    reqs = []
    for i in range(num_requests):
        in_tok = random.randint(4000, 10000)
        out_tok = random.randint(1500, 3000)
        interval = random.expovariate(arrival_rate)
        reqs.append((i + 1, in_tok, out_tok, interval))

    # Submit with Poisson timing
    threads = []
    for i, (req_id, in_tok, out_tok, interval) in enumerate(reqs):
        if i > 0:
            time.sleep(interval)
        th = threading.Thread(target=submit, args=(req_id, in_tok, out_tok))
        th.start()
        threads.append(th)

    submit_done = time.monotonic() - t0
    for th in threads:
        th.join()
    total_time = time.monotonic() - t0

    # Analyze
    results.sort(key=lambda x: x["req_id"])
    services = sorted(r["service_ms"] for r in results)
    n = len(services)

    print(f"\n{'=' * 60}")
    print(f"  Scheduler: {scheduler_name}")
    print(f"  {num_requests} requests, Poisson λ={arrival_rate}, seed={seed}")
    print(f"  Submitted by t={submit_done:.1f}s, all done at t={total_time:.1f}s")
    print(f"{'=' * 60}")

    print(f"\n  Latency (service_ms):")
    print(f"    Min:  {services[0]:>10.1f} ms")
    print(f"    Avg:  {sum(services)/n:>10.1f} ms")
    print(f"    P50:  {services[n // 2]:>10.1f} ms")
    print(f"    P90:  {services[int(n * 0.9)]:>10.1f} ms")
    print(f"    P99:  {services[int(n * 0.99)]:>10.1f} ms")
    print(f"    Max:  {services[-1]:>10.1f} ms")

    # Per-DC breakdown
    dc_ids = sorted(set(r["dc_id"] for r in results))
    print(f"\n  Per-datacenter:")
    total_energy = 0.0
    for dc_id in dc_ids:
        dc_results = [r for r in results if r["dc_id"] == dc_id]
        dc_obj = next(dc for dc in scheduler.dcs if dc.dc_id == dc_id)
        energy = dc_obj.stats()["total_energy_j"]
        total_energy += energy
        print(f"    {dc_id}: {len(dc_results):>3} reqs  energy={energy:>10.0f} J")

    print(f"\n  Total energy: {total_energy:.0f} J  ({total_energy / 3_600_000:.4f} kWh)")

    return {
        "scheduler": scheduler_name,
        "avg_ms": sum(services) / n,
        "p50_ms": services[n // 2],
        "p99_ms": services[int(n * 0.99)],
        "max_ms": services[-1],
        "total_energy_j": total_energy,
        "wall_time_s": total_time,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="In-process multi-datacenter scheduling test",
    )
    parser.add_argument("--num-dcs", type=int, default=3,
                        help="Number of datacenters")
    parser.add_argument("--num-nodes", type=int, default=1,
                        help="Nodes per datacenter")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Physical GPUs per node")
    parser.add_argument("--tp", type=int, default=2,
                        help="Tensor-parallel size")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Total requests to send")
    parser.add_argument("--arrival-rate", type=float, default=2.0,
                        help="Poisson arrival rate (requests/sec)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", default="both",
                        choices=["round_robin", "least_loaded", "both"],
                        help="Scheduling policy to test")
    parser.add_argument("--latency-csv", default="datacenter/perf_models/lookup/measured/summaries/MoE_GovReport/merged_benchmark.csv")
    args = parser.parse_args()

    lookup = LatencyLookup(args.latency_csv)

    gpu_pairs = args.num_dcs * args.num_nodes * (args.num_gpus // args.tp)
    print(f"Config: {args.num_dcs} DCs × {args.num_nodes} nodes × "
          f"{args.num_gpus} GPUs (TP={args.tp}) = {gpu_pairs} GPU pairs total")

    policies = []
    if args.policy in ("round_robin", "both"):
        policies.append(("round_robin", RoundRobinScheduler))
    if args.policy in ("least_loaded", "both"):
        policies.append(("least_loaded", LeastLoadedScheduler))

    summaries = []
    for policy_name, scheduler_cls in policies:
        dcs = [
            Datacenter(f"dc-{i}", num_nodes=args.num_nodes,
                       num_gpus_per_node=args.num_gpus,
                       lookup=lookup, tp_size=args.tp)
            for i in range(args.num_dcs)
        ]
        scheduler = scheduler_cls(dcs)
        summary = run_experiment(
            scheduler, policy_name, args.num_requests,
            args.arrival_rate, args.seed,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        print(f"\n{'=' * 60}")
        print(f"  COMPARISON")
        print(f"{'=' * 60}")
        print(f"  {'Policy':<16} {'Avg':>10} {'P50':>10} {'P99':>10} {'Max':>10} {'Energy(kJ)':>12} {'Wall(s)':>8}")
        for s in summaries:
            print(f"  {s['scheduler']:<16} {s['avg_ms']:>10.0f} {s['p50_ms']:>10.0f} "
                  f"{s['p99_ms']:>10.0f} {s['max_ms']:>10.0f} "
                  f"{s['total_energy_j']/1000:>12.1f} {s['wall_time_s']:>8.1f}")

if __name__ == "__main__":
    main()
