"""In-process carbon-aware multi-scheduler test.

Creates datacenters across 4 scheduler groups (Europe, US-East,
US-West, Asia), each with real carbon intensity from hourly CSV data.
The number of DCs per group is configurable (1-16, default 4).
Regions are ranked by average CI (cleanest first) within each group.

Compares scheduling policies at different alpha values:
  - Round-robin (global baseline)
  - alpha=1.0 (pure latency)
  - alpha=0.5 (balanced)
  - alpha=0.0 (pure carbon)

Usage:
    python -m tests.test_carbon_aware
    python -m tests.test_carbon_aware --policy all --trace
    python -m tests.test_carbon_aware --dcs-per-group 8 --num-requests 100
"""

import csv
import time
import random
import threading
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import sys
from pathlib import Path

# Run directly (`python scripts/benchmarks/<name>.py`), so the repo root
# has to be on the path before the first datacenter import. Two of these
# files did it and three did not, which meant the same command worked for
# some and raised ModuleNotFoundError for the others.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datacenter.perf_models.lookup.latency import LatencyLookup
from datacenter.meta_dc import Datacenter
from common.carbon import CarbonLoader
from common.network import (
    SCHEDULER_GROUPS,
    get_rtt_ms,
)


# ── Provisioner (polls DCs, pushes snapshots to schedulers) ──────────

class InProcessProvisioner:
    """In-process equivalent of provisioner/server.py.

    Background thread periodically:
      1. Polls every Datacenter for stats (avg_batch_size, energy).
      2. Pushes the snapshot to all registered schedulers.

    Schedulers receive the snapshot via ``push_snapshot()`` and store
    their own local copy, so load info is up to ``poll_period_s`` stale.
    """

    def __init__(
        self,
        all_dcs: Dict[str, Datacenter],
        poll_period_s: float = 0.5,
    ) -> None:
        self._all_dcs = all_dcs
        self.poll_period_s = poll_period_s
        self.epoch = 0
        self._schedulers: List["CarbonAwareScheduler"] = []
        self._stop = threading.Event()
        self._poll()  # initial snapshot (epoch 1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def register_schedulers(self, schedulers: List["CarbonAwareScheduler"]):
        self._schedulers = schedulers

    def _poll(self) -> None:
        """Poll all DCs and push snapshot to schedulers."""
        self.epoch += 1
        snap: Dict[str, float] = {}
        for dc_id, dc in self._all_dcs.items():
            snap[dc_id] = dc.stats()["avg_batch_size"]
        for sched in self._schedulers:
            sched.push_snapshot(self.epoch, snap)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_period_s):
            self._poll()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


# ── Carbon-Aware Scheduler ──────────────────────────────────────────

class CarbonAwareScheduler:
    """A scheduler managing local DCs with carbon-aware cross-region routing.

    Scores all reachable DCs (local + remote via forwarding) using:
        score = alpha * norm_latency + (1 - alpha) * norm_carbon
    Picks the DC with the lowest score. Remote dispatch adds 2*RTT.

    Load information comes from the provisioner's periodic snapshots
    (pushed via ``push_snapshot()``), not from direct DC queries.
    """

    def __init__(
        self,
        sched_id: str,
        group: str,
        local_dcs: Dict[str, Datacenter],
        remote_schedulers: Dict[str, "CarbonAwareScheduler"],
        alpha: float = 0.5,
        rtt_jitter_pct: float = 0.0,
    ) -> None:
        self.sched_id = sched_id
        self.group = group
        self.local_dcs = local_dcs
        self.remote_schedulers = remote_schedulers
        self.alpha = alpha
        self.rtt_jitter_pct = rtt_jitter_pct

        # Snapshot state (pushed by provisioner)
        self._snap_epoch = 0
        self._dc_load: Dict[str, float] = {}
        self._snap_lock = threading.Lock()

    def push_snapshot(self, epoch: int, dc_load: Dict[str, float]) -> None:
        """Receive a snapshot from the provisioner (mirrors PushSnapshot RPC)."""
        with self._snap_lock:
            if epoch > self._snap_epoch:
                self._snap_epoch = epoch
                self._dc_load = dict(dc_load)

    def _get_load(self, dc_id: str) -> float:
        with self._snap_lock:
            return self._dc_load.get(dc_id, 0.0)

    def submit(self, in_tok: int, out_tok: int) -> dict:
        """Route request to best DC. Blocks for simulated duration + RTT."""
        jp = self.rtt_jitter_pct
        # Gather candidates: (region, dc, load, ci, rtt_ms, is_remote, remote_group)
        candidates = []

        for region, dc in self.local_dcs.items():
            rtt = get_rtt_ms(self.group, self.group, jitter_pct=jp)
            load = self._get_load(region)
            ci = dc.carbon_intensity
            candidates.append((region, dc, load, ci, rtt, False, self.group))

        for remote_group, remote_sched in self.remote_schedulers.items():
            rtt = get_rtt_ms(self.group, remote_group, jitter_pct=jp)
            for region, dc in remote_sched.local_dcs.items():
                load = self._get_load(region)
                ci = dc.carbon_intensity
                candidates.append((region, dc, load, ci, rtt, True, remote_group))

        # Normalize and score
        all_latency = []
        all_ci = []
        for _, _, load, ci, rtt, is_remote, _ in candidates:
            rtt_overhead = (2 * rtt / 1000.0) if is_remote else 0.0
            all_latency.append(load + rtt_overhead)
            all_ci.append(ci)

        max_lat = max(all_latency) if all_latency else 1.0
        max_ci = max(all_ci) if all_ci else 1.0
        if max_lat == 0:
            max_lat = 1.0
        if max_ci == 0:
            max_ci = 1.0

        best_score = float("inf")
        best_idx = 0
        for i, (_, _, load, ci, rtt, is_remote, _) in enumerate(candidates):
            rtt_overhead = (2 * rtt / 1000.0) if is_remote else 0.0
            norm_lat = (load + rtt_overhead) / max_lat
            norm_ci = ci / max_ci
            score = self.alpha * norm_lat + (1 - self.alpha) * norm_ci
            if score < best_score:
                best_score = score
                best_idx = i

        region, dc, _, _, _, is_remote, remote_group = candidates[best_idx]

        # Sample fresh RTT for the actual network delay (independent of scoring)
        actual_rtt = get_rtt_ms(self.group, remote_group if is_remote else self.group,
                                jitter_pct=jp)

        t_start = time.monotonic()
        if is_remote:
            time.sleep(actual_rtt / 1000.0)
            result = dc.dispatch(in_tok, out_tok)
            time.sleep(actual_rtt / 1000.0)
            result["rtt_overhead_ms"] = 2 * actual_rtt
            result["forwarded_to"] = remote_group
        else:
            result = dc.dispatch(in_tok, out_tok)
            result["rtt_overhead_ms"] = 0.0
            result["forwarded_to"] = None

        t_end = time.monotonic()
        result["total_latency_ms"] = (t_end - t_start) * 1000.0
        result["scheduler"] = self.sched_id
        result["region"] = region
        result["carbon_intensity"] = dc.carbon_intensity
        return result


class RoundRobinGlobalScheduler:
    """Baseline: round-robin across ALL datacenters, no carbon/RTT awareness."""

    def __init__(self, all_dcs: List[Datacenter]):
        self.dcs = all_dcs
        self._idx = 0
        self._lock = threading.Lock()

    def submit(self, in_tok: int, out_tok: int) -> dict:
        with self._lock:
            dc = self.dcs[self._idx % len(self.dcs)]
            self._idx += 1

        t_start = time.monotonic()
        result = dc.dispatch(in_tok, out_tok)
        t_end = time.monotonic()
        result["total_latency_ms"] = (t_end - t_start) * 1000.0
        result["scheduler"] = "round_robin_global"
        result["region"] = dc.dc_id
        result["rtt_overhead_ms"] = 0.0
        result["forwarded_to"] = None
        result["carbon_intensity"] = dc.carbon_intensity
        return result


# ── Topology Builder ──────────────────────────────────────────────────

def build_datacenters(
    lookup: LatencyLookup,
    carbon_loader: CarbonLoader,
    sim_dt: datetime,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    tp_size: int = 2,
) -> Dict[str, Datacenter]:
    """Create one Datacenter per region with real carbon intensity."""
    dcs = {}
    for region in carbon_loader.get_all_regions():
        ci = carbon_loader.get_ci(region, sim_dt)
        dc = Datacenter(
            dc_id=region,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            lookup=lookup,
            tp_size=tp_size,
            carbon_intensity=ci,
        )
        dcs[region] = dc
    return dcs


def build_carbon_aware_schedulers(
    dcs: Dict[str, Datacenter],
    alpha: float,
    rtt_jitter_pct: float = 0.0,
    poll_period_s: float = 0.5,
) -> Tuple[Dict[str, CarbonAwareScheduler], InProcessProvisioner]:
    """Build one CarbonAwareScheduler per group + a central provisioner.

    The provisioner polls all DCs every ``poll_period_s`` and pushes
    load snapshots to every scheduler (mirrors real gRPC architecture).

    Returns (schedulers_dict, provisioner). Caller should call
    provisioner.stop() when the experiment is done.
    """
    schedulers: Dict[str, CarbonAwareScheduler] = {}

    for group, regions in SCHEDULER_GROUPS.items():
        local_dcs = {r: dcs[r] for r in regions if r in dcs}
        schedulers[group] = CarbonAwareScheduler(
            sched_id=f"sched-{group}",
            group=group,
            local_dcs=local_dcs,
            remote_schedulers={},
            alpha=alpha,
            rtt_jitter_pct=rtt_jitter_pct,
        )

    for group, sched in schedulers.items():
        sched.remote_schedulers = {
            g: s for g, s in schedulers.items() if g != group
        }

    # Start the provisioner — polls DCs and pushes snapshots to schedulers
    provisioner = InProcessProvisioner(dcs, poll_period_s=poll_period_s)
    provisioner.register_schedulers(list(schedulers.values()))

    return schedulers, provisioner


# ── Trace loader ──────────────────────────────────────────────────────

def load_trace(latency_csv: str) -> List[Tuple[int, int]]:
    """Load real ISL/OSL pairs from benchmark CSV (BS=1 rows only).

    Returns list of (in_tokens, out_tokens) tuples.
    """
    pairs = []
    with open(latency_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["batch_size"]) == 1:
                pairs.append((int(row["in_tokens"]), int(row["out_tokens"])))
    return pairs


# ── Experiment Runner ─────────────────────────────────────────────────

def run_experiment(
    submit_fn: Callable[[int, int], dict],
    name: str,
    num_requests: int,
    arrival_rate: float,
    seed: int = 42,
    trace: Optional[List[Tuple[int, int]]] = None,
) -> List[dict]:
    random.seed(seed)
    results = []
    lock = threading.Lock()
    t0 = time.monotonic()

    def worker(req_id, in_tok, out_tok):
        arrival = time.monotonic() - t0
        result = submit_fn(in_tok, out_tok)
        finish = time.monotonic() - t0
        result["req_id"] = req_id
        result["arrival_s"] = round(arrival, 3)
        result["finish_s"] = round(finish, 3)
        result["in_tok"] = in_tok
        result["out_tok"] = out_tok
        with lock:
            results.append(result)

    reqs = []
    if trace is not None:
        # Use real ISL/OSL from trace, sample with replacement if needed
        for i in range(num_requests):
            in_tok, out_tok = trace[i % len(trace)]
            interval = random.expovariate(arrival_rate)
            reqs.append((i + 1, in_tok, out_tok, interval))
    else:
        for i in range(num_requests):
            in_tok = random.randint(4000, 10000)
            out_tok = random.randint(1500, 3000)
            interval = random.expovariate(arrival_rate)
            reqs.append((i + 1, in_tok, out_tok, interval))

    threads = []
    for i, (req_id, in_tok, out_tok, interval) in enumerate(reqs):
        if i > 0:
            time.sleep(interval)
        th = threading.Thread(target=worker, args=(req_id, in_tok, out_tok))
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    return results, t0


# ── Reporting ─────────────────────────────────────────────────────────

def print_report(
    name: str,
    results: List[dict],
    all_dcs: Dict[str, Datacenter],
    num_requests: int,
    arrival_rate: float,
) -> dict:
    results.sort(key=lambda x: x["req_id"])
    latencies = sorted(r["total_latency_ms"] for r in results)
    n = len(latencies)

    print(f"\n{'=' * 70}")
    print(f"  Policy: {name}")
    print(f"  {num_requests} requests, Poisson lambda={arrival_rate}")
    print(f"{'=' * 70}")

    print(f"\n  Latency (total, including RTT overhead):")
    print(f"    Min:  {latencies[0]:>12.1f} ms")
    print(f"    Avg:  {sum(latencies)/n:>12.1f} ms")
    print(f"    P50:  {latencies[n // 2]:>12.1f} ms")
    print(f"    P90:  {latencies[int(n * 0.9)]:>12.1f} ms")
    print(f"    P99:  {latencies[int(n * 0.99)]:>12.1f} ms")
    print(f"    Max:  {latencies[-1]:>12.1f} ms")

    rtt_overheads = [r["rtt_overhead_ms"] for r in results]
    forwarded = [r for r in results if r.get("forwarded_to")]
    print(f"\n  Forwarding:")
    print(f"    Requests forwarded remotely: {len(forwarded)}/{n}")
    if rtt_overheads:
        print(f"    Avg RTT overhead: {sum(rtt_overheads)/n:.1f} ms")

    total_energy_j = 0.0
    total_carbon_g = 0.0
    dc_ids = sorted(set(r.get("dc_id", r.get("region", "")) for r in results))

    print(f"\n  Per-datacenter breakdown:")
    print(f"    {'Region':<20s} {'Reqs':>5} {'CI(g/kWh)':>10} {'Energy(kJ)':>10} {'Carbon(g)':>10}")
    for dc_id in dc_ids:
        dc_results = [r for r in results
                       if r.get("dc_id", r.get("region")) == dc_id]
        if dc_id in all_dcs:
            dc_obj = all_dcs[dc_id]
            energy_j = dc_obj.stats()["total_energy_j"]
            ci = dc_obj.carbon_intensity
            carbon_g = energy_j * ci / 3_600_000.0
            total_energy_j += energy_j
            total_carbon_g += carbon_g
            print(f"    {dc_id:<20s} {len(dc_results):>5} {ci:>10.1f} "
                  f"{energy_j/1000:>10.1f} {carbon_g:>10.1f}")

    print(f"\n  Totals:")
    print(f"    Energy: {total_energy_j:.0f} J  ({total_energy_j/3_600_000:.4f} kWh)")
    print(f"    Carbon: {total_carbon_g:.1f} g CO2eq  ({total_carbon_g/1000:.3f} kg)")

    return {
        "policy": name,
        "avg_ms": sum(latencies) / n,
        "p50_ms": latencies[n // 2],
        "p90_ms": latencies[int(n * 0.9)],
        "p99_ms": latencies[int(n * 0.99)],
        "max_ms": latencies[-1],
        "total_energy_j": total_energy_j,
        "total_carbon_g": total_carbon_g,
        "forwarded_pct": len(forwarded) / n * 100,
    }


def print_comparison(summaries: List[dict]):
    print(f"\n{'=' * 110}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 110}")
    print(f"  {'Policy':<25s} {'Avg(ms)':>10} {'P50(ms)':>10} {'P90(ms)':>10} "
          f"{'P99(ms)':>10} {'Energy(kJ)':>12} {'Carbon(g)':>12} {'Fwd%':>6}")
    print(f"  {'-' * 105}")
    for s in summaries:
        print(f"  {s['policy']:<25s} {s['avg_ms']:>10.0f} {s['p50_ms']:>10.0f} "
              f"{s['p90_ms']:>10.0f} {s['p99_ms']:>10.0f} "
              f"{s['total_energy_j']/1000:>12.1f} {s['total_carbon_g']:>12.1f} "
              f"{s['forwarded_pct']:>6.1f}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Carbon-aware multi-scheduler in-process test",
    )
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--arrival-rate", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-csv",
                        default="datacenter/perf_models/lookup/measured/summaries/MoE_GovReport/merged_benchmark.csv")
    parser.add_argument("--carbon-data",
                        default="carbon_intensity")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--sim-date", default="2024-10-15 06:00",
                        help="Simulated UTC datetime (YYYY-MM-DD HH:MM)")
    parser.add_argument("--alphas", default="1.0,0.5,0.0",
                        help="Comma-separated alpha values for carbon_aware "
                             "policy (e.g. '0.05,0.3,0.7')")
    parser.add_argument("--no-round-robin", action="store_true",
                        help="Skip the round-robin baseline")
    parser.add_argument("--policy", default=None,
                        help="(deprecated) use --alphas instead")
    parser.add_argument("--home-group", default="round_robin_across_groups",
                        choices=["europe", "us_east", "us_west", "asia",
                                 "round_robin_across_groups"])
    parser.add_argument("--trace", action="store_true",
                        help="Use real ISL/OSL from benchmark CSV (BS=1 rows)")
    parser.add_argument("--dcs-per-group", type=int, default=4,
                        help="Number of DCs per scheduler group (1-16, "
                             "picks cleanest-CI regions first)")
    parser.add_argument("--rtt-jitter", type=float, default=0.0,
                        help="RTT jitter as fraction (0.0-1.0). "
                             "E.g. 0.2 = +/-20%% uniform around base RTT")
    parser.add_argument("--poll-period", type=float, default=0.5,
                        help="Provisioner snapshot poll period in seconds "
                             "(controls load info staleness)")
    args = parser.parse_args()

    if not 1 <= args.dcs_per_group <= 16:
        parser.error("--dcs-per-group must be between 1 and 16")

    sim_dt = datetime.strptime(args.sim_date, "%Y-%m-%d %H:%M")

    # Select top-N cleanest regions per group
    active_regions = []
    for group, all_regions in SCHEDULER_GROUPS.items():
        active_regions.extend(all_regions[:args.dcs_per_group])

    lookup = LatencyLookup(args.latency_csv)
    carbon_loader = CarbonLoader(
        args.carbon_data, year=args.year, regions=active_regions,
    )
    print(carbon_loader.summary())

    # Load trace if requested
    trace = None
    if args.trace:
        trace = load_trace(args.latency_csv)
        random.Random(args.seed).shuffle(trace)
        print(f"\nTrace: {len(trace)} real requests from BS=1 "
              f"(ISL: {min(t[0] for t in trace)}-{max(t[0] for t in trace)}, "
              f"OSL: {min(t[1] for t in trace)}-{max(t[1] for t in trace)})")

    gpu_pairs_per_dc = args.num_nodes * (args.num_gpus // args.tp)
    total_regions = len(carbon_loader.get_all_regions())
    total_gpu_pairs = total_regions * gpu_pairs_per_dc
    print(f"\nConfig: {args.dcs_per_group} DCs/group x 4 groups = "
          f"{total_regions} DCs total, "
          f"{args.num_nodes} nodes x {args.num_gpus} GPUs (TP={args.tp}) = "
          f"{gpu_pairs_per_dc} GPU pairs/DC, "
          f"{total_gpu_pairs} GPU pairs total")
    print(f"Simulation time: {args.sim_date} UTC")

    policies = []
    if not args.no_round_robin:
        policies.append(("RoundRobin (global)", None))
    for alpha_str in args.alphas.split(","):
        alpha = float(alpha_str.strip())
        policies.append((f"CarbonAware a={alpha}", alpha))

    summaries = []
    for policy_name, alpha in policies:
        dcs = build_datacenters(
            lookup, carbon_loader, sim_dt,
            num_nodes=args.num_nodes,
            num_gpus_per_node=args.num_gpus,
            tp_size=args.tp,
        )

        provisioner = None
        if alpha is None:
            scheduler = RoundRobinGlobalScheduler(list(dcs.values()))
            submit_fn = scheduler.submit
        else:
            schedulers, provisioner = build_carbon_aware_schedulers(
                dcs, alpha,
                rtt_jitter_pct=args.rtt_jitter,
                poll_period_s=args.poll_period,
            )

            if args.home_group == "round_robin_across_groups":
                sched_list = list(schedulers.values())
                _rr_idx = [0]
                _rr_lock = threading.Lock()

                def make_submit(sl, idx, lk):
                    def submit_fn(in_tok, out_tok):
                        with lk:
                            sched = sl[idx[0] % len(sl)]
                            idx[0] += 1
                        return sched.submit(in_tok, out_tok)
                    return submit_fn

                submit_fn = make_submit(sched_list, _rr_idx, _rr_lock)
            else:
                home_sched = schedulers[args.home_group]
                submit_fn = home_sched.submit

        print(f"\n--- Running: {policy_name} ---")
        results, t0 = run_experiment(
            submit_fn, policy_name,
            args.num_requests, args.arrival_rate, args.seed,
            trace=trace,
        )
        if provisioner:
            provisioner.stop()

        # Compute energy from GPU time series.  The simulation window
        # spans from the first request arrival to the last completion
        # (bottleneck).  All GPUs are evaluated over this same window:
        # active segments use power_w(bs), idle segments use 60W.
        if results:
            sim_start = t0 + min(r["arrival_s"] for r in results)
            sim_end = t0 + max(r["finish_s"] for r in results)
            for dc in dcs.values():
                dc.compute_energy(sim_start, sim_end)

        summary = print_report(
            policy_name, results, dcs,
            args.num_requests, args.arrival_rate,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        print_comparison(summaries)


if __name__ == "__main__":
    main()
