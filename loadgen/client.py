"""Load generator for the distributed AI inference system.

Generates inference requests and distributes them across schedulers in
round-robin fashion.  Supports two workload sources:

  - **random**: synthetic token counts from uniform distributions.
  - **trace**: real requests replayed from CSV log files produced by
    actual inference runs.

Arrival timing is independently configurable (uniform or Poisson).
"""

import argparse
import asyncio
import csv
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List

import grpc

import inference_pb2
import inference_pb2_grpc

from common.discovery import discover_schedulers
from loadgen.arrival import make_arrival_generator
from common.stream_mux import build_scheduler_streams
from loadgen.trace import InferenceRecord, load_trace, make_trace_generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [loadgen] %(levelname)s  %(message)s",
)
log = logging.getLogger("loadgen")


def build_scheduler_stubs(endpoints: Dict[str, str]):
    """Async channels, one per scheduler. Returns (stubs, channels).

    grpc.aio, not grpc: a request is a coroutine on one event loop, not a
    thread. That is the whole point -- see the note on the arrival loop in
    run() for what the thread-per-request model measured.

    Channels must be created inside the running loop, so this is called from
    the coroutine rather than from main().
    """
    stubs, channels = {}, []
    for sid, addr in endpoints.items():
        channel = grpc.aio.insecure_channel(addr)
        channels.append(channel)
        stubs[sid] = inference_pb2_grpc.SchedulerServiceStub(channel)
    return stubs, channels




def _outbox_of(stub, req_id: str) -> float:
    """Outbox wait for this request, or 0.0 from a stub that has no outbox.

    Popped rather than read: the entry is per-request and must not outlive the
    row it goes into.
    """
    pop = getattr(stub, "pop_outbox_ms", None)
    return pop(req_id) if pop is not None else 0.0


async def send_one_request(
    stub: inference_pb2_grpc.SchedulerServiceStub,
    sched_id: str,
    req_num: int,
    record: InferenceRecord,
    verbosity: int = 0,
    t_run0: float = 0.0,
    arrival_s: float = 0.0,
    in_flight: int = 0,
    req_prefix: str = "",
) -> dict:
    # This id now goes on the wire. It used to live only in the client's own
    # CSV while the scheduler minted its own counter, so a row here and a row
    # in DECISION_LOG could not be matched -- and with the trace wrapping,
    # several identical (in_tok, out_tok, dataset) requests are in flight at
    # once, so there was nothing else to match on either.
    #
    # req_prefix is what keeps that true once a run is sharded across several
    # generator processes: each counts from 1, so without it every shard mints
    # req1 and the decision logs cannot be joined. Empty for an unsharded run,
    # which is every run before this existed.
    req_id = f"{req_prefix}req{req_num}"
    send_time = datetime.now(timezone.utc)
    t0 = time.monotonic()
    # Absolute timing, not just durations. Three questions have needed it today
    # and none could be answered from the row: what the load looked like over a
    # sub-window of the run, whether the worker pool made a request wait before
    # it was ever sent, and what the arrival process actually delivered.
    #
    # arrival_s is when the schedule said this request should go; send_s is when
    # the RPC was actually issued. The gap between them is invisible in wall_ms
    # and is exactly the signature of an open-loop generator that has quietly
    # become closed-loop.
    #
    # It still measures that under asyncio, but it measures a different cause:
    # there is no worker pool to queue behind, so the gap is EVENT-LOOP LAG --
    # how late the loop got to this coroutine. It should sit at ~0. If it grows,
    # the single loop is saturated and the next step is sharding across
    # processes, not threads. Keep reading it: this column is what caught the
    # thread model, and an instrument that stops reporting saturation is worse
    # than none.
    send_s = t0 - t_run0
    try:
        resp = await stub.SubmitRequest(
            inference_pb2.InferenceRequest(
                req_id=req_id,
                in_tok=record.in_tokens,
                out_tok=record.out_tokens,
                dataset=getattr(record, "dataset", ""),
                predicted_osl=getattr(record, "predicted_osl", 0),
            ),
            timeout=2000.0,
        )
        wall_ms = (time.monotonic() - t0) * 1000.0
        outbox_ms = _outbox_of(stub, req_id)
        recv_time = datetime.now(timezone.utc)
        result = {
            "req_id": req_id,
            "sched_id": resp.sched_path or sched_id,
            "dc_id": resp.dc_id,
            "status": resp.status,
            # The row reads as the request's life, in order.
            # client: the schedule said go, a worker issued the RPC, and the gap
            # between them is queueing the POOL imposed.
            "arrival_s": round(arrival_s, 6),
            "send_s": round(send_s, 6),
            "pool_wait_ms": round(max(0.0, send_s - arrival_s) * 1000.0, 3),
            # Streamed transport only: how long the request sat in the mux's
            # outbox before it reached the wire. 0.0 on the unary path, which
            # has no such queue. Reported because nothing else would show it --
            # it lands inside wall_ms and would otherwise look like fleet time.
            "outbox_ms": round(outbox_ms, 3),
            # send -> queue: the scheduler deciding, then the hop if it forwarded.
            # Left split rather than summed, because which of the two dominates
            # is the interesting part.
            "scheduling_ms": round(resp.scheduling_ms, 2),
            "forwarding_ms": round(resp.forwarding_ms, 2),
            # queue -> first token -> the rest of the tokens.
            "queue_ms": round(resp.queue_ms, 2),
            "ttft_ms": round(resp.ttft_ms, 2),
            "avg_tpot_ms": round(resp.avg_tpot_ms, 2),
            # How many prompts the prefill pass that produced THIS request's
            # first token had fused into it. The engine has measured it all
            # along and only the DECISION_LOG carried it, which the big sweeps
            # run without -- so the one number that says whether prefill has
            # fusion headroom left was absent from every grid. It belongs on
            # the request row: fusing amortises the ~14.6 ms fixed cost of a
            # pass, so it sets how much of the engine's wall clock prefill
            # takes, which is a term in every decoding request's TPOT.
            "prefill_bs": int(getattr(resp, "prefill_bs", 0) or 0),
            # done. wall_ms is the client's own measure, latency_ms and
            # service_ms are the scheduler's and the datacentre's.
            "recv_s": round(send_s + wall_ms / 1000.0, 6),
            "wall_ms": round(wall_ms, 2),
            "latency_ms": round(resp.latency_ms, 2),
            "service_ms": round(resp.service_ms, 2),
            "in_flight": in_flight,
            "in_tok": record.in_tokens,
            "out_tok": record.out_tokens,
            "carbon_intensity": round(resp.carbon_intensity, 2),
            # Wall clock, kept because collect_energy.py derives the window to
            # integrate DC energy over as max(recv_time) - min(send_time).
            "send_time": send_time.strftime("%H:%M:%S.%f")[:-3],
            "recv_time": recv_time.strftime("%H:%M:%S.%f")[:-3],
            "error": resp.error or "",
        }
        if verbosity >= 2:
            log.info(
                "req=%s  dc=%s  send=%s  recv=%s  wall=%.1fms",
                req_id, resp.dc_id,
                send_time.strftime("%H:%M:%S.%f")[:-3],
                recv_time.strftime("%H:%M:%S.%f")[:-3],
                wall_ms,
            )
        return result
    except Exception as e:
        wall_ms = (time.monotonic() - t0) * 1000.0
        recv_time = datetime.now(timezone.utc)
        result = {
            "req_id": req_id,
            "sched_id": sched_id,
            "dc_id": "",
            "status": "ERROR",
            # Same columns in the same order as the success path, or the two
            # write different headers into one file.
            "arrival_s": round(arrival_s, 6),
            "send_s": round(send_s, 6),
            "pool_wait_ms": round(max(0.0, send_s - arrival_s) * 1000.0, 3),
            "outbox_ms": round(_outbox_of(stub, req_id), 3),
            "scheduling_ms": 0.0,
            "forwarding_ms": 0.0,
            "queue_ms": 0.0,
            "ttft_ms": 0.0,
            "avg_tpot_ms": 0.0,
            "prefill_bs": 0,
            "recv_s": round(send_s + wall_ms / 1000.0, 6),
            "wall_ms": round(wall_ms, 2),
            "latency_ms": 0.0,
            "service_ms": 0.0,
            "in_flight": in_flight,
            "in_tok": record.in_tokens,
            "out_tok": record.out_tokens,
            "carbon_intensity": 0.0,
            "send_time": send_time.strftime("%H:%M:%S.%f")[:-3],
            "recv_time": recv_time.strftime("%H:%M:%S.%f")[:-3],
            "error": str(e),
        }
        if verbosity >= 2:
            log.warning(
                "req=%s  FAILED  sched=%s  send=%s  recv=%s  wall=%.1fms  err=%s",
                req_id, sched_id,
                send_time.strftime("%H:%M:%S.%f")[:-3],
                recv_time.strftime("%H:%M:%S.%f")[:-3],
                wall_ms, e,
            )
        return result


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)


def run(*args, **kwargs):
    """Sync entry point. The generator is one thread and one event loop."""
    return asyncio.run(run_async(*args, **kwargs))


async def run_async(
    duration_s: float,
    arrival_gen: Callable[[], float],
    token_gen: Callable[[], InferenceRecord],
    seed: int,
    output_csv: str,
    concurrent: int,
    verbosity: int = 1,
    tag: str = "",
    shard_index: int = 0,
    shard_count: int = 1,
):
    # Shard i of n: this process drives the schedulers whose sorted position
    # is i mod n, and nothing else. One generator is a single asyncio loop on
    # a single thread, so it saturates a core the same way a scheduler does --
    # measured at ~4200 req/s offered, with pool_wait p50 at 8.5 s past that,
    # which turns the run closed-loop. Splitting by scheduler needs no shared
    # state: each process has its own arrival schedule and its own output.
    #
    # The seed is offset by the shard index. Left alone, every shard samples
    # the same trace records in the same order and draws the same arrival
    # intervals, so n shards would replay one stream n times.
    random.seed(seed + shard_index)
    sched_endpoints = discover_schedulers()

    if not sched_endpoints:
        log.error("No scheduler endpoints found. Exiting.")
        sys.exit(1)

    if shard_count > 1:
        chosen = sorted(sched_endpoints)[shard_index::shard_count]
        if not chosen:
            log.error("shard %d/%d selects no scheduler out of %d. Use at most "
                      "as many shards as there are schedulers.",
                      shard_index, shard_count, len(sched_endpoints))
            sys.exit(1)
        sched_endpoints = {k: sched_endpoints[k] for k in chosen}

    stubs, channels = build_scheduler_stubs(sched_endpoints)
    # One long-lived stream per scheduler instead of one RPC per request.
    # Opt-in, and off by default, because the two transports have to stay
    # comparable: the same binary produces both, so a shift in the latency
    # distributions between them is the transport and nothing else. The mux
    # answers `SubmitRequest` with the unary signature, so nothing below here
    # knows which one it is holding. See loadgen/stream_mux.py.
    muxes = {}
    if os.environ.get("LOADGEN_STREAM", "0") == "1":
        muxes = build_scheduler_streams(stubs)
        stubs = muxes
    sched_ids = sorted(stubs.keys())
    if verbosity >= 1:
        log.info(
            "Connected to %d scheduler(s): %s",
            len(sched_ids), ", ".join(sched_ids),
        )

    # Set up per-request CSV log file for verbosity >= 2
    log_csv_file = None
    log_csv_writer = None
    log_fieldnames = [
        "req_id", "sched_id", "dc_id", "status",
        "arrival_s", "send_s", "pool_wait_ms", "outbox_ms",
        "scheduling_ms", "forwarding_ms",
        "queue_ms", "ttft_ms", "avg_tpot_ms", "prefill_bs",
        "recv_s", "wall_ms", "latency_ms", "service_ms",
        "in_flight", "in_tok", "out_tok", "carbon_intensity",
        "send_time", "recv_time", "error",
    ]
    if verbosity >= 2:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        parts = ["requests"]
        if tag:
            parts.append(tag)
        if shard_count > 1:
            parts.append(f"s{shard_index}")
        parts.append(ts)
        # Under the repo's results/, not next to the source. These used to land
        # in loadgen/ beside the trace CSVs, where 22 of them accumulated and
        # were easy to mistake for workload inputs.
        log_dir = os.path.join(_REPO_ROOT, "results", "per_request")
        os.makedirs(log_dir, exist_ok=True)
        log_csv_path = os.path.join(log_dir, f"{'_'.join(parts)}.csv")
        log_csv_file = open(log_csv_path, "w", newline="")
        log_csv_writer = csv.DictWriter(log_csv_file, fieldnames=log_fieldnames)
        log_csv_writer.writeheader()
        log.info("Logging per-request data to %s", log_csv_path)

    results: List[dict] = []
    idx = 0
    t_start = time.monotonic()
    pending: List[asyncio.Task] = []
    # `concurrent` > 0 caps in-flight requests, which makes the run DELIBERATELY
    # closed-loop; the sweep passes 0 and gets no cap at all. It used to be a
    # ThreadPoolExecutor width, and a pool that width could not help being the
    # ceiling -- measured on our testbed at 1200 req/s, a 144001-slot pool sat at 28%
    # occupancy while still delivering only 716/s, because 39789 threads
    # blocked on gRPC starved the arrival loop of the GIL. Widening it 4.8x
    # bought 9%. There is nothing to widen here.
    gate = asyncio.Semaphore(concurrent) if concurrent > 0 else None

    req_prefix = f"s{shard_index}-" if shard_count > 1 else ""

    async def _one(sid, req_num, record, arrival_rel, flight):
        if gate is None:
            return await send_one_request(stubs[sid], sid, req_num, record,
                                          verbosity, t_start, arrival_rel, flight,
                                          req_prefix)
        async with gate:
            return await send_one_request(stubs[sid], sid, req_num, record,
                                          verbosity, t_start, arrival_rel, flight,
                                          req_prefix)

    # Absolute arrival schedule, not sleep-for-the-interval. Sleeping for the
    # inter-arrival time makes the real interval (sleep + loop overhead), so
    # every iteration's cost accumulates and the achieved rate sits below the
    # nominal one -- by more as the interval shrinks, which compresses the rate
    # axis rather than just shifting it. Measured overhead here is 127-156 us
    # per iteration for a loop body far cheaper than this one, worth +0.4% of
    # wall clock at 30 req/s and +0.7% at 50. Scheduling against a running
    # deadline absorbs it instead of accumulating it.
    t_next = t_start
    peak_in_flight = 0
    # One thread, so no lock: nothing else can run between these statements.
    _in_flight = [0]

    def _flight_done(_task):
        _in_flight[0] -= 1

    while time.monotonic() - t_start < duration_s:
        sid = sched_ids[idx % len(sched_ids)]
        record = token_gen()
        idx += 1

        # t_next is where the schedule says this request belongs; passing it
        # rather than "now" is what makes pool_wait_ms measure the pool instead
        # of measuring the loop.
        _in_flight[0] += 1
        if _in_flight[0] > peak_in_flight:
            peak_in_flight = _in_flight[0]
        task = asyncio.create_task(
            _one(sid, idx, record, t_next - t_start, _in_flight[0]))
        task.add_done_callback(_flight_done)
        pending.append(task)
        # Peak in-flight is still worth reporting, but it is no longer a
        # fraction of anything: nothing refuses or queues a coroutine. It is
        # now simply how much work the fleet had outstanding at once.
        # Throttle arrival rate. Awaiting rather than sleeping is what lets the
        # in-flight requests make progress between arrivals -- on one thread,
        # a blocking sleep here would stall every outstanding RPC.
        t_next += arrival_gen()
        slack = t_next - time.monotonic()
        if slack > 0:
            await asyncio.sleep(slack)
        else:
            # Behind schedule: yield anyway, or a burst of overdue arrivals
            # starves the completions and nothing ever finishes.
            await asyncio.sleep(0)

    # What the generator actually offered, as opposed to what it was asked for.
    # Without this a rate-sweep axis is a set of labels nobody checked.
    t_arrivals_end = time.monotonic()
    achieved_rate = idx / max(t_arrivals_end - t_start, 1e-9)
    if verbosity >= 1:
        log.info("Arrivals: %d over %.1fs = %.2f req/s achieved",
                 idx, t_arrivals_end - t_start, achieved_rate)
        if concurrent > 0:
            log.info("Peak in flight: %d against a cap of %d (%.0f%%)%s",
                     peak_in_flight, concurrent,
                     100.0 * peak_in_flight / concurrent,
                     "  <-- CAPPED: arrivals were held, not open-loop"
                     if peak_in_flight >= concurrent else "")
        else:
            log.info("Peak in flight: %d (uncapped)", peak_in_flight)

    # Drain pending
    if verbosity >= 1:
        log.info("Draining %d pending requests...", len(pending))
    if pending:
        done, unfinished = await asyncio.wait(pending, timeout=2000)
        for task in done:
            try:
                r = task.result()
            except Exception as e:
                if verbosity >= 1:
                    log.warning("Request failed: %s", e)
                continue
            results.append(r)
            if log_csv_writer:
                log_csv_writer.writerow({k: r[k] for k in log_fieldnames})
        if unfinished:
            if verbosity >= 1:
                log.warning("Drain timeout: %d/%d completed",
                            len(done), len(pending))
            for task in unfinished:
                task.cancel()
    # Rows come back in completion order; sort so the file reads in send order
    # the way the threaded version's schedule implied.
    results.sort(key=lambda r: r["send_s"])
    # Close the streams before their channels: a mux still has a request
    # generator parked on its outbox, and closing the channel under it turns an
    # orderly half-close into a cancellation.
    for m in muxes.values():
        await m.close()
    for ch in channels:
        await ch.close()

    if log_csv_file:
        log_csv_file.close()

    # Write results
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        if verbosity >= 1:
            log.info("Wrote %d results to %s", len(results), output_csv)

    # Summary
    if verbosity >= 1:
        ok = [r for r in results if r["status"] == "OK"]
        err = [r for r in results if r["status"] != "OK"]
        if ok:
            def _pct(vals, p):
                s = sorted(vals)
                idx = min(int(len(s) * p), len(s) - 1)
                return s[idx]

            wall = [r["wall_ms"] for r in ok]
            lat = [r["latency_ms"] for r in ok]
            svc = [r["service_ms"] for r in ok]
            ttft = [r["ttft_ms"] for r in ok]
            tpot = [r["avg_tpot_ms"] for r in ok]
            sched = [r["scheduling_ms"] for r in ok]
            log.info(
                "Summary: %d OK, %d errors\n"
                "  wall_ms       avg=%.1f  p50=%.1f  p90=%.1f  p99=%.1f\n"
                "  latency_ms    avg=%.1f  p50=%.1f  p90=%.1f  p99=%.1f\n"
                "  service_ms    avg=%.1f  p50=%.1f  p90=%.1f  p99=%.1f\n"
                "  ttft_ms       avg=%.1f  p50=%.1f  p90=%.1f  p99=%.1f\n"
                "  avg_tpot_ms   avg=%.2f  p50=%.2f  p90=%.2f  p99=%.2f\n"
                "  scheduling_ms avg=%.2f  p50=%.2f  p90=%.2f  p99=%.2f",
                len(ok), len(err),
                sum(wall)/len(wall), _pct(wall, 0.5), _pct(wall, 0.9), _pct(wall, 0.99),
                sum(lat)/len(lat), _pct(lat, 0.5), _pct(lat, 0.9), _pct(lat, 0.99),
                sum(svc)/len(svc), _pct(svc, 0.5), _pct(svc, 0.9), _pct(svc, 0.99),
                sum(ttft)/len(ttft), _pct(ttft, 0.5), _pct(ttft, 0.9), _pct(ttft, 0.99),
                sum(tpot)/len(tpot), _pct(tpot, 0.5), _pct(tpot, 0.9), _pct(tpot, 0.99),
                sum(sched)/len(sched), _pct(sched, 0.5), _pct(sched, 0.9), _pct(sched, 0.99),
            )
        else:
            log.warning("All %d requests failed", len(err))



def main():
    parser = argparse.ArgumentParser(description="Load generator for dist AI inference")
    parser.add_argument("--duration", type=float, default=float(os.environ.get("DURATION_S", "30")),
                        help="Test duration in seconds")

    # Arrival timing
    parser.add_argument("--arrival-distribution", type=str,
                        default=os.environ.get("ARRIVAL_DISTRIBUTION", "uniform"),
                        choices=["uniform", "poisson"],
                        help="Inter-arrival time distribution (default: uniform)")
    parser.add_argument("--min-interval-ms", type=float, default=float(os.environ.get("MIN_INTERVAL_MS", "30")),
                        help="Min inter-arrival time in ms (uniform only)")
    parser.add_argument("--max-interval-ms", type=float, default=float(os.environ.get("MAX_INTERVAL_MS", "120")),
                        help="Max inter-arrival time in ms (uniform only)")
    parser.add_argument("--arrival-rate", type=float, default=float(os.environ.get("ARRIVAL_RATE", "13.3")),
                        help="Requests/sec for Poisson arrivals (poisson only)")

    # Required: a trace CSV. The default is the merged 3-dataset pool
    # (HumanEval + MBPP + GovReport) whose ISL and OSL were MEASURED from BS=1
    # benchmark runs. loadgen.make_scenarios builds the current families
    # instead, tokenizing per model; see loadgen/scenarios/README.md for what
    # separates the two.
    parser.add_argument("--trace-csv", type=str,
                        default=os.environ.get("TRACE_CSV", "loadgen/scenarios/mixed_trace_3ds.csv"),
                        help="Path to a CSV trace file from a real inference run. "
                             "Defaults to the 3-dataset merged pool; set to empty "
                             "string to use random tokens instead.")

    # Random token ranges (only used when --trace-csv is not set)

    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--output", type=str, default=os.environ.get("OUTPUT_CSV", "/tmp/loadgen_results.csv"))
    parser.add_argument("--concurrent", type=int, default=int(os.environ.get("CONCURRENT", "0")),
                        help="Worker pool width, i.e. max in-flight requests. "
                             "0 (the default) derives it from the rate and "
                             "duration so the pool provably cannot bind.")
    parser.add_argument("--verbosity", type=int, default=int(os.environ.get("VERBOSITY", "1")),
                        choices=[0, 1, 2],
                        help="Logging verbosity: 0=quiet, 1=summary(default), 2=per-request+csv")
    parser.add_argument("--shard", type=str,
                        default=os.environ.get("LOADGEN_SHARD", "0/1"),
                        help="i/n: drive only the schedulers at sorted "
                             "position i mod n. One generator is one asyncio "
                             "loop on one thread and saturates near 4200 "
                             "req/s offered; shards split that across "
                             "processes. Each shard needs its own --output "
                             "and its own share of --arrival-rate.")
    parser.add_argument("--tag", type=str, default=os.environ.get("RUN_TAG", ""),
                        help="Tag embedded in the CSV log filename (e.g. policy name)")
    args = parser.parse_args()

    try:
        _si, _sn = (int(x) for x in args.shard.split("/", 1))
    except ValueError:
        parser.error(f"--shard wants i/n, got {args.shard!r}")
    if _sn < 1 or not (0 <= _si < _sn):
        parser.error(f"--shard {args.shard}: need 0 <= i < n and n >= 1")

    # Adjust log level for verbosity 0
    if args.verbosity == 0:
        log.setLevel(logging.WARNING)

    # --- Build arrival generator ---
    arrival_gen = make_arrival_generator(
        distribution=args.arrival_distribution,
        min_interval_ms=args.min_interval_ms,
        max_interval_ms=args.max_interval_ms,
        arrival_rate=args.arrival_rate,
    )

    # --- Build token/workload generator ---
    # There is no pool to size. Under asyncio a request is a coroutine, so the
    # only reason to bound in-flight work is to make a run DELIBERATELY
    # closed-loop; --concurrent 0, which the sweep passes, means no bound.
    #
    # The history belongs here, because the wrong lever was pulled three times.
    # The width was pinned at 5000, then derived as rate x duration, then
    # capped by the machine's thread ceiling with vm.max_map_count raised to
    # 524288 -- and none of it was the constraint. Measured on our testbed at 1200
    # req/s with a 144001-slot pool: peak occupancy 28%, and the generator
    # still delivered 716/s of 1200 with pool_wait p50 at 5.2 s. Requests were
    # not waiting for a slot. The arrival loop could not submit, because 39789
    # threads blocked on gRPC were contending with it for the GIL. Widening the
    # pool 4.8x bought 9%.
    #
    # That is the property that matters: a thread-per-request generator is not
    # open-loop. The slower the system under test, the more threads are in
    # flight and the more the arrival loop starves -- so the slowest policy is
    # offered the least work and flatters itself, which is the bias the whole
    # comparison rests on not having.
    concurrent = args.concurrent
    if concurrent > 0 and args.verbosity >= 1:
        log.warning("in-flight capped at %d, so this run is CLOSED-loop by "
                    "request. Pass --concurrent 0 for open-loop.", concurrent)

    if args.trace_csv:
        records = load_trace(args.trace_csv)
        token_gen = make_trace_generator(records)
        if args.verbosity >= 1:
            log.info("Workload: trace replay from %s  (%d records)",
                     args.trace_csv, len(records))
    else:
        parser.error(
            "--trace-csv is required. The synthetic generator that used to "
            "stand in for it drew in_tok ~ U[200, 1200] and out_tok ~ U[50, 300], "
            "which is neither the shape nor the scale of the real workload: it "
            "is unimodal where the traces are bimodal (code prompts near 60 "
            "tokens, summarisation past 8000) and its ceiling is 1200 against "
            "the trace's p90 of 12190. It also carried no dataset and no "
            "predicted_osl, so OSL_SOURCE=predicted silently fell back. A run "
            "that reached it produced numbers that looked fine and meant "
            "nothing.")

    if args.verbosity >= 1:
        if args.arrival_distribution == "uniform":
            log.info("Arrival distribution: uniform [%.0f, %.0f] ms",
                     args.min_interval_ms, args.max_interval_ms)
        else:
            log.info("Arrival distribution: poisson  λ=%.2f req/s  (mean=%.1f ms)",
                     args.arrival_rate, 1000.0 / args.arrival_rate)

    run(
        duration_s=args.duration,
        arrival_gen=arrival_gen,
        token_gen=token_gen,
        seed=args.seed,
        output_csv=args.output,
        concurrent=concurrent,
        verbosity=args.verbosity,
        tag=args.tag,
        shard_index=_si,
        shard_count=_sn,
    )


if __name__ == "__main__":
    main()
