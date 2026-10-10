"""Datacenter gRPC server.

Implements DatacenterService (Dispatch + PollStats).
Each instance simulates one virtual datacenter with multiple nodes/GPUs.

Dispatch is **blocking**: the gRPC handler sleeps for the full simulated
inference duration so that round-trip latency equals simulated latency.
"""

import asyncio
import csv
import logging
import json
import os

from common.threads import set_stack_size
from common.grpc_bind import bind_with_retry
import sys
import threading
from concurrent import futures
from datetime import datetime, timezone

import grpc

# -- generated proto stubs (built at Docker image time) --
import inference_pb2
import inference_pb2_grpc

from common.config import (
    CARBON_INTENSITY,
    DATACENTER_GRPC_PORT,
    MAX_BATCH_SIZE,
    NUM_NODES_PER_DC,
    NUM_GPUS_PER_NODE,
    PUE,
    TP_SIZE,
)
from common.pue import get_pue
from datacenter.perf_models.factory import create_latency_backend
from datacenter.meta_dc import Datacenter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
)
log = logging.getLogger("datacenter")


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_LOG_FIELDNAMES = [
    "req_id", "dc_id", "gpu_id", "in_tok", "out_tok",
    "service_ms", "prefill_bs", "ttft_s", "start_time", "finish_time",
]


class DatacenterServicer(inference_pb2_grpc.DatacenterServiceServicer):
    def __init__(self, dc: Datacenter, verbosity: int = 1, log_tag: str = ""):
        self.dc = dc
        self.verbosity = verbosity
        self._csv_writer = None
        self._csv_file = None
        self._csv_lock = threading.Lock()

        if verbosity >= 2:
            safe_id = dc.dc_id.replace(":", "_")
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            tag_part = f"_{log_tag}" if log_tag else ""
            csv_path = os.path.join(_THIS_DIR, f"requests_{safe_id}{tag_part}_{ts}.csv")
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=_LOG_FIELDNAMES,
            )
            self._csv_writer.writeheader()
            self._csv_file.flush()
            log.info("Logging per-request data to %s", csv_path)

    def Dispatch(self, request, context):
        start_time = datetime.now(timezone.utc)
        # Blocks until simulated inference completes
        result = self.dc.dispatch(request.in_tok, request.out_tok,
                                  dataset=request.dataset,
                                  predicted_osl=request.predicted_osl)
        return self._respond(request, result, start_time,
                             datetime.now(timezone.utc))

    def _respond(self, request, result, start_time, finish_time):
        """Log the request and build its reply.

        Shared with AsyncDatacenterServicer so the two transports cannot come
        to report different things; only the waiting differs between them.
        """

        if self.verbosity >= 1:
            log.info(
                "DISPATCH req=%s  in=%d out=%d -> gpu=%s  svc=%.1fms  prefill_bs=%d  ttft=%.3fs",
                request.req_id,
                request.in_tok,
                request.out_tok,
                result["gpu_id"],
                result["service_ms"],
                result.get("prefill_bs", 0),
                result.get("ttft_s", 0),
            )

        if self._csv_writer:
            row = {
                "req_id": request.req_id,
                "dc_id": result["dc_id"],
                "gpu_id": result["gpu_id"],
                "in_tok": request.in_tok,
                "out_tok": request.out_tok,
                "service_ms": round(result["service_ms"], 2),
                "prefill_bs": result.get("prefill_bs", 0),
                "ttft_s": round(result.get("ttft_s", 0), 6),
                "start_time": start_time.strftime("%H:%M:%S.%f"),
                "finish_time": finish_time.strftime("%H:%M:%S.%f"),
            }
            with self._csv_lock:
                self._csv_writer.writerow(row)
                self._csv_file.flush()

        return inference_pb2.DispatchResponse(
            req_id=request.req_id,
            dc_id=result["dc_id"],
            gpu_id=result["gpu_id"],
            service_ms=result["service_ms"],
            start_ms=result["start_ms"],
            finish_ms=result["finish_ms"],
            ttft_ms=result["ttft_ms"],
            avg_tpot_ms=result["avg_tpot_ms"],
            queue_ms=result["queue_ms"],
            prefill_bs=int(result.get("prefill_bs", 0) or 0),
        )

    def PollStats(self, request, context):
        s = self.dc.stats()
        return inference_pb2.PollStatsResponse(
            dc_id=s["dc_id"],
            epoch=request.epoch,
            active_requests=s["total_active_requests"],
            pending_requests=s["total_pending_requests"],
            num_gpus=s["num_gpus"],
            num_nodes=s["num_nodes"],
            carbon_intensity=self.dc.carbon_intensity,
            total_energy_j=s["total_energy_j"],
            carbon_emissions_g=self.dc.carbon_emissions_g(),
            total_pending_isl=s["total_pending_isl"],
            total_prefill_isl=s["total_prefill_isl"],
            prefilling_requests=s["total_prefilling_requests"],
            total_active_isl=s["total_active_isl"],
            total_generated_tokens=s.get("total_generated_tokens", 0),
            decoding_requests=s.get("total_decoding_requests", 0),
            total_remaining_osl=s.get("total_remaining_osl", 0),
            idle_energy_j=s.get("total_idle_j", 0.0),
            host_energy_j=s.get("total_host_j", 0.0),
            prefill_duty=s.get("prefill_duty", 0.0),
            kv_tokens=s.get("kv_tokens", 0),
            kv_capacity_tokens=s.get("kv_capacity_tokens", 0),
            kv_blocked_steps=s.get("kv_blocked_steps", 0),
            admitted_requests=s.get("admitted_requests", 0),
            kv_queue_ms=s.get("kv_queue_ms", 0.0),
            kv_ms_per_token=s.get("kv_ms_per_token", 0.0),
            kv_room_tokens=s.get("kv_room_tokens", 0.0),
            pue=self.dc.pue,
        )

    def FinalizeEnergy(self, request, context):
        log.info("FinalizeEnergy: sim_duration=%.3fs", request.sim_duration_s)
        self.dc.compute_energy_from_duration(request.sim_duration_s)
        self._report_clock_fidelity()
        self._report_model_envelope()
        self._dump_bs_profile()
        return inference_pb2.FinalizeEnergyResponse(
            dc_id=self.dc.dc_id,
            total_energy_j=self.dc.stats()["total_energy_j"],
            carbon_emissions_g=self.dc.carbon_emissions_g(),
        )


    def _report_clock_fidelity(self):
        """Say how far the host stretched the simulated clock.

        The replay is real time: an engine prices a step with the analytical
        model and then sleeps it. A contended host lengthens every sleep, so
        the fleet reads slower for a reason that is not the model -- and the
        distortion grows with load, because there is one blocked thread per
        in-flight request. That makes it a BIASED error, not noise: the
        concentrating policies hold the most requests in flight and are
        therefore stretched the most, which is the same direction as the effect
        under study.

        The two counters have been accumulated since the beginning and read by
        nothing, so no operating point has ever been checked. WARNING above 5%:
        below that the run is within the model's own error, above it the run's
        latencies are partly the host's.
        """
        try:
            st = self.dc.stats()
            asked = float(st.get("asked_s", 0.0))
            over = float(st.get("oversleep_s", 0.0))
        except Exception:
            return
        if asked <= 0:
            return
        frac = over / asked
        # WARNING even when healthy, deliberately. The sweep runs datacentres
        # at DC_VERBOSITY=0, which sets the level to WARNING, so an INFO line
        # here is invisible in exactly the runs that matter -- and a MISSING
        # line reads as "under threshold" whether the number was 0.8% or the
        # plumbing was broken. It is one line per datacentre per run, and the
        # error budget needs the value, not just the verdict.
        log.warning(
            "clock fidelity: oversleep %.1fs / asked %.1fs = %.2f%%%s",
            over, asked, 100.0 * frac,
            "  <-- OVER 5%: latencies from this run are partly the HOST, "
            "not the model" if frac > 0.05 else "")

    def _report_model_envelope(self):
        """How far outside its fitted envelope the run priced things.

        Both counters existed and were read by nothing. The comment at the
        clamp site says "Read it at the end of a run; a non-zero count is a
        caveat the results have to carry" -- it was never read, so the caveat
        was never carried. And the warn-once flag beside it tells you the event
        HAPPENED, never how often, which is the difference between a footnote
        and a result that does not hold: at MAX_BATCH_SIZE=192 the decode fit
        is asked for KV well past the range it was measured over.

        WARNING level on purpose: datacentres run at DC_VERBOSITY=0, and an
        INFO line is invisible in exactly the runs that matter.
        """
        try:
            m = getattr(self.dc, "analytical_model", None)
            clamps, evals = m.envelope_clamps
        except Exception:
            return
        if not evals:
            return
        pct = 100.0 * clamps / evals
        log.warning(
            "model envelope: %d of %d decode-step prices extrapolated "
            "(%.1f%%)%s", clamps, evals, pct,
            "  <-- the decode fit was outside its measured range for most of "
            "this run" if pct > 50 else "")
        try:
            st = self.dc.stats()
            forced = int(st.get("kv_forced", 0))
            blocked = int(st.get("kv_blocked_steps", 0))
        except Exception:
            return
        if forced or blocked:
            log.warning(
                "KV gate: %d steps blocked, %d prompts admitted despite "
                "exceeding the WHOLE cache%s", blocked, forced,
                "  <-- vLLM would preempt or fail here; neither is modelled"
                if forced else "")

    def _dump_bs_profile(self):
        """Write this DC's batch-size occupancy next to the run, if asked.

        Side-channel on purpose: the profile is diagnostic, and threading it
        through the protobuf would mean regenerating stubs on every machine for
        a number nothing in the control path reads.
        """
        out_dir = os.environ.get("BS_PROFILE_DIR")
        if not out_dir:
            return
        try:
            prof = self.dc.stats().get("time_at_bs", {})
            total = sum(prof.values()) or 1.0
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"bs_profile_{self.dc.dc_id}.json")
            with open(path, "w") as f:
                json.dump({
                    "dc_id": self.dc.dc_id,
                    "seconds_at_bs": {str(k): v for k, v in sorted(prof.items())},
                    "fraction_at_bs": {str(k): v / total
                                       for k, v in sorted(prof.items())},
                    "gpu_seconds_total": total,
                    "p50_bs": _weighted_quantile(prof, 0.5),
                    "p95_bs": _weighted_quantile(prof, 0.95),
                    "max_bs": max(prof) if prof else 0,
                }, f, indent=2)
            log.info("wrote %s", path)
        except Exception as e:
            log.warning("bs profile dump failed: %s", e)


class AsyncDatacenterServicer(DatacenterServicer):
    """The datacentre, awaiting its engines instead of blocking a thread each.

    Why this exists. The synchronous servicer holds one gRPC worker thread for
    a request's whole life, so a site can carry only GRPC_MAX_WORKERS requests
    at once and everything past that waits in gRPC BEFORE the handler starts.
    Nothing inside the datacentre can see that wait: `service_ms` starts when
    the handler does and `queue_ms` (the seat wait) is inside it. Worse, a
    request stuck there is in neither `active_requests` nor `pending_requests`,
    so an overloaded site reports itself IDLE -- and every load-aware mechanism
    upstream reads those counters. Measured 2026-09-06: raising the pool from
    8000 to 40000 moved carbon_aware:0.0 from 2 datacentres to 12 and its
    delivered rate from 863 to 3032 req/s. The concentration was the pool's,
    not the policy's.

    Awaiting instead means a queued request costs a coroutine, joins `pending`
    where the provisioner can see it, and has its wait recorded in `queue_ms`
    where it belongs. Only PollStats stays as it was -- it is a fast read and
    grpc.aio runs a plain method directly on the loop.
    """

    async def Dispatch(self, request, context):
        start_time = datetime.now(timezone.utc)
        result = await self.dc.dispatch_async(request.in_tok, request.out_tok,
                                              dataset=request.dataset,
                                              predicted_osl=request.predicted_osl)
        return self._respond(request, result, start_time,
                             datetime.now(timezone.utc))


async def _serve_async(dc, port, verbosity, log_tag):
    server = grpc.aio.server()
    inference_pb2_grpc.add_DatacenterServiceServicer_to_server(
        AsyncDatacenterServicer(dc, verbosity=verbosity, log_tag=log_tag),
        server,
    )
    bind_with_retry(server, port)
    await server.start()
    log.info("Listening on port %d  (async, no per-request thread)", port)
    await server.wait_for_termination()


def _weighted_quantile(prof, q):
    """Batch size below which the DC spent q of its GPU-seconds."""
    if not prof:
        return 0
    items = sorted(prof.items())
    total = sum(v for _, v in items)
    acc = 0.0
    for bs, secs in items:
        acc += secs
        if acc >= q * total:
            return int(bs)
    return int(items[-1][0])


def serve():
    _kb = set_stack_size()
    if _kb:
        log.info("thread stack %d KB (SIM_THREAD_STACK_KB)", _kb)
    verbosity = int(os.environ.get("VERBOSITY", "1"))
    if verbosity == 0:
        log.setLevel(logging.WARNING)

    dc_name = os.environ.get("DC_ID", "dc-0")
    dc_index = os.environ.get("DC_INDEX", "")
    dc_id = f"DC{dc_index}:{dc_name}" if dc_index else dc_name
    num_nodes = int(os.environ.get("NUM_NODES", str(NUM_NODES_PER_DC)))
    num_gpus = int(os.environ.get("NUM_GPUS_PER_NODE", str(NUM_GPUS_PER_NODE)))
    port = int(os.environ.get("GRPC_PORT", str(DATACENTER_GRPC_PORT)))

    # Path to benchmark CSV (from scripts/benchmark_vllm.py)
    csv_path = os.environ.get("LATENCY_CSV", "benchmark_results.csv")
    latency_mode = os.environ.get("LATENCY_MODE", "lookup")
    model_name = os.environ.get("MODEL_NAME", "")

    # Load lookup backend (always needed for power data even in analytical mode)
    lookup = None
    if os.path.isfile(csv_path):
        lookup = create_latency_backend(
            "lookup",
            csv_path=csv_path,
            model_name=model_name,
        )
        log.info("Loaded latency lookup:\n%s", lookup.summary())
    elif latency_mode == "lookup":
        log.error("Latency CSV not found: %s  (set LATENCY_CSV env var)", csv_path)
        sys.exit(1)

    tp_size = int(os.environ.get("TP_SIZE", str(TP_SIZE)))

    # Load analytical model if requested
    analytical_model = None
    if latency_mode == "analytical" and model_name:
        from datacenter.perf_models.analytical.latency import AnalyticalLatencyModel
        try:
            # tp_size selects the fitted artifact. It matters more than a
            # scale factor would suggest: measured decode is 23% SLOWER at
            # TP=2 than TP=1 below BS=8 -- the all-reduce per layer outweighs
            # the halved weight traffic -- and 16% faster above BS=128. The
            # sign flips at BS~32, so no rescaling of one TP gives the other.
            analytical_model = AnalyticalLatencyModel(model_name, tp=tp_size)
            log.info("Loaded analytical model for %s:\n%s",
                     model_name, analytical_model.summary())
        except ValueError as e:
            log.warning("Failed to load analytical model: %s — falling back to lookup", e)
            latency_mode = "lookup"

    # Need at least one backend
    if lookup is None and analytical_model is None:
        log.error("No latency backend available — set LATENCY_CSV or MODEL_NAME")
        sys.exit(1)

    carbon_ci = float(os.environ.get("CARBON_INTENSITY", str(CARBON_INTENSITY)))
    max_bs = int(os.environ.get("MAX_BATCH_SIZE", str(MAX_BATCH_SIZE)))
    # The engine's max_model_len. It sets the prefill token budget
    # (max_num_batched_tokens = max(max_model_len, 2048) with chunked prefill
    # off), so it decides how many queued prompts fuse into one forward pass
    # and how many passes a long prompt is split across. 0 keeps the engine's
    # default, which is the testbed's 2048.
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "0"))
    # PUE for THIS site. The environment wins when it is set -- that is how a
    # sensitivity sweep varies one site -- otherwise the per-datacentre table
    # decides, and a site the table does not name takes the default. The table
    # currently sets every site to the default, deliberately: see
    # common/pue_by_dc.json, which says why a guess here would be worse than
    # uniformity.
    site_pue = float(os.environ["PUE"]) if os.environ.get("PUE") else get_pue(dc_id)
    dc = Datacenter(dc_id, num_nodes, num_gpus, lookup, tp_size=tp_size,
                    carbon_intensity=carbon_ci, max_batch_size=max_bs,
                    analytical_model=analytical_model, latency_mode=latency_mode,
                    model_name=model_name,
                    pue=site_pue,
                    max_model_len=max_model_len)
    bs_str = str(max_bs) if max_bs > 0 else "unlimited"
    cfg = dc.nodes[0].engine_config()
    log.info(
        "Starting datacenter %s  nodes=%d  cards/node=%d  tp=%d  engines/node=%d  ci=%.1f  pue=%.3f  max_bs=%s  latency=%s  max_model_len=%d  prefill_budget=%d",
        dc_id, num_nodes, num_gpus, tp_size, num_gpus // tp_size, carbon_ci,
        site_pue, bs_str, latency_mode,
        cfg["max_model_len"], cfg["max_prefill_tokens"],
    )

    log_tag = os.environ.get("LOG_TAG", "")

    # Async by default. Thread-per-request was not merely slower: a request
    # waiting for a worker is in neither `active_requests` nor
    # `pending_requests`, so an overloaded site reported itself idle to every
    # policy that reads those counters, and its wait appeared in no column at
    # all. DC_ASYNC=0 brings back the old server for comparison; it is the
    # fallback now, not the default.
    if os.environ.get("DC_ASYNC", "1") != "0":
        asyncio.run(_serve_async(dc, port, verbosity, log_tag))
        return

    # More workers because Dispatch now blocks for the full inference duration.
    # Each concurrent request needs its own thread.
    max_workers = int(os.environ.get("GRPC_MAX_WORKERS", "5000"))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    inference_pb2_grpc.add_DatacenterServiceServicer_to_server(
        DatacenterServicer(dc, verbosity=verbosity, log_tag=log_tag), server,
    )
    bind_with_retry(server, port)
    server.start()
    log.info("Listening on port %d  (max_workers=%d, thread-per-request)",
             port, max_workers)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
