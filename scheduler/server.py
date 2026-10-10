"""Scheduler gRPC server with carbon-aware cross-group forwarding.

Implements SchedulerService (SubmitRequest + PushSnapshot).
Each instance serves one scheduler group (europe, us_east, us_west, asia)
and can forward requests to remote schedulers when the carbon-aware policy
determines a remote DC is optimal.

Forwarding flow:
  1. Client sends SubmitRequest(forwarded=false)
  2. CarbonAware policy scores ALL DCs (local + remote)
  3. If best DC is local: dispatch directly
  4. If best DC is remote: sleep(RTT) -> SubmitRequest(forwarded=true)
     to remote scheduler -> sleep(RTT) -> return response
  5. Forwarded requests (forwarded=true) only consider local DCs
"""

import csv
import json
import logging
import functools
import os
import sys

from common.threads import set_stack_size
from common.grpc_bind import bind_with_retry
import asyncio
import signal
from collections import deque
import time
import threading
from datetime import datetime, timezone
from typing import Dict

import grpc

import inference_pb2
import inference_pb2_grpc
from scheduler.policies.common.latency_estimates import resolve_osl, OSL_SOURCES

from common.discovery import discover_datacenters, discover_schedulers
from common.network import REGION_TO_GROUP, rtt_pair_key
from common.stream_mux import StreamMux
from scheduler.policies.common.batch_worker import (BatchSolver,
                                                    BatchSolverUnavailable)
from scheduler.config_loader import load_scheduler_config
from scheduler.policies import get_policy
from scheduler.policies.common.base import PolicyScope
from scheduler.policies.common.projection import (project_placements,
                                                   project_totals)

# What the local correction added to the view, for the decision log. Thread
# local because _view_with_forwarded runs on the request's own thread: the
# counts stashed here are read back by that same request's log line, so this
# needs no lock and cannot cross requests.
_proj_note = threading.local()
from scheduler.policies.common.estimators import RoundRobinFallback, _check_sla
from scheduler.policies.common.capacity import dc_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
)
log = logging.getLogger("scheduler")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_LOG_FIELDNAMES = [
    "req_id", "dc_id", "remote_sched", "in_tok", "out_tok",
    "forwarded",
    "service_ms", "latency_ms",
    "ttft_ms", "avg_tpot_ms", "queue_ms",
    "forwarding_ms", "scheduling_ms",
    "carbon_intensity",
    "arrival_time", "completion_time",
]


class SchedulerServicer(inference_pb2_grpc.SchedulerServiceServicer):
    def __init__(self, sched_id: str, scheduler_group: str,
                 policy_name: str, alpha: float, simulate_rtt: bool,
                 verbosity: int = 1, log_tag: str = "",
                 batch_window_ms: int = 250,
                 batch_window_size: int = 300,
                 policy_kwargs: dict = None,
                 model_spec: tuple = ("", 1)):
        policy_kwargs = policy_kwargs or {}
        # Where a placement decision gets a request's output length. "oracle"
        # is the trace's true out_tok and makes this scheduler clairvoyant; it
        # is the default because it is what shipped, not because it is the
        # honest configuration to report on its own. Popped before the policy
        # is built -- it steers this class, not the policy. ref_osl is read
        # rather than popped: the greedy latency score takes it as a kwarg too.
        self._osl_source = policy_kwargs.pop("osl_source", "oracle")
        self._ref_osl = int(policy_kwargs.get("ref_osl", 500) or 500)
        self._osl_fallbacks = 0
        self._osl_warned = False
        self.sched_id = sched_id
        self.scheduler_group = scheduler_group
        self.simulate_rtt = simulate_rtt
        self.verbosity = verbosity
        self._csv_writer = None
        self._csv_file = None
        self._csv_rows_since_flush = 0
        self._csv_lock = threading.Lock()

        if verbosity >= 2:
            safe_id = sched_id.replace("-", "_")
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

        # Build policy
        self.policy = get_policy(
            policy_name,
            alpha=alpha,
            scheduler_group=scheduler_group,
            **policy_kwargs,
        )
        self._is_batch_policy = hasattr(self.policy, 'choose_dc_batch')
        # Last resort for a request the batch solve did not place, and for
        # the no-snapshot path -- see the empty-view branch in SubmitRequest.
        self._batch_rr = RoundRobinFallback()
        #: Requests placed with no snapshot at all. Nonzero means the run is
        #: void, not merely degraded.
        self._no_view_reqs = 0
        self._no_view_logged = 0.0

        # The batch build is interpreter-bound -- one _check_sla and one
        # _estimate_energy_j per (request, datacentre) cell -- and it shares a
        # GIL with the event loop. At 4800 req/s the loop wants ~96% of a core
        # and the build ~52%, so one of them starves and 26.5% of requests
        # came back unplaced. Off by ILP_SOLVE_PROCESS=0; see
        # policies/common/batch_worker.py for the measurement.
        self._solver = None
        self._solver_fallbacks = 0
        if self._is_batch_policy:
            self._tighten_gil_switching()
        if self._is_batch_policy and \
                os.environ.get("ILP_SOLVE_PROCESS", "1") != "0":
            spec_kwargs = {k: v for k, v in policy_kwargs.items()
                           if k != "analytical_model"}
            self._solver = BatchSolver({
                "policy_name": policy_name,
                "alpha": alpha,
                "scheduler_group": scheduler_group,
                "model_name": model_spec[0],
                "tp_size": model_spec[1],
                "policy_kwargs": spec_kwargs,
            })
            try:
                self._solver.start()
            except Exception as e:                               # noqa: BLE001
                log.warning("could not start the batch solver process (%s); "
                            "solving in-process", e)
                self._solver = None

        # A policy that estimates latency cannot work without the analytical
        # model, and without this check it fails once per REQUEST instead of
        # once at startup: every call raises inside the estimator and reaches
        # the load generator as an opaque gRPC error, which reads like a network
        # fault rather than a missing environment variable. quick_test.sh hit
        # exactly this -- 200 of 200 requests failed under carbon_aware because
        # MODEL_NAME was unset, and the run still looked like it had produced
        # results. Fail here, and say which variable is missing.
        if getattr(self.policy, "analytical_model", "n/a") is None:
            raise SystemExit(
                f"policy '{policy_name}' estimates latency and needs an "
                f"analytical model, but none was built -- MODEL_NAME is unset "
                f"or names a pair with no artifact. Set MODEL_NAME and TP_SIZE "
                f"to a calibrated pair, or use a policy that does not estimate "
                f"(round_robin, least_loaded).")

        # Requests forwarded IN from another scheduler and still running.
        # They land without this scheduler deciding anything, so nothing here
        # knows about them until the provisioner next polls -- and every local
        # decision taken in between prices a datacentre emptier than it is.
        # {dc_id: {req_id: (t_dispatch, isl, osl)}}. Local to this process: it
        # is never pushed to the provisioner and never leaves in a snapshot.
        #: req_id -> [t_dispatch, isl, osl, stage]. stage is "f" while the
        #: prompt is still owed, "d" once past this datacentre's observed
        #: prefill boundary, and "x" for an entry that has left (tombstoned so
        #: the queues below can skip it without an O(n) removal).
        self._fwd_inflight: Dict[str, Dict[str, list]] = {}
        #: The same entries, per datacentre, in dispatch order and split by
        #: stage, so ageing one from "f" to "d" is a popleft rather than a scan.
        self._fwd_qf: Dict[str, deque] = {}
        self._fwd_qd: Dict[str, deque] = {}
        #: [n_f, isl_f, osl_f, n_d, isl_d, osl_d] -- the six sums the
        #: projection actually consumes, maintained as dispatches begin and
        #: end instead of recomputed per request.
        self._fwd_agg: Dict[str, list] = {}
        # Per-DC EWMA of dispatch-to-first-token, which is where a projected
        # request stops being prefill and starts being decode.
        self._stage_ttft_ms: Dict[str, float] = {}
        self._fwd_lock = threading.Lock()
        # Per-engine, as the datacentres are configured; the capacity of a site
        # is this times its engine count. Read, not popped -- the policy needs
        # it too.
        self._max_num_seqs = int(policy_kwargs.get("max_num_seqs", 0) or 0)
        # How many schedulers are dispatching into the same fleet. Each one
        # projects only its OWN in-flight work, correctly, and then treats
        # the whole of a site's headroom as its own -- so with a cap of C
        # and a site at L, every one of them dispatches until ITS view
        # reads C and the site receives n x (C - L). Measured: BS_CAP=20
        # settles at 68 per engine with four schedulers, BS_CAP=90 at 105.
        # Scaling its own contribution by n is the symmetric assumption --
        # the others are doing what I am doing -- and it is proportional to
        # this scheduler's own activity, so one that is not sending to a
        # site still projects nothing there. 1 restores the old behaviour.
        self._peer_scale = max(1, int(policy_kwargs.get("peer_scale", 1) or 1))
        # ...and that symmetric assumption is false, measured. A site's load is
        # dominated by its OWN region's scheduler, so a constant n is wrong in
        # opposite directions for local and remote sites at the same instant:
        # over 120k decisions at 3200 req/s the projection came out 3.3x
        # PESSIMISTIC about local sites and 5x OPTIMISTIC about remote ones,
        # which inverts the ranking and made every scheduler export 100% of its
        # traffic while importing everyone else's.
        #
        # So measure the factor instead of assuming it. `admitted_requests` is
        # cumulative at the site and counts every scheduler's arrivals, so
        # differencing it across snapshots gives the site's whole arrival rate;
        # this scheduler knows its own. Their ratio is exactly the multiplier
        # that turns own-unreflected into total-unreflected, because
        # own_unreflected ~ lambda_own x age and (lambda_dc/lambda_own) x that
        # is lambda_dc x age. Per datacentre, and right for every scheduler at
        # once -- which no constant can be.
        #
        # peer_scale stays as the fallback for the first snapshots and for a
        # site this scheduler is not currently sending to.
        # ...and the ratio only reaches sites this scheduler DISPATCHES to.
        # `_fwd_begin` runs on the is_local branch alone, so a site it forwards
        # to instead holds no in-flight entry and gets no correction at all --
        # which is the measured 5x optimism about remote sites: not an
        # insufficient factor, an absent one. Those sites take the rate form
        # directly, `lambda_dc x snapshot_age`, which needs no tracking because
        # the counter already saw every scheduler's arrivals.
        self._peer_project = os.environ.get(
            "SCHED_PEER_PROJECT", "1") not in ("", "0", "false")
        self._own_sent = {}          # dc_id -> cumulative dispatches from here
        self._dc_rate = {}           # dc_id -> [adm, t, lam_dc, own, lam_own]
        # Token sums for the synthesised arrivals. Every scheduler replays the
        # same trace, so this scheduler's own mean is the right one to lend a
        # site it cannot see the individual requests at.
        self._mean_isl = 0.0
        self._mean_osl = 0.0
        self._peer_scale_max = float(
            os.environ.get("SCHED_PEER_SCALE_MAX", "32") or 32)
        # Held for the decision log, from the kwargs rather than from the
        # policy. Every policy is HANDED an analytical model, but only the ones
        # with an SLA constraint keep it as an attribute, so reading it back
        # off `self.policy` left the estimate blank on exactly the policies
        # that serve as controls -- round_robin and least_loaded, which are
        # where the estimator's behaviour is easiest to read precisely because
        # nothing was decided on it.
        # A forwarded request names the site it must land on, and the far end
        # honours it. That rule exists so a global policy's choice cannot be
        # silently substituted -- but the choice was made a hop and a snapshot
        # ago, and measured over 120k decisions at 3200 req/s, 77-81% of
        # arrivals land somewhere the receiving scheduler can already see is
        # over budget. Dispatching there anyway is the one outcome nobody
        # chose: the sender did not know, and the receiver was not allowed to
        # act on knowing.
        #
        # So re-check on arrival, and when the named site no longer fits, place
        # by round-robin inside the region. The REGION stands -- that is the
        # forwarding policy's decision and it is still carbon's to make. Only
        # the site moves, and only when it is already infeasible.
        #
        # Least-loaded inside the region, read off the PROJECTED view. The
        # usual objection to least-loaded here -- that the stale snapshot which
        # just proved wrong about one site would name the same emptiest site
        # for every arrival until the next poll -- does not apply, because
        # _view_with_forwarded folds this scheduler's own in-flight dispatches
        # in and it now does so with a measured factor. Each re-spot raises the
        # site it chose, so the next one goes elsewhere: the spreading is
        # self-correcting between snapshots rather than dependent on one.
        #
        # The substitution is recorded -- `target_dc` sits beside `dc_id` in
        # every decision record, so how often intent survived the hop is now a
        # measurement rather than an assumption. That was the whole objection
        # to the old re-deciding behaviour.
        self._arrival_recheck = os.environ.get(
            "FWD_ARRIVAL_RECHECK", "1") not in ("", "0", "false")
        self._respots = 0
        self._log_model = policy_kwargs.get("analytical_model")
        self._log_ttft_budget = float(policy_kwargs.get("ttft_budget_ms", 0) or 0)
        self._log_tpot_budget = float(policy_kwargs.get("tpot_budget_ms", 0) or 0)
        # Warned-about groups, so a partial fleet says it once and not per
        # request.
        self._warned_no_sched: set = set()

        # Snapshot state (updated by provisioner via PushSnapshot)
        self._epoch = -1
        self._dc_view: Dict[str, inference_pb2.DcStats] = {}
        # When each site's snapshot arrived, so a decision can record how old
        # the view it was taken on already was. DcStats carries no timestamp
        # and does not need one: the poll-to-push gap is 4-5 ms against a poll
        # period measured in seconds, so arrival time is the age that matters.
        self._dc_arrived: Dict[str, float] = {}
        self._rtt_matrix: Dict[str, float] = {}
        self._snap_lock = threading.Lock()

        # DC endpoint map: {dc_id: "host:port"}
        self._dc_endpoints: Dict[str, str] = {}
        self._refresh_dc_endpoints()

        # Scheduler endpoint map: {sched_id: "host:port"}
        self._sched_endpoints: Dict[str, str] = {}
        self._refresh_sched_endpoints()

        # gRPC channel caches
        self._dc_channels: Dict[str, grpc.Channel] = {}
        self._sched_channels: Dict[str, grpc.Channel] = {}
        # One long-lived stream per peer instead of one RPC per forward.
        # Opt-in and off by default, for the same reason the load generator's
        # is: the two transports have to stay comparable from one binary.
        # reconnect=True, unlike the generator's -- see common/stream_mux.py.
        self._peer_stream = os.environ.get("SCHED_PEER_STREAM", "0") == "1"
        self._sched_muxes: Dict[str, StreamMux] = {}

        self._req_seq = 0
        self._req_lock = threading.Lock()

        # Optional decision log, one JSON line per locally dispatched request:
        # the target datacentre's snapshot AS THE POLICY SAW IT, followed by
        # what that request actually experienced there. Both halves in one
        # record, so a candidate estimator can be scored offline without
        # re-running anything and without a join. Off unless DECISION_LOG is
        # set; the write happens after the response is timed.
        self._decision_log = None
        self._log_candidates = os.environ.get(
            "DECISION_LOG_CANDIDATES", "") not in ("", "0", "false")
        self._decision_lock = threading.Lock()
        self._decision_rows = 0
        _dpath = os.environ.get("DECISION_LOG", "")
        if _dpath:
            self._decision_log = open(_dpath, "a", buffering=1 << 16)
            log.info("Decision log: %s", _dpath)

        # Batch ILP buffer (only used when policy has choose_dc_batch)
        self._batch_buffer = []      # [(req_id, request, result_future, result_holder)]
        self._batch_lock = threading.Lock()
        self._batch_window_ms = batch_window_ms
        #: Requests that wake the batch before its window is up. A knob by this
        #: name existed here for a long time and was enforced NOWHERE, so
        #: setting it changed nothing while looking like it had; it was deleted
        #: in 7c1abcc and is reintroduced here with the wiring it never had.
        self._batch_window_size = (max(1, int(batch_window_size or 0))
                                   if batch_window_size else 0)
        #: Set by SubmitRequest when the buffer fills. An asyncio.Event, not
        #: a threading one: the batch loop now lives ON the event loop, and
        #: both sides run there. Created in start_batch so it binds to the
        #: loop that will actually run it.
        self._batch_ready = None
        self._batch_task = None
        #: Solves that did not fit their window. See _batch_loop.
        self._slow_solves = 0
        #: How long a buffered request waits for its batch before it is
        #: given up on. See the note at the wait itself for why 30 and not 5.
        self._batch_wait_s = float(os.environ.get("BATCH_WAIT_S", "30") or 30)
        #: Worker time for the last cycle, summed over its chunks.
        self._last_worker_ms = 0.0
        #: That round trip split into transit/unpack/work/return. See
        #: BatchSolver.last_split for why it is measured rather than reasoned.
        self._last_split = {}
        #: Set by serve() once the event loop exists. The batch thread starts
        #: here, before it does, and reads this to wake buffered requests.
        self._loop = None
        #: How much of the event loop the per-request view rebuild is eating.
        self._view_calls = 0
        self._view_ms = 0.0
        self._view_max_ms = 0.0
        #: The rest of the handler's on-loop time, split either side of the
        #: one await that dominates. "decide" is snapshot copy, view rebuild
        #: and the policy call; "finish" is the response object, the decision
        #: log and the CSV row. Everything the loop does that is NOT one of
        #: these is the runtime's own: protobuf parse and serialise, and task
        #: scheduling per RPC.
        self._t_decide = 0.0
        self._t_finish = 0.0
        self._t_csv = 0.0
        if self._is_batch_policy:
            # Started by serve() once the loop exists -- see start_batch.
            pass

    async def _loop_health(self, period_s: float = 10.0):
        """Report how late the loop runs its own callbacks.

        Thread starvation is gone, so CPU on the loop is what replaces it as
        the ceiling -- and nothing else would show that. `admit` stays small
        while every handler runs slowly, which reads as healthy. Lag measures
        it directly: this coroutine asks for period_s and reports how much
        longer it actually waited.

        `lag` alone does NOT detect saturation, which is the trap this fell
        into once: it measures how late ONE callback ran, and with short
        callbacks a loop at 100% still reports 1-3 ms while its task queue
        grows without bound. `cpu` is the honest saturation signal -- process
        CPU over wall time, so 100% means one core fully spent -- and `load`
        says how much of that the view rebuild is. Removing the view is only
        worth it while it is the majority of `cpu`.

        The candidate is right here in this class. _view_with_forwarded copies
        every in-flight entry across every datacentre once per request, so at
        a few thousand in flight it is millions of tuple copies a second, all
        on one thread. Under the old thread pool the GIL already serialised
        that, so this is not a regression -- but it is the next ceiling, and
        the numbers to compare are lag against view avg x request rate.
        """
        while True:
            t0 = time.monotonic()
            c0 = time.process_time()
            await asyncio.sleep(period_s)
            lag_ms = (time.monotonic() - t0 - period_s) * 1000.0
            cpu_pct = 100.0 * (time.process_time() - c0) / period_s
            calls, total, mx = self._view_calls, self._view_ms, self._view_max_ms
            self._view_calls, self._view_ms, self._view_max_ms = 0, 0.0, 0.0
            dec, fin, csv_s = self._t_decide, self._t_finish, self._t_csv
            self._t_decide = self._t_finish = self._t_csv = 0.0
            load_pct = 100.0 * total / (period_s * 1000.0)
            # WARNING when it matters, because the sweep runs the schedulers at
            # VERBOSITY=0 and would drop an info line -- which would hide the
            # one measurement that distinguishes "the loop is the new ceiling"
            # from "the fleet is finally the ceiling".
            emit = log.warning if (lag_ms > 100.0 or load_pct > 20.0
                                   or cpu_pct > 70.0) else log.info
            # decide and finish are the handler's own on-loop work either
            # side of the dispatch; view is inside decide, csv inside finish.
            # What they do NOT cover is the runtime's: protobuf parse and
            # serialise, and one task per RPC. If decide+finish is well under
            # the loop's 100%, that residue is where the time is.
            pc = lambda v: 100.0 * v / period_s
            emit("loop lag %.1fms/%.0fs | cpu=%.1f%% | n=%d | decide=%.1f%% "
                 "finish=%.1f%% (view=%.1f%% csv=%.1f%%) | rest=%.1f%% | "
                 "view avg=%.3fms max=%.2fms",
                 lag_ms, period_s, cpu_pct, calls, pc(dec), pc(fin),
                 load_pct, pc(csv_s),
                 max(0.0, 100.0 - pc(dec) - pc(fin)),
                 (total / calls) if calls else 0.0, mx)

    def _refresh_dc_endpoints(self):
        try:
            self._dc_endpoints = discover_datacenters()
            log.info("DC endpoints: %d datacenters", len(self._dc_endpoints))
        except Exception as e:
            log.warning("DC discovery failed: %s", e)

    def _refresh_sched_endpoints(self):
        try:
            all_scheds = discover_schedulers()
            # Exclude self
            self._sched_endpoints = {
                k: v for k, v in all_scheds.items() if k != self.sched_id
            }
            log.info("Scheduler endpoints: %d remote schedulers", len(self._sched_endpoints))
        except Exception as e:
            log.warning("Scheduler discovery failed: %s", e)

    def _get_dc_stub(self, dc_id: str) -> inference_pb2_grpc.DatacenterServiceStub:
        if dc_id not in self._dc_channels:
            addr = self._dc_endpoints.get(dc_id)
            if not addr:
                raise RuntimeError(f"No endpoint for datacenter {dc_id}")
            self._dc_channels[dc_id] = grpc.aio.insecure_channel(addr)
        return inference_pb2_grpc.DatacenterServiceStub(self._dc_channels[dc_id])

    #: Peer forwards go to this offset from a scheduler's advertised port.
    #: A convention rather than a discovery field, so nothing else changes.
    PEER_PORT_OFFSET = 1000

    @classmethod
    def peer_addr(cls, addr: str) -> str:
        host, _, port = addr.rpartition(":")
        return f"{host}:{int(port) + cls.PEER_PORT_OFFSET}"

    def _get_sched_stub(self, sched_id: str) -> inference_pb2_grpc.SchedulerServiceStub:
        if sched_id not in self._sched_channels:
            addr = self._sched_endpoints.get(sched_id)
            if not addr:
                raise RuntimeError(f"No endpoint for scheduler {sched_id}")
            self._sched_channels[sched_id] = grpc.aio.insecure_channel(
                self.peer_addr(addr))
        stub = inference_pb2_grpc.SchedulerServiceStub(
            self._sched_channels[sched_id])
        if not self._peer_stream:
            return stub
        mux = self._sched_muxes.get(sched_id)
        if mux is None:
            # Built on a stub over the SAME channel, so the peer address and
            # every channel option stay exactly what the unary path used.
            mux = StreamMux(stub, sched_id, reconnect=True)
            mux.start()
            self._sched_muxes[sched_id] = mux
        return mux

    def _find_sched_for_dc(self, dc_id: str) -> str | None:
        """Find which remote scheduler owns this DC based on region->group mapping."""
        dc_group = REGION_TO_GROUP.get(dc_id)
        if dc_group is None:
            return None
        target_sched_id = f"sched-{dc_group}"
        if target_sched_id in self._sched_endpoints:
            return target_sched_id
        return None

    def _next_req_id(self) -> str:
        with self._req_lock:
            self._req_seq += 1
            # Prefixed with this scheduler's id: several schedulers each count
            # from 1, and the load generator does too, so a bare req{n} means
            # different requests in different logs.
            return f"{self.sched_id}-req{self._req_seq}"

    @staticmethod
    def _tighten_gil_switching():
        """Shorten the GIL switch interval while a batch policy is running.

        A DIAGNOSTIC, opt-in, and deliberately blunt. The batch thread blocks
        in poll() with the GIL released, the reply arrives, and then it has to
        REACQUIRE the GIL to act on it -- and CPython hands the GIL over only
        every `switchinterval`, 5 ms by default, after which the OS often
        reschedules the CPU-bound thread again. Measured 2026-09-07 on a clean
        single-chunk cycle: the worker answered in 93.9 ms and the caller took
        862.7 ms to collect it.

        ANSWERED, and the answer is NO. At 4800 req/s over single-chunk
        cycles: 5 ms default gave worker 77.4 ms / wait 793.8 ms, and 0.5 ms
        gave worker 76.6 ms / wait 832.1 ms. The wait did not move, so GIL
        hand-off is not the mechanism and the asyncio conversion this would
        have justified was not started. Kept so the experiment is not repeated,
        and left off by default -- shortening the interval globally slows the
        loop for nothing.
        """
        val = os.environ.get("GIL_SWITCH_INTERVAL_MS", "")
        if not val:
            return
        try:
            secs = float(val) / 1000.0
        except ValueError:
            return
        if secs <= 0:
            return
        sys.setswitchinterval(secs)
        log.warning("GIL switch interval set to %.1f ms (was %.1f ms) -- "
                    "DIAGNOSTIC for the batch thread's collect latency",
                    secs * 1000.0, 5.0)

    async def _solve_in_chunks(self, req_infos, dc_view, rtt_matrix):
        """Solve the buffer, never handing the LP more than one window's worth.

        A single solve over a huge buffer is what turned a stalled solver into
        an OOM: the constraint matrix is now sparse, so the memory is linear,
        but the LP itself is still superlinear in the batch and a 12000-request
        solve would stall the loop long enough to collect the next one. The
        buffer only gets that large when something has already gone wrong --
        the solver timeout is 10 s and 1200 req/s fills it -- so this is a
        guard for the bad case, not the normal one. In steady state the batch
        equals the window size and this runs exactly one chunk.

        NOT a drain cap: everything buffered is still solved, in pieces, in
        this cycle. Leaving the remainder for later would turn a throughput
        deficit into a growing backlog.

        THE VIEW IS PROJECTED BETWEEN CHUNKS, and without that the chunking
        would quietly undo the capacity constraint it is meant to protect:
        every chunk would see the same free seats and fill the same greenest
        site, so N chunks would seat N times its capacity. That is the same
        class of fault as the constraint being absent altogether.
        """
        chunk = self._batch_window_size or len(req_infos)
        view = dc_view
        out = {}
        # Summed across chunks. `last_worker_ms` is per solve() call, so
        # reporting it directly credited a multi-chunk cycle with only its
        # LAST chunk's work and charged the rest to `wait` -- which is the
        # number the next decision rests on, so it has to be the real one.
        self._last_worker_ms = 0.0
        self._last_split = {"in": 0.0, "unpack": 0.0, "work": 0.0, "out": 0.0}
        for start in range(0, len(req_infos), chunk):
            part = req_infos[start:start + chunk]
            got = None
            if self._solver is not None:
                try:
                    # Generous relative to the window: this bounds a dead
                    # worker, not a slow one, and a needless fallback puts the
                    # build back on the loop's interpreter.
                    got = await self._solver.solve_async(
                        part, view, self.scheduler_group, rtt_matrix,
                        max_batch_size=self._max_num_seqs, timeout=10.0,
                    )
                except BatchSolverUnavailable as e:
                    self._solver_fallbacks += 1
                    log.warning("batch solver unavailable (%s); solving "
                                "in-process (%d so far)",
                                e, self._solver_fallbacks)
            if got is None:
                # To a thread, not inline: a 77 ms solve on the event loop
                # stalls every request for 77 ms. Rare -- this path is taken
                # only when the solver process is unreachable.
                got = await asyncio.get_running_loop().run_in_executor(
                    None, functools.partial(
                        self.policy.choose_dc_batch,
                        part, view, self.scheduler_group,
                        max_batch_size=self._max_num_seqs,
                        rtt_matrix=rtt_matrix))
            out.update(got)
            if self._solver is not None:
                self._last_worker_ms += self._solver.last_worker_ms
                for k, v in (self._solver.last_split or {}).items():
                    self._last_split[k] = self._last_split.get(k, 0.0) + v

            if start + chunk < len(req_infos):
                placed = {}
                by_id = {r.req_id: r for r in part}
                for req_id, dc_id in got.items():
                    r = by_id.get(req_id)
                    if r is not None:
                        placed.setdefault(dc_id, []).append(r)
                view = dict(view)
                for dc_id, rs in placed.items():
                    base = view.get(dc_id)
                    if base is None:
                        continue
                    view[dc_id] = project_placements(
                        base, [r.in_tok for r in rs],
                        [int(getattr(r, "predicted_osl", 0) or r.out_tok)
                         for r in rs],
                        self._max_num_seqs)
        return out

    def start_batch(self):
        """Start the batch loop as a task on the running event loop.

        It used to be a thread, and the reason it is not any more was measured
        rather than argued. Splitting the solver round trip four ways at 4800
        req/s: the job reached the child in 0.3 ms, the child deserialised it
        in 0.1 ms and solved in 77 ms -- and the parent then took 833 ms to
        READ THE REPLY. Nothing was slow except a blocked thread waiting to be
        scheduled inside a busy process, and a 0.5 ms GIL switch interval did
        not move it.

        On the loop there is nothing to wake: `add_reader` hands the pipe to a
        thread that is already running. The in-process fallback still goes to
        an executor, because a 77 ms solve on the loop would stall every
        request for 77 ms.
        """
        if not self._is_batch_policy or self._batch_task is not None:
            return
        self._batch_ready = asyncio.Event()
        self._batch_task = asyncio.get_running_loop().create_task(
            self._batch_loop())
        # Two limits, and they do different jobs. `window_ms` is the MAXIMUM
        # wait: it bounds the latency a request pays for being batched. `max`
        # is a TRIGGER: it wakes the loop early so a burst does not wait out
        # the window and arrive as one oversized problem.
        #
        # `max` is deliberately not a drain cap. The loop still takes
        # everything buffered, because capping what it takes would convert a
        # throughput deficit into an unbounded backlog instead of fixing it --
        # draining 300 while 400 arrive is worse than draining 400.
        #
        # A knob by this name lived here for a long time, printed in this very
        # line and enforced nowhere; it was deleted in 7c1abcc and is back with
        # the wiring it never had.
        # The LP's capacity constraint comes from the hardware: max_num_seqs
        # per engine, scaled by each site's own engine count inside
        # _solve_lp. It was never passed before, so `cap_per_engine` was 0,
        # no A_ub was built, and the problem SEPARATED BY REQUEST -- an LP
        # with no coupling constraint is a per-request argmin, which is
        # exactly the greedy rule the ILP is supposed to beat. That is why
        # every ILP tied carbon_aware:0.0. BS_CAP still applies on top as an
        # optional tighter guard; see policies/common/capacity.py.
        log.info("Batch ILP loop started (max wait=%dms, wakes at %s, "
                 "LP cap=%d seats/engine)",
                 self._batch_window_ms,
                 f"{self._batch_window_size} requests"
                 if self._batch_window_size else "no size trigger",
                 self._max_num_seqs)

    async def _batch_loop(self):
        from scheduler.policies.common.estimators import RequestInfo
        window_s = self._batch_window_ms / 1000.0
        # Absolute cadence, not sleep-for-the-interval. Sleeping the whole
        # window at the top of every iteration made the PERIOD
        # `window + solve`, not `max(window, solve)` -- so the solve time
        # fed straight back into the batch size, which fed back into the
        # solve time. Measured 2026-09-06 at 4000 req/s: a 250 ms window
        # and a ~200 ms solve gave a 450 ms cycle carrying 450 requests
        # instead of 250, and scheduling_ms p50 landed on 449 ms, which is
        # the cycle itself. At 4800 the loop outran the handler's 5 s
        # buffer wait and 32.8% of ilp_carbon_sla's requests were returned
        # as "batch solver did not answer" -- an implementation limit that
        # would otherwise have been reported as the ILP failing to scale.
        #
        # The load generator already learned this one; see the note on the
        # arrival schedule in loadgen/client.py.
        next_at = time.monotonic() + window_s
        while True:
            delay = next_at - time.monotonic()
            if delay > 0:
                # The window is a MAXIMUM wait, not a fixed period: a full
                # buffer wakes this early. Without that, a burst waits out
                # the whole window and arrives as one oversized problem.
                try:
                    await asyncio.wait_for(self._batch_ready.wait(), delay)
                except asyncio.TimeoutError:
                    pass
            # else: the last cycle overran its window. Do not accumulate
            # debt -- that spins without sleeping and starves the solve.
            self._batch_ready.clear()
            next_at = time.monotonic() + window_s
            with self._batch_lock:
                if not self._batch_buffer:
                    continue
                batch = list(self._batch_buffer)
                self._batch_buffer.clear()

            # Build RequestInfo list
            req_infos = [
                RequestInfo(req_id=item[0], in_tok=item[1].in_tok,
                            out_tok=item[1].out_tok,
                            predicted_osl=self._osl_for(item[1]))
                for item in batch
            ]

            with self._snap_lock:
                dc_view = dict(self._dc_view)
                rtt_matrix = dict(self._rtt_matrix)
            dc_view = self._view_with_forwarded(dc_view)

            # Solve batch assignment
            t_solve = time.monotonic()
            assignments = await self._solve_in_chunks(req_infos, dc_view,
                                                      rtt_matrix)
            t_done = time.monotonic()
            solve_ms = (t_done - t_solve) * 1000.0
            # WARNING when the solve does not fit its window: the sweep
            # runs the schedulers at VERBOSITY=0, so an info line is
            # dropped and this is the one measurement that says whether a
            # batch policy is keeping up. Without it, a solver falling
            # behind reaches the results only as requests that were never
            # placed, with nothing naming the cause.
            # Split the round trip from the work. solve_ms is what this
            # thread waited; worker_ms is what the solver actually spent.
            # A large gap between them is not a slow solver -- it is this
            # thread not being scheduled to read the reply, which is the
            # thing the offload exists to expose.
            worker_ms = (self._last_worker_ms
                         if self._solver is not None else solve_ms)
            if solve_ms > self._batch_window_ms:
                self._slow_solves += 1
                sp = self._last_split or {}
                log.warning("ILP batch: %d requests, solve=%.1fms "
                            "(worker=%.1fms, wait=%.1fms | in=%.1f "
                            "unpack=%.1f out=%.1f) exceeds the %dms "
                            "window (%d so far); the cadence holds but "
                            "the buffer grows",
                            len(batch), solve_ms, worker_ms,
                            solve_ms - worker_ms,
                            sp.get("in", 0.0), sp.get("unpack", 0.0),
                            sp.get("out", 0.0), self._batch_window_ms,
                            self._slow_solves)
            else:
                log.info("ILP batch: %d requests, solve=%.1fms "
                         "(worker=%.1fms)", len(batch), solve_ms,
                         worker_ms)

            # Distribute assignments to waiting threads. Each request's
            # scheduling_ms is the full wall-clock from when SubmitRequest
            # buffered it to when the LP solve finished — i.e. buffer
            # wait + LP solve. This is what the user actually pays.
            for req_id, request, result_future, result_holder, t_sched_start in batch:
                dc_id = assignments.get(req_id)
                if dc_id is None:
                    # A request the policy left out entirely. This used to
                    # pick the least loaded datacentre, which is the third
                    # copy of a rule the carbon policies deliberately do not
                    # follow: they send work to the low-carbon sites, so the
                    # least loaded one is reliably the carbon-expensive one,
                    # and choosing it turns a scheduling failure into a
                    # carbon failure. Round-robin is not smart, but unlike
                    # least-loaded it does not correlate with carbon.
                    dc_id = self._batch_rr.pick(dc_view)
                result_holder["dc_id"] = dc_id
                result_holder["scheduling_ms"] = (t_done - t_sched_start) * 1000.0
                if dc_id:
                    dc_group = REGION_TO_GROUP.get(dc_id, self.scheduler_group)
                    result_holder["is_local"] = (dc_group == self.scheduler_group)
                    result_holder["dc_group"] = dc_group
                else:
                    result_holder["is_local"] = True
                    result_holder["dc_group"] = self.scheduler_group
                # Resolved in place, because this runs ON the loop now. It
                # used to go through call_soon_threadsafe, which the thread
                # version needed and which costs a self-pipe write per
                # request -- 300 syscalls a cycle for nothing here.
                if not result_future.done():
                    result_future.set_result(None)



    # ----- dispatches not yet in a snapshot -----
    def _fwd_begin(self, req_id: str, dc_id: str, request):
        """Record a dispatch from the moment it is sent.

        This tracks THIS scheduler's own placements as well as the ones
        forwarded in. It used to track only the forwarded ones, on the reasoning
        that "this scheduler's own placements are the ones its policy already
        knows it made" -- which is not true of any policy here. They all read
        self._dc_view, a snapshot from the last poll; none of them carries
        memory of what it dispatched since. Measured over probe_rate2, that is
        the whole of the TTFT estimator's error: on decisions where the engine
        really did prefill the request alone, the estimate lands within 13% of
        measured, and it degrades to 0.04x exactly as the share of such
        decisions falls.
        """
        e = [time.monotonic(), int(request.in_tok),
             int(self._osl_for(request)), "f"]
        with self._fwd_lock:
            self._fwd_inflight.setdefault(dc_id, {})[req_id] = e
            self._fwd_qf.setdefault(dc_id, deque()).append(e)
            a = self._fwd_agg.get(dc_id)
            if a is None:
                a = self._fwd_agg[dc_id] = [0, 0.0, 0.0, 0, 0.0, 0.0]
            a[0] += 1
            a[1] += e[1]
            a[2] += e[2]
            # Cumulative, never aged: _fwd_agg is the in-flight set and shrinks
            # as dispatches land, but the rate estimate needs a monotonic
            # counter to difference against the site's own.
            self._own_sent[dc_id] = self._own_sent.get(dc_id, 0) + 1

    def _note_stage_boundary(self, dc_id: str, queue_ms: float, ttft_ms: float):
        """Learn where prefill ends at this datacentre, from completions.

        The stage split needs one number per datacentre: how long a request
        spends between dispatch and its first token. The scheduler observes it
        on every response, so no extra measurement and no circular dependency
        on the estimate being corrected.
        """
        v = max(1.0, float(queue_ms) + float(ttft_ms))
        with self._fwd_lock:
            prev = self._stage_ttft_ms.get(dc_id)
            self._stage_ttft_ms[dc_id] = v if prev is None else 0.9 * prev + 0.1 * v

    def _fwd_drop(self, dc_id: str, e: list):
        """Take one entry out of the sums. Caller holds _fwd_lock.

        The entry is tombstoned rather than removed from its queue: the queues
        exist only to age entries in dispatch order, and a dead one costs a
        popleft when the front reaches it. Removing it in place would be the
        O(n) scan this whole structure exists to avoid.
        """
        stage = e[3]
        if stage == "x":
            return
        e[3] = "x"
        a = self._fwd_agg.get(dc_id)
        if a is None:
            return
        if stage == "f":
            a[0] -= 1
            a[1] -= e[1]
            a[2] -= e[2]
        else:
            a[3] -= 1
            a[4] -= e[1]
            a[5] -= e[2]

    def _fwd_age(self, dc_id: str, now: float):
        """Move entries past this datacentre's prefill boundary to "d".

        Caller holds _fwd_lock. Amortised O(1) per entry over its lifetime:
        dispatches are appended in time order, so everything that has aged out
        is at the front, and each entry migrates once.

        One deliberate difference from recomputing the stage per request: the
        boundary is an EWMA and can move DOWN, which would put an already-aged
        entry back in "f". It does not go back here. A request that has passed
        its prompt has passed it; the previous behaviour let a moving average
        un-decide that.
        """
        b = self._stage_ttft_ms.get(dc_id)
        if b is None:
            return              # nothing observed yet: everything is "f"
        cutoff = now - b / 1000.0
        qd = self._fwd_qd.get(dc_id)
        while qd and qd[0][3] == "x":
            qd.popleft()
        qf = self._fwd_qf.get(dc_id)
        if not qf:
            return
        a = self._fwd_agg.get(dc_id)
        while qf:
            e = qf[0]
            if e[3] == "x":
                qf.popleft()
                continue
            if e[0] >= cutoff:
                break
            qf.popleft()
            if a is not None:
                a[0] -= 1
                a[1] -= e[1]
                a[2] -= e[2]
                a[3] += 1
                a[4] += e[1]
                a[5] += e[2]
            e[3] = "d"
            if qd is None:
                qd = self._fwd_qd[dc_id] = deque()
            qd.append(e)

    def _fwd_end(self, req_id: str, dc_id: str):
        """Forget it: Dispatch returned, so it is no longer in flight."""
        with self._fwd_lock:
            at_dc = self._fwd_inflight.get(dc_id)
            if at_dc:
                e = at_dc.pop(req_id, None)
                if e is not None:
                    self._fwd_drop(dc_id, e)
                if not at_dc:
                    del self._fwd_inflight[dc_id]

    def _respot(self, dc_id, dc_view, request):
        """The least-loaded site in the SAME region, if the named one no longer fits.

        Returns None to keep the sender's choice -- when it is still feasible,
        when there is no model or budget to judge it with, or when the rotation
        lands back on it.

        No `fwd_ms`: the hop has been paid and cannot be paid again, which is
        the same reason the local path omits it.
        """
        model = self._log_model
        stats = dc_view.get(dc_id)
        if model is None or stats is None or not self._log_ttft_budget:
            return None
        try:
            ok, _, _ = _check_sla(
                stats, int(request.in_tok),
                self._log_ttft_budget, self._log_tpot_budget, model,
                max_num_seqs=self._max_num_seqs,
                req_osl=self._osl_for(request))
        except Exception:          # a judgement we could not make is not a veto
            return None
        if ok:
            return None
        alt, best = None, None
        for cand, cstats in dc_view.items():
            if REGION_TO_GROUP.get(cand) != self.scheduler_group:
                continue
            # Per engine: the region's sites are not all the same size, and the
            # site-wide sum would send everything to the largest one.
            load = dc_load(cstats) / max(1, int(
                getattr(cstats, "num_gpus", 1) or 1))
            if best is None or load < best:
                alt, best = cand, load
        if alt is None or alt == dc_id:
            return None
        self._respots += 1
        return alt

    def _update_rates(self, dc_id, stats, now):
        """Two arrival rates for one site: the site's own, and this scheduler's.

        Both are differenced over the SAME interval so their ratio is a share
        and not an artefact of two sampling periods. Smoothed lightly -- one
        poll carries hundreds of arrivals, so the noise is small, but a site
        that briefly receives nothing from here would otherwise send the ratio
        to its clamp and back.

        `admitted_requests` is 0 on a datacentre that predates the field; the
        rate then never becomes available and `_k_for` keeps using peer_scale.
        """
        adm = int(getattr(stats, "admitted_requests", 0) or 0)
        own = int(self._own_sent.get(dc_id, 0))
        prev = self._dc_rate.get(dc_id)
        if prev is None:
            self._dc_rate[dc_id] = [adm, now, None, own, None]
            return
        dt = now - prev[1]
        if dt <= 0.0 or adm < prev[0]:
            return
        lam_dc = (adm - prev[0]) / dt
        lam_own = max(0.0, (own - prev[3])) / dt
        s = lambda old, new: new if old is None else 0.6 * old + 0.4 * new
        self._dc_rate[dc_id] = [adm, now, s(prev[2], lam_dc),
                                own, s(prev[4], lam_own)]

    def _note_request_size(self, request):
        """Running mean ISL/OSL, for arrivals this scheduler did not place."""
        isl = float(request.in_tok or 0)
        osl = float(self._osl_for(request) or 0)
        if self._mean_isl <= 0.0:
            self._mean_isl, self._mean_osl = isl, osl
        else:
            self._mean_isl = 0.99 * self._mean_isl + 0.01 * isl
            self._mean_osl = 0.99 * self._mean_osl + 0.01 * osl

    def _peer_only_counts(self, dc_id, stats, now):
        """Arrivals at a site since its snapshot, when none of them are mine.

        `admitted_requests` is cumulative at the datacentre and counts every
        scheduler, so the rate differenced from it times the age of the
        snapshot IS the unreflected arrivals -- no in-flight tracking, and no
        assumption about who sent them.

        Split by age against the site's own dispatch-to-first-token time: what
        arrived within the last `stage_b` is still prefilling, the rest has
        joined the decode set. With no boundary learned yet, call it all
        prefill -- these are the newest arrivals in the system and the age is
        at most a poll period.

        Read without the snapshot lock, as `_log_decision` already reads
        `_dc_arrived`: a dict lookup that races a rewrite returns one epoch or
        the other, and both are snapshots this projection is correcting anyway.
        """
        r = self._dc_rate.get(dc_id)
        arrived = self._dc_arrived.get(dc_id)
        if r is None or r[2] is None or arrived is None:
            return 0, 0
        age = now - arrived
        if age <= 0.0:
            return 0, 0
        n = r[2] * age
        engines = max(1, int(getattr(stats, "num_gpus", 1) or 1))
        if self._max_num_seqs > 0:
            # A rate estimate cannot imply more arrivals than the site has
            # seats; past that the projection says nothing the cap does not.
            n = min(n, float(engines * self._max_num_seqs))
        n = int(round(n))
        if n <= 0:
            return 0, 0
        b = self._stage_ttft_ms.get(dc_id)
        if b is None:
            return n, n
        return n, int(round(n * min(1.0, (b / 1000.0) / age)))

    def _k_for(self, dc_id) -> float:
        """How much of this site's unreflected load is NOT mine.

        1.0 would mean I am its only source; the fleet-wide constant this
        replaces asserted the scheduler count everywhere. Clamped below at 1
        because a site cannot receive less than what I sent it, and above
        because lambda_own goes to zero on a site I have stopped using, where
        the ratio is unbounded and the projection it feeds is zero anyway.
        """
        r = self._dc_rate.get(dc_id)
        if r is None or r[2] is None or not r[4]:
            return float(self._peer_scale)
        return min(self._peer_scale_max, max(1.0, r[2] / r[4]))

    def _view_with_forwarded(self, dc_view):
        """`dc_view` with the forwarded-in requests it cannot know about.

        Folded in with the same rule the iterative ILP uses between its own
        rounds, so the admitted-vs-queued split and the disjointness of
        active/pending are whatever the projection says they are.

        Costs O(datacentres), not O(in-flight). It used to copy every in-flight
        entry this scheduler held, tag each one with a stage, and build two
        lists per datacentre -- per request -- to produce six sums that
        project_totals could have been handed directly. Measured on the
        512-engine fleet, that copy grew from 0.20 ms a call at 2400 req/s to
        0.64 ms at 5000, where it was 52% of the scheduler process's CPU: the
        cost per call rises with in-flight work while the calls per second rise
        with the rate, so it is quadratic in load and it became the ceiling as
        soon as the thread-per-request one was removed.

        The sums are now maintained as dispatches begin, age and end. Nothing
        is approximated -- the same six numbers, not recomputed.

        The `arrived` filter this used to apply is gone because PushSnapshot
        already deletes anything dispatched before the snapshot landed, so
        every surviving entry passed it by construction.
        """
        t_view = time.monotonic()
        notes = {}
        _proj_note.by_dc = notes
        with self._fwd_lock:
            for dc_id in self._fwd_agg:
                self._fwd_age(dc_id, t_view)
            totals = {d: (tuple(a), self._stage_ttft_ms.get(d))
                      for d, a in self._fwd_agg.items() if a[0] or a[3]}
        for dc_id, (a, b) in totals.items():
            base = dc_view.get(dc_id)
            if base is None:
                continue        # not in this wave's snapshot; nothing to correct
            k = self._k_for(dc_id)
            dc_view[dc_id] = project_totals(base, int(round(a[0] * k)),
                                            a[1] * k, a[2] * k,
                                            int(round(a[3] * k)),
                                            a[4] * k, a[5] * k,
                                            self._max_num_seqs)
            notes[dc_id] = (a[0] + a[3], a[0], 0.0 if b is None else b)
        if self._peer_project and self._mean_isl > 0.0:
            for dc_id, base in list(dc_view.items()):
                if dc_id in totals:
                    continue        # already corrected from real in-flight
                n, n_f = self._peer_only_counts(dc_id, base, t_view)
                if n <= 0:
                    continue
                n_d = n - n_f
                dc_view[dc_id] = project_totals(
                    base, n_f, n_f * self._mean_isl, n_f * self._mean_osl,
                    n_d, n_d * self._mean_isl, n_d * self._mean_osl,
                    self._max_num_seqs)
                notes[dc_id] = (n, n_f, self._stage_ttft_ms.get(dc_id) or 0.0)
        _proj_note.by_dc = notes
        # Diagnostic only, so the increments are unlocked: the loop thread and
        # the batch thread both land here and a lost count costs nothing. See
        # _loop_health for why this number is worth having at all.
        ms = (time.monotonic() - t_view) * 1000.0
        self._view_calls += 1
        self._view_ms += ms
        self._view_max_ms = max(self._view_max_ms, ms)
        return dc_view

    async def _dispatch_to_dc(self, req_id, dc_id, request, timeout=2000.0):
        """Dispatch directly to a datacenter."""
        stub = self._get_dc_stub(dc_id)
        return await stub.Dispatch(
            inference_pb2.DispatchRequest(
                req_id=req_id,
                in_tok=request.in_tok,
                out_tok=request.out_tok,
                reply_to=self.sched_id,
                dataset=request.dataset,
                predicted_osl=int(getattr(request, "predicted_osl", 0) or 0),
            ),
            timeout=timeout,
        )

    def _osl_for(self, request) -> int:
        """The output length to price this placement on."""
        osl, fell_back = resolve_osl(request.out_tok,
                                     getattr(request, "predicted_osl", 0),
                                     self._osl_source, self._ref_osl)
        if fell_back:
            self._osl_fallbacks += 1
            if not self._osl_warned:
                self._osl_warned = True
                log.warning(
                    "osl_source=%s but the request carries no such length; "
                    "falling back to ref_osl=%d. A run that does this for most "
                    "requests is not measuring the source it names.",
                    self._osl_source, self._ref_osl)
        return osl

    def _log_decision(self, req_id, dc_id, stats, request, resp,
                      fwd_ms: float = 0.0, dc_view=None):
        """Record one (snapshot, outcome) pair for offline estimator scoring.

        The record now carries the estimator's OWN output beside the snapshot
        it came from. It used to carry only the snapshot, so scoring meant
        recomputing the estimate offline -- which silently rescored old runs
        with today's code and could not answer what the filter actually decided
        at the time. `est_ttft_ms` and `sla_ok` are what this scheduler
        believed when it chose.

        On a policy with no SLA constraint `sla_ok` is what a filter WOULD
        have said, not what was enforced -- those runs are controls, and the
        estimate is logged for them precisely because nothing was decided on
        it there.

        `snap_age_ms` is the point of the exercise: the estimate is computed
        from a snapshot up to a poll period old, and a prefill backlog drains
        in a few hundred ms. Whether that staleness is what makes the estimate
        wrong is a question the log could not previously be asked.
        """
        if self._decision_log is None or stats is None:
            return
        now = time.monotonic()
        model = self._log_model
        est_ttft = est_tpot = None
        sla_ok = None
        if model is not None:
            try:
                sla_ok, est_ttft, est_tpot = _check_sla(
                    stats, int(request.in_tok),
                    self._log_ttft_budget, self._log_tpot_budget,
                    model, fwd_ms=fwd_ms, max_num_seqs=self._max_num_seqs,
                    req_osl=self._osl_for(request))
            except Exception:                       # never lose a request to logging
                pass
        arrived = self._dc_arrived.get(dc_id)
        _n_proj = getattr(_proj_note, "by_dc", {}).get(dc_id, (0, 0, 0.0))
        rec = {
            "req_id": req_id, "dc_id": dc_id,
            # What the forwarding scheduler asked for. Equal to dc_id unless
            # the site was infeasible on arrival and _respot moved it, so the
            # substitution rate is `target_dc != dc_id` over the forwarded set.
            "target_dc": request.target_dc,
            "isl": request.in_tok, "osl": request.out_tok,
            "fwd_ms": round(fwd_ms, 3),
            # when, and how old the view was
            "t_ms": round(now * 1000.0, 3),
            "snap_age_ms": (None if arrived is None
                            else round((now - arrived) * 1000.0, 3)),
            "snap_epoch": self._epoch,
            "n_view": (len(dc_view) if dc_view is not None else None),
            # what the scheduler believed at the time
            "est_ttft_ms": (None if est_ttft is None else round(est_ttft, 3)),
            "est_tpot_ms": (None if est_tpot is None else round(est_tpot, 4)),
            "sla_ok": sla_ok,
            # the snapshot as the policy saw it
            "num_gpus": stats.num_gpus,
            "active": stats.active_requests,
            "pending": stats.pending_requests,
            "prefilling": stats.prefilling_requests,
            # The correction actually applied to est_tpot_ms. Logged because
            # the residual bias could not be attributed without it.
            "duty": round(float(getattr(stats, 'prefill_duty', 0.0) or 0.0), 4),
            # Running count of requests whose OSL source was unavailable and
            # fell back to ref_osl. It was counted and warned about ONCE, so a
            # run could fall back on every request and look the same as one
            # that fell back on one. A run that does this for most requests is
            # not measuring the source it names.
            "osl_fb": self._osl_fallbacks,
            # What the LOCAL correction contributed to the numbers above, so a
            # run can be asked whether the projection fired at all rather than
            # only whether the estimate moved. n_proj is how many of this
            # scheduler's own in-flight dispatches were folded in, n_proj_f how
            # many of those were classified as still prefilling, and stage_b
            # the per-DC boundary (ms) that classified them.
            "n_proj": _n_proj[0], "n_proj_f": _n_proj[1],
            "stage_b": round(_n_proj[2], 1),
            "decoding": stats.decoding_requests,
            "pending_isl": stats.total_pending_isl,
            "prefill_isl": stats.total_prefill_isl,
            "active_isl": stats.total_active_isl,
            "generated": stats.total_generated_tokens,
            "remaining_osl": stats.total_remaining_osl,
            # The KV gate's wait as the site forecast it; m_queue_ms below
            # ends at prefill admission, so the two can be scored together.
            "kv_queue_ms": round(stats.kv_queue_ms, 1),
            "kv_ms_per_token": round(stats.kv_ms_per_token, 4),
            "kv_room_tokens": round(stats.kv_room_tokens),
            "pue": round(float(getattr(stats, "pue", 0.0) or 0.0), 3),
            # what actually happened to this request there
            # What the engine ACTUALLY fused this request's prefill with,
            # against the `prefilling`/`prefill_isl` the estimate assumed it
            # would share a pass with. The two are not the same number and the
            # gap between them is the fused-prefill term's whole error.
            "m_prefill_bs": int(getattr(resp, "prefill_bs", 0) or 0),
            "m_queue_ms": round(resp.queue_ms, 3),
            "m_ttft_ms": round(resp.ttft_ms, 3),
            "m_tpot_ms": round(resp.avg_tpot_ms, 4),
        }
        # The whole candidate set, off by default: it is what shows which
        # sites the filter REJECTED, which the chosen-site record cannot. It
        # costs one model call per site per decision and roughly ten times the
        # log size, so it is opt-in via DECISION_LOG_CANDIDATES=1.
        if self._log_candidates and dc_view and model is not None:
            cands = []
            for cid, cstats in dc_view.items():
                try:
                    ok, tt, tp = _check_sla(
                        cstats, int(request.in_tok),
                        self._log_ttft_budget, self._log_tpot_budget,
                        model, fwd_ms=self._rtt_matrix.get(cid, 0.0),
                        max_num_seqs=self._max_num_seqs,
                        # The same horizon the real decision used. Without it the
                        # REJECTED candidates are scored by a different estimator
                        # than the chosen one, and the two cannot be compared --
                        # which is the entire point of logging them.
                        req_osl=self._osl_for(request))
                except Exception:
                    continue
                cands.append([cid, round(tt, 1), round(tp, 2), bool(ok)])
            rec["cands"] = cands
            rec["n_feasible"] = sum(1 for c in cands if c[3])
        with self._decision_lock:
            self._decision_log.write(json.dumps(rec) + "\n")
            self._decision_rows += 1
            if self._decision_rows >= self._FLUSH_EVERY:
                self._decision_rows = 0
                self._decision_log.flush()

    def close(self):
        """Flush what the batched writes are still holding.

        Both logs used to flush every row, so a killed run lost nothing. They
        no longer do, and the last partial block is only on disk if someone
        asks for it.
        """
        with self._csv_lock:
            if self._csv_file:
                self._csv_file.flush()
        with self._decision_lock:
            if self._decision_log:
                self._decision_log.flush()

    def close_solver(self):
        if self._solver is not None:
            self._solver.close()
            self._solver = None

    async def aclose(self):
        """Half-close the peer streams. Separate from close() because it has
        to await, and close() is called from synchronous shutdown paths."""
        for mux in self._sched_muxes.values():
            try:
                await mux.close()
            except Exception as e:                               # noqa: BLE001
                log.warning("closing peer stream failed: %s", e)

    #: Rows between flushes. Flushing every row cost a write syscall per
    #: request on the event loop; a kill -9 now loses at most this many rows,
    #: which is the trade the run can afford and the syscall was not.
    _FLUSH_EVERY = 256

    def _log_csv(self, row: dict):
        if self._csv_writer:
            _t = time.monotonic()
            with self._csv_lock:
                self._csv_writer.writerow(row)
                self._csv_rows_since_flush += 1
                if self._csv_rows_since_flush >= self._FLUSH_EVERY:
                    self._csv_rows_since_flush = 0
                    self._csv_file.flush()
            self._t_csv += time.monotonic() - _t

    # ----- gRPC handlers -----
    async def SubmitRequest(self, request, context):
        arrival_time = datetime.now(timezone.utc)
        # The caller's id wins, so one id follows the request from the load
        # generator through every hop and into DECISION_LOG. Only mint one when
        # the caller set none.
        req_id = request.req_id or self._next_req_id()
        t0 = time.monotonic()
        # Time this handler spends AWAITING rather than working. `decide` is
        # supposed to be the loop's own cost in the decision path, and it was
        # measured as plain wall clock from t0 -- which for a batch policy is
        # dominated by the buffered wait, and for a direct cross-region
        # dispatch includes the simulated RTT sleep. Neither is loop work.
        #
        # Concurrent waits also SUM: at 1200 req/s each waiting ~300 ms, the
        # counter accumulated 360 s per second and `decide` printed 34085%,
        # which clamped `rest` (100 - decide - finish) to zero. Both columns
        # were unreadable for exactly the policies whose loop cost was in
        # question.
        awaited = 0.0

        with self._snap_lock:
            dc_view = dict(self._dc_view)
            rtt_matrix = dict(self._rtt_matrix)
        self._note_request_size(request)
        dc_view = self._view_with_forwarded(dc_view)

        # ── Scheduling decision (measure scheduling_ms) ──
        t_sched_start = time.monotonic()

        # Non-zero only when a cross-region placement has to be dispatched
        # directly because no scheduler owns the target group; see the
        # partial-fleet fallback below. Added to forwarding_ms and simulated
        # around the dispatch.
        direct_rtt_ms = 0.0

        # For batch ILP policies: buffer non-forwarded requests
        if self._is_batch_policy and not request.forwarded and dc_view:
            result_future = asyncio.get_running_loop().create_future()
            result_holder = {}
            with self._batch_lock:
                # Include t_sched_start so the batch loop can record the
                # full scheduling latency (buffer wait + LP solve), not
                # just the solve time.
                self._batch_buffer.append(
                    (req_id, request, result_future, result_holder, t_sched_start)
                )
                # Wake the batch thread early once the buffer is full, rather
                # than making a burst wait out the whole window and arrive as
                # one oversized problem. Set under the lock the loop drains
                # under, so the count it reacts to is the count it will see.
                if self._batch_window_size and \
                        len(self._batch_buffer) >= self._batch_window_size:
                    self._batch_ready.set()
            # Wait for batch solver to assign a DC. The two ways this ends
            # are NOT the same failure and used to share one message.
            #
            # The batch loop assigns a round-robin fallback to any request the
            # policy left out, so `dc_id is None` after the event fires would
            # mean the fallback itself found no datacentre. If the event never
            # fires, the request simply outwaited the loop -- the solver may be
            # perfectly healthy and merely behind, which is a throughput
            # problem in the scheduler, not a modelling one in the policy.
            # Measured on grid_v1: facility_ilp lost 8.0% of its requests at
            # 1200 req/s this way, reported as "failed to assign DC".
            #
            # 30 s, not 5. The bound exists for a solver that is STUCK, and
            # since the solve moved to its own process with an in-process
            # fallback behind it, a request waiting five seconds no longer
            # means that -- it means the loop is congested. Deleting those
            # requests biases the result in the worst possible direction: they
            # are not a random 0.1%, they are the ones that arrived during the
            # worst cycles, so dropping them removes exactly the tail the
            # latency percentiles are meant to report. A congested scheduler
            # should show up as latency and as SLA violations, both of which
            # are measured, rather than as requests that never existed.
            # Measured at 4800 req/s: 806 of 957,340 lost this way, on a run
            # whose solver fallback count was zero.
            _t_wait = time.monotonic()
            try:
                await asyncio.wait_for(result_future,
                                       timeout=self._batch_wait_s)
                assigned = True
            except asyncio.TimeoutError:
                assigned = False
            awaited += time.monotonic() - _t_wait
            # Order matters below: on timeout the batch thread may still be
            # writing result_holder, so dc_id is read and the error returned
            # before anything else touches it. Nothing here may start reading
            # the holder earlier.
            scheduling_ms = result_holder.get("scheduling_ms", 0)
            dc_id = result_holder.get("dc_id")
            is_local = result_holder.get("is_local", True)
            if dc_id is None:
                return inference_pb2.InferenceResponse(
                    req_id=req_id, status="ERROR",
                    error=("batch solver returned no DC (round-robin fallback "
                           "found none either)" if assigned else
                           f"batch solver did not answer within "
                           f"{self._batch_wait_s:.0f}s; the request was "
                           f"buffered and never reached a solve"),
                )
            # A cross-region assignment goes through the scheduler that owns
            # the target group, like every other policy's does. It used to
            # dispatch straight to the remote datacentre and sleep the RTT
            # itself, which charged the right latency but left the owning
            # scheduler blind to the load until its next poll. Falls through to
            # the forward branch below; the RTT accounting is identical
            # (2 x rtt either way).
        elif not dc_view:
            # No snapshot has arrived, so no policy can run: there is nothing
            # to score. This is a BROKEN RUN, not a placement decision, and it
            # used to be neither logged nor spread.
            #
            # `sorted(...)[0]` sent every request to the alphabetically first
            # datacentre. On 2026-09-07 a provisioner failed to bind its port,
            # no snapshot was ever pushed, and carbon_aware:0.5 at 1200 req/s
            # put 100% of its load on AU_VIC -- one of the dirtiest sites in
            # the fleet -- while reporting zero errors. The cell was caught
            # only because it ran for 622 s instead of 220.
            #
            # Round-robin instead, so a partial outage degrades into a spread
            # rather than into a single site's carbon number, and LOUD, so the
            # next one is found by reading the log rather than by noticing a
            # run took three times too long. Rate-limited: at 1200 req/s an
            # unthrottled error line per request is its own outage.
            self._refresh_dc_endpoints()
            if self._dc_endpoints:
                dc_id = self._batch_rr.pick(
                    {d: None for d in self._dc_endpoints})
                is_local = True
                self._no_view_reqs += 1
                now = time.monotonic()
                if now - self._no_view_logged > 5.0:
                    self._no_view_logged = now
                    log.error(
                        "NO SNAPSHOT: %d requests placed with an empty view. "
                        "The policy has not run at all -- check that the "
                        "provisioner started. Placements are round-robin and "
                        "this run's carbon numbers mean nothing.",
                        self._no_view_reqs)
            else:
                return inference_pb2.InferenceResponse(
                    req_id=req_id, status="ERROR",
                    error="No datacenters available",
                )
        elif request.forwarded:
            # The forwarding scheduler already chose. Honour it.
            #
            # This used to re-decide with choose_dc_local_only, so a global
            # policy's choice only ever selected the GROUP: carbon_aware could
            # name the greenest site in the fleet and have the far end quietly
            # substitute another one of its own. Nothing recorded the original
            # intent, so the substitution was invisible in every result -- for
            # exactly the cross-region population those policies exist to
            # create.
            dc_id = request.target_dc
            if dc_id:
                # Not trusted: a request must name a site in THIS group. One
                # naming a third group is the shape of the leak where a second
                # hop is taken and charged nothing, and there is no sensible
                # recovery -- the RTT for that hop has already been paid to the
                # wrong pair.
                named_group = REGION_TO_GROUP.get(dc_id)
                if named_group != self.scheduler_group:
                    log.error("forwarded req=%s names %s (group=%s), not mine "
                              "(%s)", req_id, dc_id, named_group,
                              self.scheduler_group)
                    return inference_pb2.InferenceResponse(
                        req_id=req_id, status="ERROR",
                        error=(f"forwarded request names {dc_id} in group "
                               f"{named_group}, but this is "
                               f"{self.scheduler_group}"),
                    )
                if self._arrival_recheck:
                    alt = self._respot(dc_id, dc_view, request)
                    if alt is not None:
                        dc_id = alt
            else:
                # A caller that predates target_dc. Decide locally, as before.
                dc_id = self.policy.choose_dc_local_only(
                    dc_view, self.scheduler_group,
                    req_isl=request.in_tok, req_osl=self._osl_for(request))
            # True by construction: a forwarded request is dispatched here and
            # cannot cross another region.
            is_local = True
            if dc_id is None:
                return inference_pb2.InferenceResponse(
                    req_id=req_id, status="ERROR",
                    error="Policy returned no DC",
                )
        elif self.policy.scope in (PolicyScope.GLOBAL, PolicyScope.GLOBAL_WITH_LOCAL):
            result = self.policy.choose_dc_extended(
                dc_view, self.scheduler_group, rtt_matrix=rtt_matrix,
                req_isl=request.in_tok, req_osl=self._osl_for(request),
            )
            if result is None:
                return inference_pb2.InferenceResponse(
                    req_id=req_id, status="ERROR",
                    error="Policy returned no DC",
                )
            dc_id, is_local = result
        else:
            # LOCAL_ONLY. The scope is enforced here rather than trusted: the
            # server dispatches this path directly and charges no RTT, so a
            # policy that declares LOCAL_ONLY and then reads the whole view
            # places requests in other regions for free. weighted_least_loaded
            # did exactly that -- the only one of the four that did not filter
            # itself. Filtering an already-filtered view costs nothing, and a
            # new policy now gets the behaviour its scope promises.
            local_view = {k: v for k, v in dc_view.items()
                          if k in self.policy._local_regions}
            dc_id = self.policy.choose_dc(local_view or dc_view,
                                          self.scheduler_group,
                                          req_isl=request.in_tok,
                                          req_osl=self._osl_for(request))
            is_local = True
            if dc_id is None:
                return inference_pb2.InferenceResponse(
                    req_id=req_id, status="ERROR",
                    error="Policy returned no DC",
                )

        if not (self._is_batch_policy and not request.forwarded):
            scheduling_ms = (time.monotonic() - t_sched_start) * 1000.0

        # A cross-region placement needs the scheduler that owns the target
        # group. In a deliberately partial fleet that scheduler may not be
        # running while its datacentres are -- the provisioner still polls
        # them, so they are in dc_view and a global policy can pick them. That
        # placement used to succeed by dispatching directly, and turning it
        # into an error would break a configuration that works today. Dispatch
        # it directly and charge the same RTT, but say so.
        if not is_local and self._find_sched_for_dc(dc_id) is None:
            dc_group = REGION_TO_GROUP.get(dc_id, self.scheduler_group)
            if dc_group not in self._warned_no_sched:
                self._warned_no_sched.add(dc_group)
                log.warning(
                    "no scheduler owns group %s; dispatching to its "
                    "datacentres directly. RTT is charged, but %s cannot see "
                    "the load until it is polled.", dc_group, dc_group)
            direct_rtt_ms = rtt_matrix.get(
                rtt_pair_key(self.scheduler_group, dc_group), 0.0)
            if self.simulate_rtt:
                _t_rtt = time.monotonic()
                await asyncio.sleep(direct_rtt_ms / 2000.0)
                awaited += time.monotonic() - _t_rtt
            is_local = True

        # ── Dispatch ──
        self._t_decide += max(0.0, (time.monotonic() - t0) - awaited)
        forwarding_ms = 0.0
        carbon_intensity = 0.0

        if is_local:
            # Every dispatch is tracked, not just the forwarded ones: no
            # policy here remembers what it sent between polls. See _fwd_begin.
            tracked = True
            self._fwd_begin(req_id, dc_id, request)
            try:
                resp = await self._dispatch_to_dc(req_id, dc_id, request)
            except Exception as e:
                log.error("Dispatch to %s failed: %s", dc_id, e)
                return inference_pb2.InferenceResponse(
                    req_id=req_id, status="ERROR", error=str(e),
                )
            finally:
                if tracked:
                    self._fwd_end(req_id, dc_id)
            t_fin = time.monotonic()
            self._note_stage_boundary(dc_id, resp.queue_ms, resp.ttft_ms)
            # Return-path RTT, when this was a direct cross-region dispatch
            if direct_rtt_ms > 0 and self.simulate_rtt:
                await asyncio.sleep(direct_rtt_ms / 2000.0)
            # The matrix holds a ROUND TRIP, so each leg is half of it and
            # the two together are the whole. See common/network.py.
            forwarding_ms = direct_rtt_ms
            if dc_id in dc_view:
                carbon_intensity = dc_view[dc_id].carbon_intensity

            latency_ms = (time.monotonic() - t0) * 1000.0
            completion_time = datetime.now(timezone.utc)
            if self.verbosity >= 1:
                log.info(
                    "COMPLETE req=%s dc=%s svc=%.1fms ttft=%.1fms tpot=%.2fms lat=%.1fms sched=%.1fms fwd=%.1fms",
                    req_id, resp.dc_id, resp.service_ms,
                    resp.ttft_ms, resp.avg_tpot_ms,
                    latency_ms, scheduling_ms, forwarding_ms,
                )
            self._log_decision(req_id, dc_id, dc_view.get(dc_id), request,
                               resp, fwd_ms=forwarding_ms, dc_view=dc_view)
            self._log_csv({
                "req_id": req_id,
                "dc_id": resp.dc_id,
                "remote_sched": "",
                "in_tok": request.in_tok,
                "out_tok": request.out_tok,
                "forwarded": request.forwarded,
                "service_ms": round(resp.service_ms, 2),
                "latency_ms": round(latency_ms, 2),
                "ttft_ms": round(resp.ttft_ms, 2),
                "avg_tpot_ms": round(resp.avg_tpot_ms, 4),
                "queue_ms": round(resp.queue_ms, 2),
                "forwarding_ms": round(forwarding_ms, 2),
                "scheduling_ms": round(scheduling_ms, 2),
                "carbon_intensity": round(carbon_intensity, 2),
                "arrival_time": arrival_time.strftime("%H:%M:%S.%f"),
                "completion_time": completion_time.strftime("%H:%M:%S.%f"),
            })
            _r = inference_pb2.InferenceResponse(
                req_id=req_id,
                dc_id=resp.dc_id,
                status="OK",
                service_ms=resp.service_ms,
                latency_ms=latency_ms,
                carbon_intensity=carbon_intensity,
                rtt_overhead_ms=0.0,
                sched_path=self.sched_id,
                ttft_ms=resp.ttft_ms,
                avg_tpot_ms=resp.avg_tpot_ms,
                queue_ms=resp.queue_ms,
                prefill_bs=resp.prefill_bs,
                forwarding_ms=forwarding_ms,
                scheduling_ms=scheduling_ms,
            )
            self._t_finish += time.monotonic() - t_fin
            return _r
        else:
            # Forward to remote scheduler
            remote_sched_id = self._find_sched_for_dc(dc_id)
            if remote_sched_id is None:
                log.error("No remote scheduler for DC %s", dc_id)
                return inference_pb2.InferenceResponse(
                    req_id=req_id, status="ERROR",
                    error=f"No scheduler for remote DC {dc_id}",
                )

            dc_group = REGION_TO_GROUP.get(dc_id, self.scheduler_group)
            key = rtt_pair_key(self.scheduler_group, dc_group)
            rtt_ms = rtt_matrix.get(key, 0.0)

            if self.simulate_rtt:
                await asyncio.sleep(rtt_ms / 2000.0)

            try:
                sched_stub = self._get_sched_stub(remote_sched_id)
                remote_resp = await sched_stub.SubmitRequest(
                    inference_pb2.InferenceRequest(
                        in_tok=request.in_tok,
                        out_tok=request.out_tok,
                        req_id=req_id,
                        user_id=request.user_id,
                        forwarded=True,
                        dataset=request.dataset,
                        # The whole point of the hop: the far end dispatches
                        # HERE rather than deciding again.
                        target_dc=dc_id,
                        # Carried across the hop, or the receiving scheduler
                        # sees no prediction and silently falls back to the
                        # flat reference length for exactly the requests that
                        # crossed a region -- which is the population the
                        # carbon-aware policies move the most.
                        predicted_osl=request.predicted_osl,
                    ),
                    timeout=2000.0,
                )
            except Exception as e:
                log.error("Forward to %s failed: %s", remote_sched_id, e)
                return inference_pb2.InferenceResponse(
                    req_id=req_id, status="ERROR", error=str(e),
                )

            if self.simulate_rtt:
                await asyncio.sleep(rtt_ms / 2000.0)

            t_fin = time.monotonic()
            # The invariant the target_dc field exists to establish. A
            # mismatch means the far end substituted a datacentre, which is
            # what this replaced -- worth a line rather than silence, because
            # every carbon number for this request would then describe a site
            # the policy did not choose.
            if remote_resp.status == "OK" and remote_resp.dc_id != dc_id:
                log.error("req=%s: asked %s for %s, it used %s",
                          req_id, remote_sched_id, dc_id, remote_resp.dc_id)

            # The matrix holds a ROUND TRIP, so each leg is half of it and
            # the two together are the whole. See common/network.py.
            forwarding_ms = rtt_ms
            carbon_intensity = remote_resp.carbon_intensity

            latency_ms = (time.monotonic() - t0) * 1000.0
            completion_time = datetime.now(timezone.utc)
            if self.verbosity >= 1:
                log.info(
                    "FORWARD req=%s -> %s dc=%s svc=%.1fms fwd=%.1fms sched=%.1fms lat=%.1fms",
                    req_id, remote_sched_id, remote_resp.dc_id,
                    remote_resp.service_ms, forwarding_ms, scheduling_ms, latency_ms,
                )
            # The deciding scheduler's own record. Only the local branch used
            # to write one, so a cross-region request produced no decision at
            # sched-A at all -- only sched-B's, attributed to sched-B. The
            # snapshot recorded here is the one THIS scheduler chose against,
            # which is what an offline estimator score needs.
            if remote_resp.status == "OK":
                self._log_decision(req_id, dc_id, dc_view.get(dc_id), request,
                                   remote_resp, fwd_ms=forwarding_ms, dc_view=dc_view)
            self._log_csv({
                "req_id": req_id,
                "dc_id": remote_resp.dc_id,
                "remote_sched": remote_sched_id,
                "in_tok": request.in_tok,
                "out_tok": request.out_tok,
                "forwarded": request.forwarded,
                "service_ms": round(remote_resp.service_ms, 2),
                "latency_ms": round(latency_ms, 2),
                "ttft_ms": round(remote_resp.ttft_ms, 2),
                "avg_tpot_ms": round(remote_resp.avg_tpot_ms, 4),
                "queue_ms": round(remote_resp.queue_ms, 2),
                "forwarding_ms": round(forwarding_ms, 2),
                "scheduling_ms": round(scheduling_ms, 2),
                "carbon_intensity": round(carbon_intensity, 2),
                "arrival_time": arrival_time.strftime("%H:%M:%S.%f"),
                "completion_time": completion_time.strftime("%H:%M:%S.%f"),
            })
            remote_path = remote_resp.sched_path
            sched_path = f"{self.sched_id},{remote_path}" if remote_path else self.sched_id
            _r = inference_pb2.InferenceResponse(
                req_id=req_id,
                dc_id=remote_resp.dc_id,
                status=remote_resp.status,
                service_ms=remote_resp.service_ms,
                latency_ms=latency_ms,
                error=remote_resp.error,
                carbon_intensity=carbon_intensity,
                rtt_overhead_ms=forwarding_ms,
                sched_path=sched_path,
                ttft_ms=remote_resp.ttft_ms,
                avg_tpot_ms=remote_resp.avg_tpot_ms,
                queue_ms=remote_resp.queue_ms,
                prefill_bs=remote_resp.prefill_bs,
                forwarding_ms=forwarding_ms,
                scheduling_ms=scheduling_ms,
            )
            self._t_finish += time.monotonic() - t_fin
            return _r

    async def SubmitStream(self, request_iterator, context):
        """Many submissions over one HTTP/2 stream, scheduled independently.

        Every message is handed to `SubmitRequest` unchanged, so there is one
        scheduling path and not two: this method owns concurrency and ordering
        and nothing else. `context` is per-stream rather than per-message here,
        which is only sound because `SubmitRequest` never reads it.

        Replies leave in COMPLETION order. A request placed next door and one
        forwarded across the planet differ by hundreds of milliseconds, and
        writing them in arrival order would make every reply wait for the
        slowest one ahead of it -- the head-of-line blocking that the unary path
        does not have. Callers match on `req_id`.

        Termination is the part that is easy to get wrong. The stream is done
        when the client has half-closed AND no scheduled request is still
        owed a reply; `pending` counts the latter and is decremented only
        AFTER the reply is on the queue, never before, or a task sitting
        between its dispatch and its `put` would look finished and the reply
        would be dropped. An empty queue is not evidence of anything.
        """
        out: asyncio.Queue = asyncio.Queue()
        tasks: set = set()
        state = {"reading": True, "pending": 0}
        done = object()

        def finish_if_drained():
            if not state["reading"] and state["pending"] == 0:
                out.put_nowait(done)

        async def serve(msg):
            try:
                resp = await self.SubmitRequest(msg, context)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # One request's failure must not tear down the stream that
                # every other request on this connection is using.
                log.exception("SubmitStream: req=%s failed", msg.req_id)
                resp = inference_pb2.InferenceResponse(
                    req_id=msg.req_id, status="ERROR", error=str(e),
                )
            await out.put(resp)
            state["pending"] -= 1
            finish_if_drained()

        async def read():
            try:
                async for msg in request_iterator:
                    # Incremented BEFORE the task exists, so there is no window
                    # in which a message is in flight and uncounted.
                    state["pending"] += 1
                    t = asyncio.create_task(serve(msg))
                    tasks.add(t)
                    t.add_done_callback(tasks.discard)
            finally:
                # Reached on a clean half-close and on a client that vanished
                # mid-stream alike; either way nothing more is arriving.
                state["reading"] = False
                finish_if_drained()

        reader = asyncio.create_task(read())
        try:
            while True:
                item = await out.get()
                if item is done:
                    break
                yield item
        finally:
            # The client is gone: stop reading and drop the work it will never
            # collect, rather than leaving tasks to finish into a dead queue.
            reader.cancel()
            for t in list(tasks):
                t.cancel()

    async def PushSnapshot(self, request, context):
        """Merge an incoming snapshot into the local DC view.

        The provisioner may send partial snapshots (async/staggered mode)
        containing only the DCs refreshed in the current wave, so we
        merge rather than replace. In sync mode the provisioner still
        sends a full snapshot every epoch, and merging is equivalent to
        replacing (all keys are overwritten). The RTT matrix is always
        recomputed from the full fleet cache on the provisioner side so
        it's safe to replace the scheduler's local copy on every push.
        """
        arrived = time.monotonic()
        with self._snap_lock:
            if request.epoch > self._epoch:
                self._epoch = request.epoch
                # Merge DCs from the incoming wave into the existing view
                for dc_id, stats in request.dc_view.items():
                    self._dc_view[dc_id] = stats
                    self._dc_arrived[dc_id] = arrived
                    self._update_rates(dc_id, stats, arrived)
                # Anything dispatched before this arrived was already running
                # when the datacentre was polled, so its counters include it
                # and the local correction would count it twice. Drop those --
                # but only for the datacentres THIS wave refreshed. In async
                # mode a snapshot carries one wave, and a global sweep would
                # throw away corrections for sites whose counters are still
                # pre-dispatch. The residual error is the poll-to-push gap,
                # measured at 4-5 ms.
                with self._fwd_lock:
                    for dc_id in request.dc_view:
                        at_dc = self._fwd_inflight.get(dc_id)
                        if not at_dc:
                            continue
                        for rid in [r for r, v in at_dc.items()
                                    if v[0] < arrived]:
                            self._fwd_drop(dc_id, at_dc.pop(rid))
                        if not at_dc:
                            del self._fwd_inflight[dc_id]
                # RTT matrix: replace entirely (always recomputed globally)
                if len(request.rtt_matrix) > 0:
                    self._rtt_matrix = dict(request.rtt_matrix)
                log.debug(
                    "Snapshot epoch=%d  merged_dcs=%d  total_dcs=%d  rtt_entries=%d",
                    self._epoch, len(request.dc_view),
                    len(self._dc_view), len(self._rtt_matrix),
                )
        return inference_pb2.Ack(ok=True)


async def serve():
    _kb = set_stack_size()
    if _kb:
        log.info("thread stack %d KB (SIM_THREAD_STACK_KB)", _kb)
    cfg = load_scheduler_config()

    verbosity = cfg["verbosity"]
    if verbosity == 0:
        log.setLevel(logging.WARNING)

    sched_id = cfg["sched_id"]
    scheduler_group = cfg["scheduler_group"]
    policy_name = cfg["routing_policy"]
    alpha = cfg["routing_alpha"]
    simulate_rtt = cfg["simulate_rtt"]
    port = cfg["grpc_port"]
    max_workers = cfg["grpc_max_workers"]
    log_tag = cfg["log_tag"]
    batch_window_ms = cfg["batch_window_ms"]
    batch_window_size = cfg.get("batch_window_size", 300)

    # Build policy kwargs for SLA and analytical model
    policy_kwargs = {}
    model_name = cfg.get("model_name", "")
    tp_size = cfg.get("tp_size", 1)
    if model_name:
        from datacenter.perf_models.analytical.latency import AnalyticalLatencyModel
        try:
            # Same tp as the datacentres run at, or the scheduler predicts
            # latencies for a different device count than the one serving.
            policy_kwargs["analytical_model"] = AnalyticalLatencyModel(
                model_name, tp=tp_size)
            log.info("Loaded analytical model: %s", model_name)
        except ValueError as e:
            log.warning("Failed to load analytical model: %s", e)

    ttft_budget = cfg.get("ttft_budget_ms", 0)
    tpot_budget = cfg.get("tpot_budget_ms", 0)
    bs_cap = int(cfg.get("bs_cap", 0) or 0)
    max_num_seqs = int(cfg.get("max_num_seqs", 0) or 0)
    osl_source = str(cfg.get("osl_source", "oracle") or "oracle")
    ref_osl = int(cfg.get("ref_osl", 500) or 500)
    if osl_source not in OSL_SOURCES:
        log.warning("Unknown osl_source %r -- falling back to 'oracle'. "
                    "Valid: %s", osl_source, ", ".join(OSL_SOURCES))
        osl_source = "oracle"
    if ttft_budget > 0:
        policy_kwargs["ttft_budget_ms"] = ttft_budget
    if tpot_budget > 0:
        policy_kwargs["tpot_budget_ms"] = tpot_budget
    if tp_size > 0:
        policy_kwargs["tp_size"] = tp_size
    if bs_cap > 0:
        policy_kwargs["bs_cap"] = bs_cap
    if max_num_seqs > 0:
        policy_kwargs["max_num_seqs"] = max_num_seqs
    policy_kwargs["osl_source"] = osl_source
    policy_kwargs["ref_osl"] = ref_osl
    policy_kwargs["peer_scale"] = int(cfg.get("peer_scale", 1) or 1)

    ilp_max_iter = cfg.get("ilp_max_iterations", 3)
    policy_kwargs["max_iterations"] = ilp_max_iter

    log.info(
        "Starting scheduler %s  group=%s  policy=%s  alpha=%.2f  simulate_rtt=%s  "
        "ttft_budget=%s  tpot_budget=%s  bs_cap=%s/engine  max_num_seqs=%s  "
        "osl=%s(ref=%d)  batch=%dms",
        sched_id, scheduler_group, policy_name, alpha, simulate_rtt,
        ttft_budget, tpot_budget, bs_cap, max_num_seqs or "unknown",
        osl_source, ref_osl, batch_window_ms,
    )

    servicer = SchedulerServicer(
        sched_id, scheduler_group, policy_name, alpha, simulate_rtt,
        verbosity=verbosity, log_tag=log_tag,
        batch_window_ms=batch_window_ms,
        batch_window_size=batch_window_size,
        policy_kwargs=policy_kwargs,
        # The solver process rebuilds the analytical model from these rather
        # than receiving the fitted object; see policies/common/batch_worker.
        model_spec=(model_name, tp_size),
    )
    # Started here rather than in __init__: it is a task now, so it needs the
    # running loop that only exists inside serve().
    servicer.start_batch()

    # TWO servers, ONE event loop, one servicer.
    #
    # The split used to be about thread pools. A request this scheduler OWNS
    # forwards out and used to hold its worker until a peer answered; a
    # request forwarded IN only waits on a datacentre and never forwards
    # again. Sharing one pool, the first class starved the second: the peer's
    # forward timed out, its own pool filled with threads waiting on us, and
    # both schedulers stopped answering while every datacentre sat idle.
    #
    # Under asyncio there are no workers to starve -- a request waiting on a
    # peer or a datacentre holds a coroutine, not a thread -- so the split no
    # longer defends against anything. Both ports stay because peer_addr() is
    # a wire convention: a scheduler that stopped listening on port+1000 would
    # be unreachable to every peer still dialling it.
    #
    # What the conversion is actually for, measured on the 512-engine fleet at
    # rate 2400: service_ms flat (p50 1445 ms, the same as at 2000) while
    # wall_ms - latency_ms -- the wait for a worker, which no counter reported
    # because scheduling_ms starts inside the handler -- went from 1.3 ms to
    # 7453 ms at p50. The fleet was never the limit; the thread-per-request
    # scheduler was.
    loop = asyncio.get_running_loop()
    servicer._loop = loop

    server = grpc.aio.server()
    inference_pb2_grpc.add_SchedulerServiceServicer_to_server(servicer, server)
    bind_with_retry(server, port)
    await server.start()

    peer_port = port + SchedulerServicer.PEER_PORT_OFFSET
    peer_server = grpc.aio.server()
    inference_pb2_grpc.add_SchedulerServiceServicer_to_server(servicer, peer_server)
    bind_with_retry(peer_server, peer_port)
    await peer_server.start()
    log.info("Peer forwards on port %d", peer_port)
    log.info("Listening on port %d (asyncio; grpc_max_workers=%d is unused)",
             port, max_workers)

    health = asyncio.create_task(servicer._loop_health())

    # The sweep stops a scheduler with SIGTERM and follows with SIGKILL five
    # seconds later. Python's default SIGTERM ends the process outright, so no
    # `finally` runs -- which was harmless while both logs flushed every row
    # and is not now that they flush every 256. Ask the loop to stop instead,
    # and let the code below flush on the way out.
    stopping = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stopping.set)
        except (NotImplementedError, RuntimeError):
            pass        # not a platform with signal handlers on the loop

    terminated = asyncio.create_task(server.wait_for_termination())
    try:
        await asyncio.wait([terminated, asyncio.create_task(stopping.wait())],
                           return_when=asyncio.FIRST_COMPLETED)
    finally:
        health.cancel()
        if servicer._batch_task is not None:
            servicer._batch_task.cancel()
        servicer.close()
        servicer.close_solver()
        await servicer.aclose()
        await server.stop(grace=1.0)
        await peer_server.stop(grace=1.0)


if __name__ == "__main__":
    asyncio.run(serve())
