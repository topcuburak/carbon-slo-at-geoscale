"""The node: one box, its inference engines, and the within-node scheduler.

The middle of three layers. `Datacenter` hands a node a request and asks it for
statistics; the node places that request on one of its own engines and answers
from them. Neither direction skips a level -- the datacentre never touches an
engine, and an engine never knows which node it is on.

A node is three things at once, and that is why it is its own layer rather than
a list inside the datacentre:

  a slicing of cards into engines   `num_gpus // tp_size` tensor-parallel
                                    groups, each configuring itself from
                                    tp_size: measured KV capacity, the fitted
                                    per-step energy model, and per-card power.
  a power domain                    HOST_POWER_W is everything on the box that
                                    is not a card, charged once per node.
  a scheduling domain               the WITHIN-NODE scheduler, one instance per
                                    node, so one node's rotation cannot be
                                    perturbed by another's.

What a node does NOT model is interference between its own engines, and the
measurement says it should not have to: one TP=2 engine at 8 req/s reads TPOT
p50 29.45 ms, two at the same per-engine rate read 30.14 ms, and at p90 the two
are 35.65 and 35.68.
"""

import logging
import os
import threading
from collections import defaultdict
from typing import Dict, List, Tuple

from datacenter.inference_engine import GPU
from datacenter.perf_models.base import LatencyBackend
from datacenter.scheduling import make_scheduler

log = logging.getLogger(__name__)


# Per-node power drawn outside the GPUs: CPU, DRAM, fans, NICs, drives and
# PSU conversion loss. Allocated equally across the node's GPU pairs so the
# existing per-pair energy accounting stays one sum. Drawn for the whole
# window regardless of GPU activity.
#
# Measured on the testbed node (single-socket EPYC 75F3, 4x A100-40GB) by
# reading the RAPL package domain, once with the node idle and once under a
# TP=2 decode load:
#
#   CPU package      124.6 W idle -> 145.6 W loaded   MEASURED (RAPL)
#     of which cores   0.4 W ->       2.7 W
#   DRAM                    ~60 W                     estimated, no RAPL domain
#   fans, NICs, drives      ~50 W                     estimated, not instrumented
#   PSU conversion loss     ~45 W                     estimated, scales with draw
#   ------------------------------------------------
#   total                  ~300 W
#
# Two things that measurement settled. The constant-power STRUCTURE is right:
# GPU draw rose 2.6x between the two readings while the package moved only
# 17%, and nearly all of that came from uncore (memory controllers, Infinity
# Fabric, PCIe feeding the GPUs) rather than cores. And the previous default
# of 500 W was high: the measurable part accounts for 29% of it, and the
# estimates above do not close the remaining gap.
#
# Only the first line is measured. The rest are estimates for parts this node
# does not instrument, so the paper must present them as such. Host power is
# charged per unit TIME, not per unit work, so it favours whichever policy
# finishes sooner -- run the sensitivity sweep (300/400/500/800) rather than
# treating any single value as settled.
HOST_POWER_W = float(os.environ.get("HOST_POWER_W", "300.0"))


# The WITHIN-NODE scheduler: node -> inference engine. The default keeps the
# behaviour every result so far was produced under.
WITHIN_NODE_POLICY = os.environ.get("WITHIN_NODE_POLICY", "round_robin")


class Node:
    """One compute node: the datacentre above it, inference engines below.

    A node owns *num_gpus* physical cards and slices them into
    ``num_gpus // tp_size`` **inference engines**, each a tensor-parallel group
    of *tp_size* cards that holds its own batch, its own KV cache and its own
    scheduling loop. `tp_size` is therefore not a label -- it decides how many
    engines exist, and each engine configures itself from it: measured KV
    capacity for (model, TP), the fitted per-step energy model for (model, TP),
    and idle and prefill draw, which are per-card constants scaled by the
    cards in the group. A `tp_size` with no fitted artifact fails at
    construction rather than falling back to a plausible wrong number.

    Those are the only two directions it talks in. From above, the datacentre
    hands it a request through `dispatch`; from below, it places that request
    on one of its engines using its own policy -- the third of the simulator's
    three schedulers, after request -> datacentre and datacentre -> node. The
    policy instance belongs to this node alone, so its rotation is unaffected
    by what any other node is doing.

    It also carries the node's host draw. `HOST_POWER_W` is everything on the
    box that is not a card, charged once per node and split across its engines,
    so the total is invariant of TP.

    A node does not model interference between its own engines, and the
    measurement says it should not have to. Comparing like with like -- same TP,
    same per-engine rate, near-identical prompts -- one TP=2 engine at 8 req/s
    reads TPOT p50 29.45 ms and two at 8 req/s each read 30.14 ms, with p90 at
    35.65 against 35.68. Independent is what the data supports. If a regime
    turns up where it does not, this class is where it would go.
    """

    def __init__(
        self,
        node_id: str,
        num_gpus: int,
        lookup: LatencyBackend,
        tp_size: int = 2,
        max_batch_size: int = 0,
        analytical_model=None,
        latency_mode: str = "lookup",
        model_name: str = "",
        max_model_len: int = 0,
        within_node_policy: str = "",
    ) -> None:
        self.node_id = node_id
        self.tp_size = tp_size
        num_slots = num_gpus // tp_size
        if num_slots <= 0:
            # Silent before: num_gpus < tp_size built a node with no engines,
            # which the datacentre then flattened into an empty list and only
            # failed on much later, as a modulo by zero inside dispatch, far
            # from the configuration that caused it.
            raise ValueError(
                f"{node_id}: num_gpus={num_gpus} is smaller than tp_size="
                f"{tp_size}, so the node has no engines to place work on")
        if num_gpus % tp_size:
            log.warning(
                "%s: num_gpus=%d is not a multiple of tp_size=%d; %d card(s) "
                "are unused. The node's host power is still charged in full "
                "and split across the %d engine(s) that do exist.",
                node_id, num_gpus, tp_size, num_gpus - num_slots * tp_size,
                num_slots)
        # The engines take their KV capacity, energy model and power constants
        # from tp_size, but the latency model is built by the caller and passed
        # in already fixed at some TP. Nothing used to compare the two, so a
        # node could run TP=4 engines against a TP=2 latency fit and report
        # confident numbers from two different machines.
        model_tp = getattr(analytical_model, "tp", None)
        if model_tp is not None and int(model_tp) != int(tp_size):
            raise ValueError(
                f"{node_id}: engines are built at tp_size={tp_size} but the "
                f"latency model was fitted at TP={model_tp}. Latency would come "
                f"from one configuration and energy and KV capacity from "
                f"another.")

        # Allocate the node's host power equally across its engines so the
        # existing per-engine energy accounting captures the full host draw
        # (sum across engines == HOST_POWER_W per node, invariant of TP).
        host_share = HOST_POWER_W / num_slots
        self.inference_engines: List[GPU] = [
            GPU(f"{node_id}:gpu_pair{i}", lookup, max_batch_size=max_batch_size,
                analytical_model=analytical_model, latency_mode=latency_mode,
                tp_size=tp_size, host_power_share_w=host_share,
                model_name=model_name, max_model_len=max_model_len)
            for i in range(num_slots)
        ]

        # The THIRD scheduler: node -> inference engine. Its own instance per
        # node, so one node's rotation cannot be perturbed by another's -- the
        # nodes are independent, which the flat datacentre-wide rotation they
        # replaced did not give them.
        self.within_node_policy = within_node_policy or WITHIN_NODE_POLICY
        self._within_node_scheduler = make_scheduler(self.within_node_policy)

        # The node's own view of its engines, refreshed by stats(). Held here
        # rather than recomputed by whoever asks, so the per-engine detail
        # survives one level up instead of collapsing into a site total.
        self._stats_lock = threading.Lock()
        self._stats_epoch = 0
        self._per_engine_stats: List[dict] = []
        self._node_stats: dict = {
            "num_engines": len(self.inference_engines),
            "active_requests": 0, "pending_requests": 0,
            "prefilling_requests": 0, "decoding_requests": 0,
            "pending_isl": 0, "active_isl": 0,
            "generated_tokens": 0, "remaining_osl": 0,
            "empty_engines": len(self.inference_engines),
            "max_engine_active": 0,
            "prefill_duty": 0.0,
        }

    def next_inference_engine(self) -> GPU:
        """Take the next engine from this node's placement policy.

        NOT a query. Round-robin advances its counter here, so calling this and
        discarding the answer shifts the rotation for the next real request.
        Only `dispatch` should call it.
        """
        return self._within_node_scheduler.pick(self.inference_engines)

    def dispatch(self, in_tok: int, out_tok: int, gpu: GPU = None,
                 dataset: str = "", predicted_osl: int = 0) -> Tuple[GPU, dict]:
        """Run one request on this node.  Blocks until the engine is done.

        `gpu` pins the engine and bypasses the node's policy, which is what the
        within-node replay needs: it reproduces the engine the hardware really
        used, so any disagreement is about latency and not about placement.
        """
        if gpu is None:
            gpu = self.next_inference_engine()
        result = gpu.process(in_tok, out_tok, dataset=dataset,
                             predicted_osl=predicted_osl)
        return gpu, result

    async def dispatch_async(self, in_tok: int, out_tok: int, gpu: GPU = None,
                             dataset: str = "",
                             predicted_osl: int = 0) -> Tuple[GPU, dict]:
        """`dispatch`, awaited. Same placement, same engine, same result.

        Only the wait differs: a request queued for a seat costs a coroutine
        rather than one of the datacentre's worker threads. See
        GPU.process_async.
        """
        if gpu is None:
            gpu = self.next_inference_engine()
        result = await gpu.process_async(in_tok, out_tok, dataset=dataset,
                                         predicted_osl=predicted_osl)
        return gpu, result

    # ── what the node reports upward ──────────────────────────────────────
    #
    # The datacentre asks the node; the node asks its engines. It used to
    # reach past the node into a flat list of every engine in the site, which
    # left the node with nothing to answer and no way to report anything of
    # its own -- host draw, engine interference, a failed card. Everything
    # below is the node's own aggregate over its own engines.

    @property
    def load(self) -> int:
        """Requests in flight on this node, prefilling and decoding.

        Deliberately the same name the engines use, so a placement policy can
        run at either level without being told which one it is on.
        """
        return sum(g.load for g in self.inference_engines)

    def _get_kv_capacity_tokens(self) -> int:
        """Tokens this node's engines can hold between them."""
        return sum(g._get_kv_capacity_tokens() for g in self.inference_engines)

    @property
    def kv_utilisation(self) -> float:
        """Share of this node's KV cache in use, 0.0 if capacity is unknown."""
        cap = self._get_kv_capacity_tokens()
        if cap <= 0:
            return 0.0
        return sum(g.kv_tokens for g in self.inference_engines) / cap

    def _get_stats(self) -> List[dict]:
        """Read every engine's counters, one engine at a time.

        The engines maintain these at runtime -- each is written inside the
        engine loop as the event happens -- so this only collects, it does not
        compute. The reads are plain attribute loads and take no engine lock,
        which matters because the lock they would take is the one the engine
        loop holds on every step.

        Not a consistent cut across engines: an engine read early and one read
        late are microseconds apart, and a request can move between them. The
        counters that must be exact are exact per engine, which is what the
        consumers need.
        """
        return [
            {"active": g.active_count,
             "pending": g.pending_count,
             "prefilling": g.prefill_count,
             "decoding": g.decoding_count,
             "prefill_isl": g.total_prefill_isl,
             "pending_isl": g.total_pending_isl,
             "active_isl": g.total_active_isl,
             "generated_tokens": g.total_generated_tokens,
             "remaining_osl": g.total_remaining_osl,
             # Health and the two counters that were computed and dropped.
             "faults": g.engine_faults,
             "osl_clamped": g.osl_clamped,
             "kv_blocked_steps": g.kv_blocked_steps,
             "kv_forced": g.kv_forced,
             "kv_tokens": g.kv_tokens,
             "kv_capacity_tokens": g.kv_capacity_tokens or 0}
            for g in self.inference_engines
        ]

    def _set_stats(self, per_engine: List[dict]) -> None:
        """Store a collected snapshot on the node, with its totals.

        The node holds both, and this is the point of holding them here rather
        than summing on the way past. A datacentre-wide total divided by the
        engine count cannot say how many engines were EMPTY, and that is the
        term the TTFT estimator gets wrong: an engine with nothing decoding
        imposes neither a step boundary nor an admission wait, but the mean is
        nonzero whenever any engine is busy, so the whole site gets charged for
        a wait the empty ones would not impose. Measured, splitting on it moves
        the median error from -0.08% to +30.08%. Keeping the per-engine row
        here is what would let that be answered. See SIMULATOR_TODO item 15.
        """
        agg = {"num_engines": len(per_engine),
               "active_requests": 0, "pending_requests": 0,
               "prefilling_requests": 0, "decoding_requests": 0,
               "prefill_isl": 0, "pending_isl": 0, "active_isl": 0,
               "generated_tokens": 0, "remaining_osl": 0,
               "empty_engines": 0, "max_engine_active": 0,
               "engine_faults": 0, "osl_clamped": 0, "kv_blocked_steps": 0,
               "kv_forced": 0,
               "kv_tokens": 0, "kv_capacity_tokens": 0,
               # Read off the engines directly rather than threaded through the
               # per-engine dicts: it is a property of the engine's clock, not
               # of the request populations those dicts carry.
               "prefill_duty": (sum(g.prefill_duty() for g in self.inference_engines)
                                / max(1, len(self.inference_engines))),
               # Each engine's KV gate forecast, for the site to combine; see
               # GPU.kv_forecast and meta_dc.kv_queue_terms. It takes the
               # engine lock, like prefill_duty, so it is read here.
               "kv_forecasts": [g.kv_forecast() for g in self.inference_engines
                                if g.kv_capacity_tokens],
               # Clock fidelity, summed not averaged: the ratio of the two is
               # what says whether the host distorted this run.
               "oversleep_s": sum(g.oversleep_s for g in self.inference_engines),
               "asked_s": sum(g.asked_s for g in self.inference_engines)}
        for e in per_engine:
            agg["active_requests"] += e["active"]
            agg["pending_requests"] += e["pending"]
            agg["prefilling_requests"] += e["prefilling"]
            agg["decoding_requests"] += e["decoding"]
            agg["prefill_isl"] += e["prefill_isl"]
            agg["pending_isl"] += e["pending_isl"]
            agg["active_isl"] += e["active_isl"]
            agg["generated_tokens"] += e["generated_tokens"]
            agg["remaining_osl"] += e["remaining_osl"]
            agg["engine_faults"] += e["faults"]
            agg["osl_clamped"] += e["osl_clamped"]
            agg["kv_blocked_steps"] += e["kv_blocked_steps"]
            agg["kv_forced"] += e["kv_forced"]
            agg["kv_tokens"] += e["kv_tokens"]
            agg["kv_capacity_tokens"] += e["kv_capacity_tokens"]
            if e["decoding"] == 0:
                agg["empty_engines"] += 1
            agg["max_engine_active"] = max(agg["max_engine_active"], e["active"])
        with self._stats_lock:
            self._per_engine_stats = per_engine
            self._node_stats = agg
            self._stats_epoch += 1

    def stats(self, refresh: bool = True) -> dict:
        """This node's contribution to its datacentre's snapshot.

        Collect, store, return. `refresh=False` returns the last stored
        snapshot without touching the engines, for a second reader in the same
        poll that should see the same numbers.

        Counters only. Energy is separate because it does not exist until
        compute_energy has run over a window.
        """
        if refresh:
            self._set_stats(self._get_stats())
        with self._stats_lock:
            out = dict(self._node_stats)
            out["node_id"] = self.node_id
            out["epoch"] = self._stats_epoch
            out["per_engine"] = [dict(e) for e in self._per_engine_stats]
        return out

    def compute_energy(self, sim_start: float, sim_end: float) -> None:
        for g in self.inference_engines:
            g.compute_energy(sim_start, sim_end)

    def energy_stats(self) -> dict:
        """Joules this node's engines have been charged, split by cause.

        Zero until compute_energy has run: the engines integrate over a window
        that is only known once the run is over.
        """
        return {
            "total_j": sum(g.energy_j() for g in self.inference_engines),
            "idle_j": sum(g.idle_energy_j() for g in self.inference_engines),
            "host_j": sum(g.host_energy_j() for g in self.inference_engines),
        }

    def earliest_transition(self):
        """When this node's first engine first held work, or None if never."""
        first = None
        for g in self.inference_engines:
            with g._lock:
                if g._time_series:
                    t = g._time_series[0][0]
                    if first is None or t < first:
                        first = t
        return first

    def time_at_bs_profile(self) -> Dict[int, float]:
        agg: Dict[int, float] = defaultdict(float)
        for g in self.inference_engines:
            for bs, secs in g.time_at_bs_snapshot().items():
                agg[int(bs)] += secs
        return dict(agg)

    def engine_config(self) -> dict:
        """The engine settings, for logging. Identical across this node."""
        g = self.inference_engines[0]
        return {"max_model_len": g.max_model_len,
                "max_prefill_tokens": g.max_prefill_tokens}
