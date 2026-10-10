"""The datacentre: one site, its nodes, and the within-DC scheduler.

The top of three layers, and the one the outside world talks to. The
provisioner polls it; it asks its nodes; a node asks its engines. Nothing here
reaches past a node -- placement asks a node for its load, statistics are summed
from `Node.stats`, and energy is charged through `Node.compute_energy`. There is
no flat list of every engine in the site, which is what lets a node own anything
of its own: its host draw, the engines that share it, a card that fails.

Two site properties live here because they are genuinely per-site:

  carbon intensity   gCO2eq/kWh, fixed for the length of a run. A run is around
                     two minutes of simulated time and grid intensity does not
                     move meaningfully inside that; temporal variation is
                     studied by running the same experiment at a different
                     timestamp, not by varying it mid-run.
  PUE                the facility multiplier, from common/pue_by_dc.json. It
                     multiplies every joule on the way to carbon, and because
                     it is per-site the error does not cancel between policies
                     when sites differ.

Energy is zero until `compute_energy` has run. That is not a gap: the engines
integrate over a window that is only known once the run is over, which is what
`FinalizeEnergy` is for. `PollStats` carries the same fields because it shares
this `stats()`, and no scheduler reads them.
"""

import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from common.config import PUE
from datacenter.inference_engine import GPU
from datacenter.perf_models.base import LatencyBackend
from datacenter.meta_node import Node
from datacenter.scheduling import make_scheduler

log = logging.getLogger(__name__)

# The WITHIN-DC scheduler: datacentre -> node. The default keeps the behaviour
# every result so far was produced under.
WITHIN_DC_POLICY = os.environ.get("WITHIN_DC_POLICY", "round_robin")

#: How the engines' KV gate forecasts become the site's: "mean" is what an
#: arrival expects under the default round-robin dispatch, "min" what it would
#: get from a dispatcher that picks the engine with the shortest wait.
KV_QUEUE_AGG = os.environ.get("KV_QUEUE_AGG", "mean")
if KV_QUEUE_AGG not in ("mean", "min"):
    raise ValueError(f"KV_QUEUE_AGG={KV_QUEUE_AGG!r}; expected mean or min")


def kv_queue_terms(forecasts, how: str = "") -> Tuple[float, float, float]:
    """The site's KV gate wait, from its engines' GPU.kv_forecast.

    Returns (kv_queue_ms, kv_room_tokens, kv_ms_per_token): a prompt of isl
    tokens waits kv_queue_ms + max(0, isl - kv_room_tokens) * kv_ms_per_token.
    "min" is the engine whose mean-length prompt would wait least; "mean"
    averages the three terms, which is not the mean wait exactly -- the wait is
    not linear in them -- but keeps each one's meaning.
    """
    if not forecasts:
        return 0.0, 0.0, 0.0
    if (how or KV_QUEUE_AGG) == "min":
        ahead, room, per_tok, _ = min(
            forecasts, key=lambda f: f[0] + max(0.0, f[3] - f[1]) * f[2])
        return ahead, room, per_tok
    n = len(forecasts)
    return (sum(f[0] for f in forecasts) / n, sum(f[1] for f in forecasts) / n,
            sum(f[2] for f in forecasts) / n)


class Datacenter:
    """One site: the provisioner above it, its nodes below.

    It talks to nodes and nothing further down. Placement asks a node for its
    load, statistics are summed from `Node.stats`, and energy is charged
    through `Node.compute_energy` -- there is no flat list of every engine in
    the site any more, and no method here reaches past a node to touch one.
    That is what lets a node own anything of its own: its host draw, the
    interference between the engines that share it, a card that fails.

    Placement is two decisions, not one. This class runs the WITHIN-DC
    scheduler, datacentre -> node; the node runs the WITHIN-NODE one,
    node -> engine. Both are pluggable (`datacenter/scheduling.py`) and both
    default to round-robin, which is what every result so far was produced
    under.

    The two levels are not the same as the flat rotation they replaced when
    nodes differ in size: round-robin over nodes gives each node an equal share
    of REQUESTS, while a flat rotation over engines gives each node a share
    proportional to its engine count. They coincide for identical nodes, which
    is the sweep's configuration (2 nodes x 8 cards at TP=2).
    """
    def __init__(
        self,
        dc_id: str,
        num_nodes: int,
        num_gpus_per_node: int,
        lookup: LatencyBackend,
        tp_size: int = 2,
        carbon_intensity: float = 0.0,
        max_batch_size: int = 0,
        analytical_model=None,
        latency_mode: str = "lookup",
        model_name: str = "",
        pue: float = PUE,
        max_model_len: int = 0,
        within_dc_policy: str = "",
        within_node_policy: str = "",
    ) -> None:
        self.dc_id = dc_id
        # Power usage effectiveness, PER DATACENTRE. The problem formulation
        # indexes it by site (PUE_{i,j}) and facilities genuinely differ --
        # hyperscale sites report around 1.1 while colocation is commonly
        # 1.4-1.8 -- so a single global constant made the code narrower than
        # the model it claims to implement. Each DC runs in its own process, so
        # setting PUE in that process's environment gives a per-site value; the
        # default keeps every site identical, which is the old behaviour.
        self.pue = float(pue)
        self.lookup = lookup
        self.tp_size = tp_size
        self.carbon_intensity = carbon_intensity  # gCO2eq/kWh
        self.max_batch_size = max_batch_size
        # Kept so the run-end report can read the fit's extrapolation counters.
        self.analytical_model = analytical_model
        self.nodes: List[Node] = [
            Node(f"{dc_id}:node{i}", num_gpus_per_node, lookup, tp_size,
                 max_batch_size=max_batch_size,
                 analytical_model=analytical_model,
                 latency_mode=latency_mode,
                 model_name=model_name,
                 max_model_len=max_model_len,
                 within_node_policy=within_node_policy)
            for i in range(num_nodes)
        ]

        # The SECOND scheduler: datacentre -> node. The load it compares is
        # the node's own, asked for rather than computed from the outside.
        self.within_dc_policy = within_dc_policy or WITHIN_DC_POLICY
        self._within_dc_scheduler = make_scheduler(self.within_dc_policy)

        # Cumulative arrivals, reported so a scheduler can difference it and
        # learn this site's TOTAL arrival rate rather than assuming its own
        # share of it. Not locked: it feeds a rate estimate over hundreds of
        # requests per sample, and a lock on the dispatch path would cost more
        # than the handful of increments a race can drop.
        self._admitted_total = 0

    def next_node(self) -> Node:
        """Take the next node from this datacentre's placement policy.

        NOT a query, for the same reason `Node.next_inference_engine` is not:
        round-robin advances its counter here, so calling this and throwing the
        answer away shifts the rotation for the next real request. Only
        `dispatch` should call it.
        """
        return self._within_dc_scheduler.pick(self.nodes)

    def dispatch(self, in_tok: int, out_tok: int, dataset: str = "",
                 predicted_osl: int = 0) -> dict:
        """Place one request and block for its simulated duration.

        Two decisions: this picks the node, the node picks its own engine.

        Neither is an admission check. Nothing here asks whether the chosen
        node has room -- the only bound in the path is the engine's semaphore,
        so a request placed on a busy node queues there rather than being sent
        somewhere emptier. That matches a real deployment, where admission is
        the engine's business, and it means the node policy is a preference and
        not a limit.
        """
        self._admitted_total += 1
        node = self.next_node()
        _, result = node.dispatch(in_tok, out_tok, dataset=dataset,
                                  predicted_osl=predicted_osl)
        result["dc_id"] = self.dc_id
        result["node_id"] = node.node_id
        return result

    async def dispatch_async(self, in_tok: int, out_tok: int,
                             dataset: str = "", predicted_osl: int = 0) -> dict:
        """`dispatch`, awaited. Placement is identical; only the wait changes.

        Still not an admission check -- see `dispatch`. The difference is where
        a request waits when the site is full: on the engine's seat gate, which
        `queue_ms` reports, instead of in the gRPC pool ahead of the handler,
        which nothing inside the datacentre can see.
        """
        self._admitted_total += 1
        node = self.next_node()
        _, result = await node.dispatch_async(in_tok, out_tok, dataset=dataset,
                                              predicted_osl=predicted_osl)
        result["dc_id"] = self.dc_id
        result["node_id"] = node.node_id
        return result

    def stats(self) -> dict:
        """Lightweight stats for the provisioner.

        Asked of the nodes, one level at a time. The provisioner talks to the
        datacentre, the datacentre to its nodes, the node to its engines; this
        method used to skip the middle and iterate every engine in the site.
        """
        total_active = 0
        total_pending = 0
        total_prefilling = 0
        total_decoding = 0
        total_prefill_isl = 0
        total_pending_isl = 0
        total_active_isl = 0
        total_generated_tokens = 0
        total_remaining_osl = 0
        total_engines = 0
        duty_weighted = 0.0
        total_energy_j = 0.0
        total_idle_j = 0.0
        total_host_j = 0.0
        total_empty_engines = 0
        max_engine_active = 0
        total_faults = 0
        total_osl_clamped = 0
        total_kv_blocked = 0
        total_kv_tokens = 0
        total_kv_capacity = 0
        total_kv_forced = 0
        kv_forecasts = []
        total_oversleep_s = 0.0
        total_asked_s = 0.0
        for node in self.nodes:
            n = node.stats()
            total_oversleep_s += n.get("oversleep_s", 0.0)
            total_asked_s += n.get("asked_s", 0.0)
            total_empty_engines += n["empty_engines"]
            max_engine_active = max(max_engine_active, n["max_engine_active"])
            total_faults += n["engine_faults"]
            total_osl_clamped += n["osl_clamped"]
            total_kv_blocked += n["kv_blocked_steps"]
            total_kv_forced += n.get("kv_forced", 0)
            total_kv_tokens += n["kv_tokens"]
            total_kv_capacity += n["kv_capacity_tokens"]
            kv_forecasts.extend(n.get("kv_forecasts", ()))
            total_active += n["active_requests"]
            total_pending += n["pending_requests"]
            total_prefilling += n["prefilling_requests"]
            total_decoding += n["decoding_requests"]
            total_prefill_isl += n["prefill_isl"]
            total_pending_isl += n["pending_isl"]
            total_active_isl += n["active_isl"]
            total_generated_tokens += n["generated_tokens"]
            total_remaining_osl += n["remaining_osl"]
            total_engines += n["num_engines"]
            duty_weighted += n.get("prefill_duty", 0.0) * n["num_engines"]
            e = node.energy_stats()
            total_energy_j += e["total_j"]
            total_idle_j += e["idle_j"]
            total_host_j += e["host_j"]
        kv_queue_ms, kv_room_tokens, kv_ms_per_token = kv_queue_terms(
            kv_forecasts)
        return {
            "dc_id": self.dc_id,
            "num_nodes": len(self.nodes),
            # Counts INFERENCE ENGINES, not cards: at TP=2 a node of 8 cards
            # reports 4. Every consumer divides by it to get one engine's
            # share and is right to; the name is on the wire (PollStatsResponse
            # .num_gpus) and renaming it would break the proto, so it stays and
            # this comment carries the meaning.
            "num_gpus": total_engines,
            # Engine-weighted, so a site with uneven nodes is not skewed by
            # whichever node has fewer engines.
            "prefill_duty": duty_weighted / max(1, total_engines),
            "total_active_requests": total_active,
            "total_pending_requests": total_pending,
            "total_prefilling_requests": total_prefilling,
            # Reported separately rather than left to be derived: the two
            # counters below it are summed over the decoding set, and a
            # consumer that pairs them with total_active_requests mixes two
            # different populations. active - prefilling does not recover it.
            "total_decoding_requests": total_decoding,
            "total_prefill_isl": total_prefill_isl,
            "total_pending_isl": total_pending_isl,
            "total_active_isl": total_active_isl,
            "total_generated_tokens": total_generated_tokens,
            "total_remaining_osl": total_remaining_osl,
            "admitted_requests": self._admitted_total,
            "avg_batch_size": total_active / max(total_engines, 1),
            # What avg_batch_size cannot say. An engine with nothing decoding
            # imposes neither a step boundary nor an admission wait, but the
            # mean is nonzero whenever any engine is busy, so a consumer that
            # divides by the engine count charges the whole site for a wait the
            # empty ones would not impose -- measured at +30% on the median
            # once any engine is empty. These two carry the shape the mean
            # throws away; SIMULATOR_TODO item 15 decides whether it binds.
            "empty_engines": total_empty_engines,
            "max_engine_active": max_engine_active,
            # Health, and three things the engines counted for nobody. A run
            # that ends with engine_faults > 0 dropped requests; osl_clamped
            # says how many outputs were cut to the context window; and
            # kv_blocked_steps is how the KV gate stays measured rather than
            # assumed never to bind.
            "engine_faults": total_faults,
            "osl_clamped": total_osl_clamped,
            "kv_blocked_steps": total_kv_blocked,
            # Prompts larger than the whole cache, admitted anyway. vLLM would
            # preempt or fail; the simulator models neither, so a non-zero
            # value bounds how far the run left the modelled regime.
            "kv_forced": total_kv_forced,
            "kv_tokens": total_kv_tokens,
            "kv_capacity_tokens": total_kv_capacity,
            # The KV gate's wait, priced here because only the site knows its
            # cache. See kv_queue_terms.
            "kv_queue_ms": kv_queue_ms,
            "kv_room_tokens": kv_room_tokens,
            "kv_ms_per_token": kv_ms_per_token,
            # How far the host stretched the simulated clock. The replay sleeps
            # in real time, so this is the error budget term that was computed
            # from the beginning and reported to nobody.
            "oversleep_s": total_oversleep_s,
            "asked_s": total_asked_s,
            "total_energy_j": total_energy_j,
            # The same total, split by what caused it. work = total - idle -
            # host is the part a placement decision can move; the other two are
            # what the site costs to have. See the note on _idle_energy_j.
            "total_idle_j": total_idle_j,
            "total_host_j": total_host_j,
            "time_at_bs": self.time_at_bs_profile(),
            "total_energy_kwh": total_energy_j / 3_600_000.0,
            "carbon_intensity": self.carbon_intensity,
        }

    def compute_energy(self, sim_start: float, sim_end: float) -> None:
        """Charge every node for this window."""
        for node in self.nodes:
            node.compute_energy(sim_start, sim_end)

    def time_at_bs_profile(self) -> Dict[int, float]:
        """Seconds this DC spent at each batch size, summed over its GPUs.

        The energy model is fitted over a measured envelope, and outside it the
        linear form extrapolates with no evidence behind it. Whether that
        matters depends on where the simulator actually operates, which this
        answers from the runs themselves instead of from a guess: if the time
        is concentrated well inside the envelope there is nothing to chase, and
        if it is not, this says exactly which corner needs measuring.
        """
        agg: Dict[int, float] = defaultdict(float)
        for node in self.nodes:
            for bs, secs in node.time_at_bs_profile().items():
                agg[int(bs)] += secs
        return dict(agg)

    def compute_energy_from_duration(self, sim_duration_s: float) -> None:
        """Compute energy using sim_duration_s (for gRPC path).

        Each DC process has its own monotonic clock, so we can't use
        absolute timestamps shared across processes.  Instead, we find the
        earliest GPU transition in *this* DC as sim_start and set
        sim_end = sim_start + sim_duration_s.

        DCs that received no requests get full idle power for the duration.
        """
        # Find the earliest transition across this DC, asked of the nodes
        earliest = None
        for node in self.nodes:
            t = node.earliest_transition()
            if t is not None and (earliest is None or t < earliest):
                earliest = t

        if earliest is not None:
            sim_start = earliest
            sim_end = earliest + sim_duration_s
        else:
            # No requests at all — pure idle for the full duration
            sim_start = time.monotonic()
            sim_end = sim_start + sim_duration_s

        self.compute_energy(sim_start, sim_end)

    def carbon_emissions_g(self) -> float:
        """Total CO2 emissions in grams: energy_j * PUE * ci / 3_600_000.

        Uses this datacentre's own PUE. Note the sensitivity: PUE multiplies
        every joule, so a site assumed at 1.1 that is really 1.5 under-reports
        by 36%, and the error does not cancel between sites when they differ.
        """
        energy_j = sum(node.energy_stats()["total_j"] for node in self.nodes)
        return energy_j * self.pue * self.carbon_intensity / 3_600_000.0
