"""Carbon-aware scheduling policy with TTFT SLA enforcement.

Scores each datacenter with a weighted combination of normalised
latency and normalised carbon intensity::

    score = alpha * norm_latency + (1 - alpha) * norm_carbon

- ``alpha = 1.0`` → pure latency optimisation (ignores carbon).
- ``alpha = 0.0`` → pure carbon optimisation (ignores latency).

TTFT SLA is enforced as a hard pre-filter before scoring:
    estimated_TTFT = queue_wait + pure_prefill(my_ISL)
    where queue_wait = sum of prefill times for pending requests
    DC is feasible only if estimated_TTFT < ttft_budget_ms

Latency estimation uses the AnalyticalLatencyModel to compute
per-request TTFT from ISL + TPOT from (BS, avg_ISL).

For remote (cross-group) DCs the latency estimate adds ONE leg of the
inter-group RTT from the provisioner's matrix, which is itself one-way.

That is a deliberate choice and it does not match what the run then costs.
The server sleeps the RTT twice on a forward, out and back, and records
forwarding_ms = 2 * rtt_ms. The estimate counts the outbound leg only: what
delays the request reaching the engine. The return leg carries the first token
back and is not charged against the decision.

The consequence is worth stating rather than discovering. For a remote
datacentre the TTFT filter is optimistic by one RTT leg, so it can admit a
placement whose measured TTFT exceeds the budget it was checked against. It
biases toward forwarding, and the bias grows with distance.
"""

import logging
from typing import Dict, Optional, Tuple

import inference_pb2
from common.network import REGION_TO_GROUP, rtt_pair_key
from scheduler.policies.common.capacity import at_cap, greenest_under_cap
from scheduler.policies.common.carbon_signal import effective_carbon_intensity
from scheduler.policies import register_policy
from scheduler.policies.common.base import BasePolicy, PolicyScope
from scheduler.policies.common.estimators import RoundRobinFallback
from scheduler.policies.common.latency_estimates import (
    estimate_ttft_ms, estimate_tpot_ms)

log = logging.getLogger(__name__)


@register_policy("carbon_aware")
class CarbonAware(BasePolicy):
    """Weighted score: alpha * norm_latency + (1-alpha) * norm_carbon.

    TTFT SLA is enforced as a hard constraint — DCs where the estimated
    TTFT (queue wait + prefill) exceeds the budget are excluded.

    Args:
        alpha: Latency vs. carbon weight (0.0–1.0).
        scheduler_group: Geographic group this scheduler belongs to.
        analytical_model: AnalyticalLatencyModel instance (required).
        ref_osl: Reference output sequence length for latency scoring.
        ttft_budget_ms: Hard TTFT SLA budget in ms (0 = no constraint).
        tpot_budget_ms: Hard TPOT SLA budget in ms (0 = no constraint).
            Uses projected BS (current + 1) for estimation.
    """

    def __init__(self, alpha: float = 0.5,
                 analytical_model=None, ref_osl: int = 500,
                 ttft_budget_ms: float = 0, tpot_budget_ms: float = 0,
                 bs_cap: int = 0, **kwargs):
        # Popped before super, not after: BasePolicy takes **kwargs and would
        # otherwise swallow it, leaving a pop that reads as if it removed
        # something.
        self.max_num_seqs = int(kwargs.pop('max_num_seqs', 0) or 0)
        super().__init__(**kwargs)
        # Said out loud because the direction is a trap: alpha is the LATENCY
        # weight, so a higher value is LESS carbon-aware. A sweep of
        # 0.5/0.75/0.9/0.95 reads like a carbon sweep and is the opposite half
        # of the range; measured on 2026-09-02, every value in it placed like
        # round_robin. Print it, and a wrong direction shows up in the log
        # rather than in the conclusions.
        self.alpha = alpha
        log.info("carbon_aware: alpha=%.2f -> %.0f%% latency / %.0f%% carbon "
                 "(alpha is the LATENCY weight; 0.0 is carbon-only)",
                 alpha, alpha * 100, (1 - alpha) * 100)
        self.analytical_model = analytical_model
        self.ref_osl = ref_osl
        self.ttft_budget_ms = ttft_budget_ms
        self.tpot_budget_ms = tpot_budget_ms
        # Hard per-DC cap on (active + pending + this request).
        # 0 disables. Excludes DCs whose projected load would exceed bs_cap.
        self.bs_cap = bs_cap
        # Last resort when the SLA excludes everything. See RoundRobinFallback
        # for why this is not least-loaded.
        #
        # One counter per path, not one shared. pick() indexes ids[_i % len(ids)]
        # and the two paths pass lists of different length -- every datacentre
        # for a fresh request, this group's ~15 for a forwarded one -- so a
        # shared counter has each rotation stepping the other's index and
        # neither completing a clean cycle.
        self._rr_fallback = RoundRobinFallback()          # global path
        self._rr_fallback_local = RoundRobinFallback()    # forwarded path

    @property
    def scope(self) -> PolicyScope:
        return PolicyScope.GLOBAL_WITH_LOCAL

    def _estimate_ttft(self, stats, req_isl: int, fwd_ms: float = 0.0) -> float:
        return estimate_ttft_ms(stats, req_isl, self.analytical_model,
                                fwd_ms=fwd_ms,
                                max_num_seqs=self.max_num_seqs)

    def _estimate_tpot(self, stats, req_osl: int = 0,
                       req_isl: int = 0) -> float:
        return estimate_tpot_ms(stats, self.analytical_model, bs_offset=1,
                                req_osl=req_osl, req_isl=req_isl)

    def _estimate_latency(self, stats, req_isl: int = 0,
                          fwd_ms: float = 0.0, req_osl: int = 0) -> float:
        """Total user-perceived latency for scoring: TTFT + TPOT x OSL.

        req_osl is the length the scheduler decided to price this placement
        on -- see latency_estimates.resolve_osl. It falls back to ref_osl, a
        flat constant for every request, which is what this used
        unconditionally: on a trace whose output lengths run from 1 to 16384
        that made every candidate's decode term identical, so the score was
        effectively TTFT alone plus a constant.
        """
        ttft_ms = self._estimate_ttft(stats, req_isl if req_isl > 0 else 5000,
                                       fwd_ms=fwd_ms)
        tpot_ms = self._estimate_tpot(stats, req_osl=req_osl)
        return ttft_ms + tpot_ms * (req_osl if req_osl > 0 else self.ref_osl)

    def _check_ttft_sla(self, stats, req_isl: int,
                        fwd_ms: float = 0.0) -> bool:
        """Check if TTFT SLA is met at this DC for this request.

        Compared against the FLAT ``ttft_budget_ms``; see the note below.

        The estimated TTFT includes the forwarding round-trip (``fwd_ms``)
        so cross-region DCs that would blow the budget purely on network
        latency are correctly rejected.
        """
        if self.ttft_budget_ms <= 0:
            return True
        # TTFT_BUDGET is a FLAT ceiling in ms, compared against the estimate
        # directly. It used to be an additive slack over 1.5 x prefill_ms(ISL),
        # which meant the budget grew with the prompt -- 1530 ms at ISL 128 and
        # 3104 ms at ISL 20000 -- so a long prompt could never breach it and the
        # constraint bound nothing: measured 0 of 6656 (request, DC) pairs
        # infeasible at the loads the sweep runs.
        return (self._estimate_ttft(stats, req_isl, fwd_ms=fwd_ms)
                <= self.ttft_budget_ms)

    def _check_tpot_sla(self, stats, req_osl: int = 0,
                        req_isl: int = 0) -> bool:
        """Is the TPOT budget met at this DC if this request goes there?

        req_isl is what TPOT_FILTER=batch charges the running batch for: this
        request's prompt has to go through a forward pass before it decodes,
        and the batch already there stops for the whole of it.
        """
        if self.tpot_budget_ms <= 0:
            return True
        return (self._estimate_tpot(stats, req_osl=req_osl, req_isl=req_isl)
                <= self.tpot_budget_ms)


    def choose_dc(
        self,
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        **kwargs,
    ) -> Optional[str]:
        result = self.choose_dc_extended(dc_view, region_id, **kwargs)
        return result[0] if result else None

    def choose_dc_extended(
        self,
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        **kwargs,
    ) -> Optional[Tuple[str, bool]]:
        """Score all DCs globally and return (best_dc_id, is_local).

        DCs that violate TTFT SLA are excluded before scoring.
        """
        rtt_matrix = kwargs.get("rtt_matrix")
        req_isl = kwargs.get("req_isl", 0)
        req_osl = kwargs.get("req_osl", 0)
        if not dc_view:
            return None

        candidates = []
        for dc_id, stats in dc_view.items():
            dc_group = REGION_TO_GROUP.get(dc_id)
            if dc_group is None:
                continue

            # Forwarding round-trip for this DC (0 for same-region)
            is_local = dc_id in self._local_regions
            if is_local:
                fwd_ms = 0.0
            elif rtt_matrix:
                key = rtt_pair_key(self.scheduler_group, dc_group)
                fwd_ms = float(rtt_matrix.get(key, 0.0))
            else:
                fwd_ms = 0.0

            # Hard per-DC BS cap: exclude DCs already at (active + pending)
            # ≥ bs_cap so new assignments can't pile onto an overloaded DC.
            # PER ENGINE; see policies/common/capacity.py for why.
            if at_cap(stats, self.bs_cap):
                continue

            # Hard SLA checks — TTFT now includes forwarding, TPOT unchanged
            if req_isl > 0 and not self._check_ttft_sla(stats, req_isl, fwd_ms=fwd_ms):
                continue
            if not self._check_tpot_sla(stats, req_osl=req_osl,
                                        req_isl=req_isl):
                continue

            # Latency score for the α-weighted rank (already includes fwd_ms
            # via _estimate_latency → _estimate_ttft)
            latency_est = self._estimate_latency(stats, req_isl, fwd_ms=fwd_ms,
                                                 req_osl=req_osl)
            ci = effective_carbon_intensity(stats)
            candidates.append((dc_id, is_local, latency_est, ci))

        if not candidates:
            # Nothing met the SLA. Drop the estimate, keep the count: the
            # greenest site still under bs_cap. Only if every site is over it
            # does this fall through to round-robin, which is carbon-blind and
            # was measured to cost the whole of the carbon lost down the cap
            # axis. See capacity.greenest_under_cap.
            dc_id = greenest_under_cap(dc_view, self.bs_cap)
            if dc_id is None:
                dc_id = self._rr_fallback.pick(dc_view)
            if dc_id is None:
                return None
            return (dc_id, dc_id in self._local_regions)

        # Separate finite and infinite latency candidates
        finite = [(dc_id, is_local, lat, ci) for dc_id, is_local, lat, ci in candidates
                  if lat != float("inf")]

        if finite:
            max_lat = max(c[2] for c in finite) or 1.0
            max_ci = max(c[3] for c in finite) or 1.0
            scoring_set = finite
        else:
            max_lat = 1.0
            max_ci = max(c[3] for c in candidates) or 1.0
            scoring_set = [(dc_id, is_local, 0.0, ci) for dc_id, is_local, _, ci in candidates]

        best_score = float("inf")
        best = None
        for dc_id, is_local, lat, ci in scoring_set:
            score = self.alpha * (lat / max_lat) + (1 - self.alpha) * (ci / max_ci)
            if score < best_score:
                best_score = score
                best = (dc_id, is_local)

        return best

    def choose_dc_local_only(
        self,
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        **kwargs,
    ) -> Optional[str]:
        """Pick the best DC from this scheduler's local group only."""
        req_isl = kwargs.get("req_isl", 0) if kwargs else 0
        req_osl = kwargs.get("req_osl", 0) if kwargs else 0
        local_view = {
            dc_id: stats for dc_id, stats in dc_view.items()
            if dc_id in self._local_regions
        }
        if not local_view:
            return None

        candidates = []
        for dc_id, stats in local_view.items():
            # The same three filters as the global path. They used to differ:
            # this path checked TTFT only, so bs_cap and the TPOT budget were
            # applied to a fresh request and waived for a forwarded one. That is
            # backwards -- a forwarded request has already spent a region hop,
            # so its remaining budget is tighter, not looser -- and it meant the
            # same datacentre could be rejected for one request and accepted for
            # another in the same instant.
            #
            # No fwd_ms here: the hop is already paid and cannot be paid again,
            # since a forwarded request is never forwarded on.
            # PER ENGINE; see policies/common/capacity.py for why.
            if at_cap(stats, self.bs_cap):
                continue
            if req_isl > 0 and not self._check_ttft_sla(stats, req_isl):
                continue
            if not self._check_tpot_sla(stats, req_osl=req_osl,
                                        req_isl=req_isl):
                continue

            latency_est = self._estimate_latency(stats, req_isl,
                                                 req_osl=req_osl)
            candidates.append((dc_id, latency_est,
                               effective_carbon_intensity(stats)))

        if not candidates:
            # Same ladder as the global path: greenest still under the load
            # guard, and round-robin only when every one of them is over it.
            return (greenest_under_cap(local_view, self.bs_cap)
                    or self._rr_fallback_local.pick(local_view))

        finite = [(dc_id, lat, ci) for dc_id, lat, ci in candidates
                  if lat != float("inf")]

        if finite:
            max_lat = max(c[1] for c in finite) or 1.0
            max_ci = max(c[2] for c in finite) or 1.0
            scoring_set = finite
        else:
            max_lat = 1.0
            max_ci = max(c[2] for c in candidates) or 1.0
            scoring_set = [(dc_id, 0.0, ci) for dc_id, _, ci in candidates]

        best_score = float("inf")
        best_dc = None
        for dc_id, lat, ci in scoring_set:
            score = self.alpha * (lat / max_lat) + (1 - self.alpha) * (ci / max_ci)
            if score < best_score:
                best_score = score
                best_dc = dc_id

        return best_dc
