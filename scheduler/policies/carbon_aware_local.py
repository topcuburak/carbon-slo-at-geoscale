"""Carbon-aware scheduling policy (local group only) with TTFT + TPOT SLA enforcement.

Same scoring as ``CarbonAware``::

    score = alpha * norm_latency + (1 - alpha) * norm_carbon

but only considers datacenters within the scheduler's own geographic
group. Requests are never forwarded to remote schedulers.

SLA is enforced as a hard pre-filter:
    TTFT: queue_wait + pure_prefill(my_ISL) < ttft_budget_ms
    TPOT: tpot(projected_BS, avg_ISL) < tpot_budget_ms
"""

from typing import Dict, Optional

import inference_pb2
from scheduler.policies.common.capacity import at_cap, greenest_under_cap
from scheduler.policies.common.carbon_signal import effective_carbon_intensity
from scheduler.policies import register_policy
from scheduler.policies.common.base import BasePolicy, PolicyScope
from scheduler.policies.common.latency_estimates import (
    estimate_ttft_ms, estimate_tpot_ms)


@register_policy("carbon_aware_local")
class CarbonAwareLocal(BasePolicy):
    """Local-only carbon-aware with TTFT + TPOT SLA enforcement.

    Args:
        alpha: Latency vs. carbon weight (0.0–1.0).
        scheduler_group: Geographic group this scheduler belongs to.
        analytical_model: AnalyticalLatencyModel instance (required).
        ref_osl: Reference output sequence length for latency scoring.
        ttft_budget_ms: Hard TTFT SLA budget in ms (0 = no constraint).
        tpot_budget_ms: Hard TPOT SLA budget in ms (0 = no constraint).
            Uses projected BS (current + 1) for estimation.
    """

    def __init__(self, alpha: float = 0.5, analytical_model=None,
                 ref_osl: int = 500, ttft_budget_ms: float = 0,
                 tpot_budget_ms: float = 0, bs_cap: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.analytical_model = analytical_model
        self.ref_osl = ref_osl
        self.ttft_budget_ms = ttft_budget_ms
        self.tpot_budget_ms = tpot_budget_ms
        # Hard per-DC cap on (active + pending). 0 disables.
        self.bs_cap = bs_cap
        # The engines' max_num_seqs; gates the admission-wait term. 0 = ungated.
        self.max_num_seqs = int(kwargs.pop('max_num_seqs', 0) or 0)

    @property
    def scope(self) -> PolicyScope:
        return PolicyScope.LOCAL_ONLY

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
        """TTFT SLA check against the FLAT ``ttft_budget_ms``; see the note below."""
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
        req_isl = kwargs.get("req_isl", 0)
        req_osl = kwargs.get("req_osl", 0)
        local_view = {
            dc_id: stats for dc_id, stats in dc_view.items()
            if dc_id in self._local_regions
        }
        if not local_view:
            return None

        # LocalOnly policy: by definition all candidates are in-region, so
        # fwd_ms=0 for every candidate. No RTT lookup needed.
        candidates = []
        for dc_id, stats in local_view.items():
            # Hard per-DC BS cap
            # PER ENGINE; see policies/common/capacity.py for why.
            if at_cap(stats, self.bs_cap):
                continue

            # Hard SLA checks (fwd_ms=0 for local DCs)
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
            # Nothing met the SLA. Greenest still under the load guard first --
            # least_loaded was the whole fallback here, and for a carbon policy
            # its loaded sites ARE its green ones, so it reliably picked the
            # carbon-expensive one. It stays as the last rung, for when every
            # site is over the guard and something must still be chosen.
            green = greenest_under_cap(local_view, self.bs_cap)
            if green is not None:
                return green
            best_load, best_dc = float("inf"), None
            for dc_id, stats in local_view.items():
                if stats.active_requests < best_load:
                    best_load, best_dc = stats.active_requests, dc_id
            return best_dc

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
