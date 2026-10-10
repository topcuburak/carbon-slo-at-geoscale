"""Batch placement as an integer linear program over carbon, subject to the SLA."""

import logging
from typing import Dict, List

import inference_pb2
from scheduler.policies import register_policy
from scheduler.policies.common.ilp_base import ILPBase
from scheduler.policies.common.estimators import (
    HAS_NUMPY,
    HAS_SCIPY,
    RequestInfo,
    _check_sla,
    _estimate_energy_j,
    _get_carbon_intensity,
    _lookup_rtt_ms,
    np,
)

log = logging.getLogger(__name__)



@register_policy("ilp_carbon_sla")
class ILPCarbonSLA(ILPBase):
    """Solve the whole batch once: minimise carbon subject to the SLA."""

    def choose_dc_batch(
        self,
        requests: List[RequestInfo],
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        max_batch_size: int = 0,
        rtt_matrix: Dict[str, float] = None,
    ) -> Dict[str, str]:
        """Solve ILP for a batch of requests with per-request SLA.

        For each (request_i, DC_j):
          - TTFT_ij = ISL-aware analytical TTFT (prefill + queue drain + RTT/2)
          - TPOT_j  = analytical_model.tpot_ms(DC_j.BS, DC_j.total_KV)
          - feasible_ij = (TTFT_ij < budget) AND (TPOT_j < budget)
          - cost_ij = energy(OSL_i, TPOT_j) × carbon_intensity_j
        """
        if not HAS_NUMPY or not HAS_SCIPY:
            log.warning("scipy/numpy not available — falling back to greedy")
            return self._fallback_greedy(requests, dc_view, rtt_matrix)

        n_req = len(requests)
        dc_ids = sorted(dc_view.keys())
        n_dc = len(dc_ids)

        if n_req == 0 or n_dc == 0:
            return {}

        # Precompute per-DC forwarding RTT (ms) once per batch
        rtt_matrix = rtt_matrix or {}
        dc_fwd_ms = {
            dc_id: _lookup_rtt_ms(rtt_matrix, region_id, dc_id)
            for dc_id in dc_ids
        }

        cost = np.zeros((n_req, n_dc))
        feasible = np.ones((n_req, n_dc), dtype=bool)

        for j, dc_id in enumerate(dc_ids):
            stats = dc_view[dc_id]
            ci = _get_carbon_intensity(stats)
            if ci <= 0:
                ci = 1.0
            fwd_ms = dc_fwd_ms[dc_id]

            for i, req in enumerate(requests):
                # Per-request SLA check (queue-wait TTFT + projected TPOT)
                sla_ok, ttft_ms, tpot_ms = _check_sla(
                    stats, req.in_tok,
                    self.ttft_budget_ms, self.tpot_budget_ms,
                    self.analytical_model,
                    bs_offset=0,  # ILP: use snapshot/projected BS directly
                    fwd_ms=fwd_ms,
                    max_num_seqs=self.max_num_seqs,
                    # server.py sets predicted_osl from _osl_for(), i.e. it is
                    # already what OSL_SOURCE resolved -- the priced length.
                    req_osl=int(getattr(req, "predicted_osl", 0) or 0)
                )

                if not sla_ok:
                    feasible[i][j] = False
                    cost[i][j] = 1e12
                else:
                    dc_bs = int(stats.active_requests / max(stats.num_gpus, 1))
                    energy = _estimate_energy_j(req.predicted_osl, tpot_ms, dc_bs,
                                                self.analytical_model,
                                                req_isl=req.in_tok,
                                                tp_size=self.tp_size,
                                    stats=stats)
                    cost[i][j] = energy * ci

        return self._solve_lp(requests, dc_ids, dc_view, cost, feasible,
                              n_req, n_dc, max_batch_size, rtt_matrix)

