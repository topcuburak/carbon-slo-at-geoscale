"""The batch ILP, re-solved as placements land so later requests see the earlier ones."""

import logging
from typing import Dict, List

import inference_pb2
from scheduler.policies import register_policy
from scheduler.policies.common.projection import project_placements
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
from scheduler.policies.common.ilp_base import ILPBase

log = logging.getLogger(__name__)


@register_policy("iterative_ilp_carbon_sla")
class IterativeILPCarbonSLA(ILPBase):
    """Iterative ILP with per-request SLA and load-coupled BS projection.

    Each iteration:
    1. Solve the ILP against the current (first round) or projected view
    2. Project the placements onto each datacentre, splitting them into the
       ones a free slot exists for and the ones that queue
    3. Recompute per-request TTFT/TPOT from the projected view
    4. Re-check SLA -- some (req, DC) pairs may become infeasible
    5. Re-solve until the assignment repeats or max_iterations rounds have run

    NOTE: repeating is not the same as converging. Measured on a 64-request
    batch, 8 rounds scores WORSE than 3 (4.06 vs 3.53 mgCO2): the assignment
    can cycle between two states rather than settle, and the loop then returns
    whichever it was on when the round budget ran out.

    Args:
        ttft_budget_ms: Max acceptable TTFT in ms.
        tpot_budget_ms: Max acceptable TPOT in ms.
        max_iterations: Max ILP solve rounds (default 3).
        analytical_model: AnalyticalLatencyModel for TPOT projection and SLA.
    """

    def __init__(self, scheduler_group: str = "europe_north",
                 ttft_budget_ms: float = 0, tpot_budget_ms: float = 0,
                 max_iterations: int = 3, analytical_model=None,
                 tp_size: int = 2, bs_cap: int = 0, **kwargs):
        super().__init__(scheduler_group=scheduler_group,
                         ttft_budget_ms=ttft_budget_ms,
                         tpot_budget_ms=tpot_budget_ms,
                         analytical_model=analytical_model,
                         tp_size=tp_size,
                         bs_cap=bs_cap, **kwargs)
        self.max_iterations = max_iterations

    def choose_dc_batch(
        self,
        requests: List[RequestInfo],
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        max_batch_size: int = 0,
        rtt_matrix: Dict[str, float] = None,
    ) -> Dict[str, str]:
        """Iterative ILP: solve, project load, re-check SLA, re-solve."""
        if not HAS_NUMPY or not HAS_SCIPY:
            log.warning("scipy/numpy not available — falling back to greedy")
            return self._fallback_greedy(requests, dc_view, rtt_matrix)

        # There used to be a "no analytical model -> fall back to the standard
        # ILP" branch here, and it could not have worked: this class inherits
        # from ILPBase, which has no choose_dc_batch, so super().choose_dc_batch
        # raised AttributeError. It was written when this policy inherited from
        # ILPCarbonSLA and was not updated when the two were made siblings --
        # the same refactor that dropped the numpy/scipy imports from all three.
        # Nothing reached it: scheduler/server.py refuses to start a policy that
        # estimates latency without a model, so the process exits before any
        # request arrives. Removed rather than repaired; neither sibling has
        # such a branch, and a broken degrade path is worse than none.

        dc_ids = sorted(dc_view.keys())
        n_req = len(requests)
        n_dc = len(dc_ids)

        if n_req == 0 or n_dc == 0:
            return {}

        # Precompute per-DC forwarding RTT (ms) once
        rtt_matrix = rtt_matrix or {}
        dc_fwd_ms = {
            dc_id: _lookup_rtt_ms(rtt_matrix, region_id, dc_id)
            for dc_id in dc_ids
        }

        prev_assignments = None

        for iteration in range(self.max_iterations):
            # Build projected DC stats
            if prev_assignments is None:
                projected_view = dc_view
            else:
                projected_view = self._project_load(
                    dc_view, dc_ids, requests, prev_assignments,
                )

            # Build cost and feasibility with per-request SLA
            cost = np.zeros((n_req, n_dc))
            feasible = np.ones((n_req, n_dc), dtype=bool)

            for j, dc_id in enumerate(dc_ids):
                stats = projected_view[dc_id]
                ci = _get_carbon_intensity(stats)
                if ci <= 0:
                    ci = 1.0
                fwd_ms = dc_fwd_ms[dc_id]

                for i, req in enumerate(requests):
                    sla_ok, ttft_ms, tpot_ms = _check_sla(
                        stats, req.in_tok,
                        self.ttft_budget_ms, self.tpot_budget_ms,
                        self.analytical_model,
                        # 0, as in ilp_carbon_sla and facility_ilp_carbon_sla.
                        # The default of 1 had this policy price every
                        # datacentre one request fuller than its two siblings
                        # did, for the same snapshot.
                        bs_offset=0,
                        fwd_ms=fwd_ms,
                        max_num_seqs=self.max_num_seqs,
                        # server.py sets predicted_osl from _osl_for(), so
                        # it is already what OSL_SOURCE resolved.
                        req_osl=int(getattr(req, "predicted_osl", 0) or 0),
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

            # Solve
            assignments = self._solve_lp(requests, dc_ids, projected_view,
                                         cost, feasible, n_req, n_dc,
                                         max_batch_size, rtt_matrix)

            if not assignments:
                log.warning("Iteration %d: ILP failed — using previous",
                            iteration + 1)
                break

            # Check convergence
            if prev_assignments == assignments:
                log.info("Iterative ILP converged at iteration %d", iteration + 1)
                break

            log.info("Iteration %d: assignments changed (%d requests), re-solving",
                     iteration + 1, n_req)
            prev_assignments = assignments

        if prev_assignments is None:
            return self._fallback_greedy(requests, dc_view, rtt_matrix)

        return prev_assignments

    def _project_load(
        self,
        original_view: Dict[str, inference_pb2.DcStats],
        dc_ids: List[str],
        requests: List[RequestInfo],
        assignments: Dict[str, str],
    ) -> Dict[str, inference_pb2.DcStats]:
        """The view the next round prices against: this round's placements
        folded into each datacentre's counters, admitted-vs-queued split so the
        disjoint pairs (active/pending, active_isl/pending_isl) stay disjoint.
        """
        # Count assignments per DC and collect ISLs
        assigned_per_dc: Dict[str, int] = {}
        assigned_isl_per_dc: Dict[str, List[int]] = {}
        assigned_osl_per_dc: Dict[str, List[int]] = {}
        for req in requests:
            dc_id = assignments.get(req.req_id)
            if dc_id:
                assigned_per_dc[dc_id] = assigned_per_dc.get(dc_id, 0) + 1
                if dc_id not in assigned_isl_per_dc:
                    assigned_isl_per_dc[dc_id] = []
                assigned_isl_per_dc[dc_id].append(req.in_tok)
                # predicted_osl is whatever OSL_SOURCE resolved upstream in
                # server._osl_for -- oracle, predicted or ref. It used to read
                # `predicted_osl or out_tok`, which silently reverted to the
                # TRUE length whenever the resolved value was 0, so a run
                # configured for `predicted` could be scored partly on oracle
                # knowledge without saying so.
                assigned_osl_per_dc.setdefault(dc_id, []).append(
                    max(0, int(getattr(req, 'predicted_osl', 0) or 0)))

        projected = {}
        for dc_id in dc_ids:
            # The admitted-vs-queued split and the disjointness invariants it
            # keeps live in project_placements, because the scheduler applies
            # the same correction to requests forwarded in from another group.
            # Two copies of this rule would drift.
            projected[dc_id] = project_placements(
                original_view[dc_id],
                assigned_isl_per_dc.get(dc_id, []),
                assigned_osl_per_dc.get(dc_id, []),
                self.max_num_seqs,
            )

        return projected

