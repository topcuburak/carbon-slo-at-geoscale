"""Batch placement as a capacitated facility-location MILP.

WHY THIS EXISTS, and what it fixes in ``ilp_carbon_sla``.

The linear programme prices each request independently: cost_ij is the carbon
of putting request i at datacentre j, and the objective is their sum. That is
exact only if the cost of two requests at one site is the cost of one plus the
cost of the other. It is not. A datacentre that is idle has to be woken: the
engine's per-step floor e0 is paid for every decode step, by whoever is there,
however many of them there are. Measured on this testbed at TP=2, one request
landing alone costs 4322 J; the same request landing where any other work is
already running costs 597 J. An 86% drop, and it is entirely the ``solo_steps``
term -- the steps this request is the only reason for.

So the true cost is CONCAVE in the number of requests sent to a site: the first
one is dear, the rest are cheap. A linear programme cannot represent that, and
the direction of its error is not random. Minimising a linear cost when the
truth is concave makes spreading look free -- the LP has no way to know that
the second request at a site rides for a tenth of the price of the first -- so
it scatters the batch across sites that each then pay their own wake-up. That
is the opposite of what the physics wants, which is to concentrate.

The standard form for "a fixed charge to open a site, then a per-unit cost" is
capacitated facility location:

    min   Σ_j f_j·y_j  +  Σ_ij c_ij·x_ij
    s.t.  Σ_j x_ij = 1                    every request goes somewhere
          Σ_i x_ij ≤ cap_j · y_j          nothing lands on a closed site
          y_j ∈ {0,1},  x_ij ∈ [0,1]

f_j is the wake-up carbon at site j and y_j is "site j is running this window".
Paying f_j once however many requests land there is exactly the economy of
scale the LP was missing.

WHAT THIS COSTS US, stated plainly. ``solver_choice_highs.md`` records why the
naive ILP needs no commercial solver: its constraint matrix is a transportation
problem, totally unimodular, so the LP relaxation is already integral -- we
measured 0 of 300 fractional variables. The binary y_j breaks that. This is a
real MILP with a branch-and-bound tree, and it is the first place in this work
where the no-Gurobi argument stops applying on its own terms. It is still
HiGHS, via ``scipy.optimize.milp``, but the justification is now empirical (it
solves fast enough, see SOLVE_TIME_LIMIT_S below) rather than structural.

One thing the relaxation still gives us: once y is fixed, what remains is the
transportation problem again, so the x's come out integral at the optimum
without being declared binary. Only the n_dc site variables enter the tree.
"""

import logging
import os
from typing import Dict, List

import inference_pb2
from scheduler.policies.common.capacity import dc_headroom
from scheduler.policies import register_policy
from scheduler.policies.common.ilp_base import ILPBase, HAS_NUMPY, HAS_SCIPY, np
from scheduler.policies.common.estimators import (
    RequestInfo,
    _check_sla,
    _get_carbon_intensity,
    _lookup_rtt_ms,
    _split_energy_j,
)

log = logging.getLogger(__name__)

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except ImportError:  # scipy < 1.9
    HAS_MILP = False

# Wall-clock ceiling for the branch-and-bound tree, then the LP relaxation.
# Measured on this topology (104 DCs, bs_cap 64, mixed trace): the tree closes
# inside 50 ms for batches up to 128, and needs between 50 and 250 ms at 256.
# 0.05 was the first guess and it was wrong in the expensive direction -- at
# 256 it timed out on all 6 seeds and the timeout PLUS the fallback solve came
# to 428 ms, against 353 ms for simply letting the tree finish. A limit that
# fires is not a saving.
SOLVE_TIME_LIMIT_S = float(os.environ.get("FACILITY_MILP_TIME_LIMIT_S", "0.25"))

# Which of the batch's output lengths stands in for "how long will the engine
# be up if we open this site". The wake-up charge is e0 per decode step, so it
# scales linearly with this number and the choice is not cosmetic -- across the
# scenario families OSL spans 66 to 707. "max" is the physically faithful one
# (the engine steps until the last request in the batch finishes) but overcharges
# a site that ends up with only short requests. Default is the median; the
# sensitivity is measured in FACILITY_MILP_NOTES.md.
OPEN_COST_OSL = os.environ.get("FACILITY_MILP_OSL", "median")


@register_policy("facility_ilp_carbon_sla")
class FacilityILPCarbonSLA(ILPBase):
    """Minimise carbon subject to the SLA, with an explicit wake-up charge."""

    def _representative(self, requests: List[RequestInfo]):
        """The (OSL, ISL) pair that stands in for the batch when pricing f_j."""
        osls = sorted(r.predicted_osl for r in requests)
        isls = sorted(r.in_tok for r in requests)
        if not osls:
            return 0, 0
        mid = len(osls) // 2
        if OPEN_COST_OSL == "max":
            return osls[-1], isls[-1]
        if OPEN_COST_OSL == "mean":
            return (int(sum(osls) / len(osls)), int(sum(isls) / len(isls)))
        return osls[mid], isls[mid]

    def choose_dc_batch(
        self,
        requests: List[RequestInfo],
        dc_view: Dict[str, inference_pb2.DcStats],
        region_id: str,
        max_batch_size: int = 0,
        rtt_matrix: Dict[str, float] = None,
    ) -> Dict[str, str]:
        if not HAS_NUMPY or not HAS_SCIPY or not HAS_MILP:
            log.warning("scipy.optimize.milp unavailable — falling back to greedy")
            return self._fallback_greedy(requests, dc_view, rtt_matrix)

        n_req = len(requests)
        dc_ids = sorted(dc_view.keys())
        n_dc = len(dc_ids)
        if n_req == 0 or n_dc == 0:
            return {}

        rtt_matrix = rtt_matrix or {}
        dc_fwd_ms = {dc_id: _lookup_rtt_ms(rtt_matrix, region_id, dc_id)
                     for dc_id in dc_ids}

        rep_osl, rep_isl = self._representative(requests)

        cost = np.zeros((n_req, n_dc))
        feasible = np.ones((n_req, n_dc), dtype=bool)
        open_cost = np.zeros(n_dc)

        for j, dc_id in enumerate(dc_ids):
            stats = dc_view[dc_id]
            ci = _get_carbon_intensity(stats)
            if ci <= 0:
                ci = 1.0
            fwd_ms = dc_fwd_ms[dc_id]
            dc_bs = int(stats.active_requests / max(stats.num_gpus, 1))

            # The wake-up charge, from the SAME function and the SAME snapshot
            # that prices the requests. Deliberately not a separate "is this DC
            # idle" predicate: two predicates drift apart at the boundary, and
            # the marginal/fixed split would then either double-charge or lose
            # the term. Here the split is exact by construction.
            _, f_j = _split_energy_j(rep_osl, 0.0, dc_bs,
                                     self.analytical_model,
                                     req_isl=rep_isl, tp_size=self.tp_size,
                                     stats=stats)
            open_cost[j] = f_j * ci

            for i, req in enumerate(requests):
                sla_ok, ttft_ms, tpot_ms = _check_sla(
                    stats, req.in_tok,
                    self.ttft_budget_ms, self.tpot_budget_ms,
                    self.analytical_model,
                    bs_offset=0,
                    fwd_ms=fwd_ms,
                    max_num_seqs=self.max_num_seqs,
                    # server.py sets predicted_osl from _osl_for(), i.e. it is
                    # already what OSL_SOURCE resolved -- the priced length.
                    req_osl=int(getattr(req, "predicted_osl", 0) or 0),
                )
                if not sla_ok:
                    feasible[i][j] = False
                    cost[i][j] = 1e12
                else:
                    # MARGINAL ONLY. The keepalive half of this same call is
                    # already in open_cost[j]; leaving it here too would charge
                    # the wake-up once per request, which is the bug the whole
                    # formulation exists to avoid.
                    marg, _ = _split_energy_j(
                        req.predicted_osl, tpot_ms, dc_bs,
                        self.analytical_model, req_isl=req.in_tok,
                        tp_size=self.tp_size, stats=stats)
                    cost[i][j] = marg * ci

        return self._solve_milp(requests, dc_ids, dc_view, cost, feasible,
                                open_cost, n_req, n_dc, max_batch_size,
                                rtt_matrix)

    def _solve_milp(self, requests, dc_ids, dc_view, cost, feasible,
                    open_cost, n_req, n_dc, max_batch_size, rtt_matrix=None):
        """Variables: [x_00..x_{n-1,m-1}, y_0..y_{m-1}]."""
        n_x = n_req * n_dc
        n_var = n_x + n_dc

        # Σ_j x_ij <= 1, not = 1, with a reward for placing. The equality made
        # the programme infeasible whenever the batch exceeded the fleet's free
        # headroom -- it cannot leave one request out, so it failed entirely and
        # the whole batch fell through the ladder, burning the MILP solve and
        # its time limit on the way. See the note in ilp_base._solve_lp; this is
        # the same relaxation, applied here so the MILP handles overflow itself.
        finite = cost[cost < 1e11]
        reward = (float(finite.max()) + 1.0) if finite.size else 1.0
        c = np.concatenate([(cost - reward).flatten(), open_cost])

        A_assign = np.zeros((n_req, n_var))
        for i in range(n_req):
            A_assign[i, i * n_dc:(i + 1) * n_dc] = 1.0
        cons = [LinearConstraint(A_assign, lb=-np.inf, ub=1.0)]

        # Σ_i x_ij − cap_j·y_j ≤ 0. Same effective cap and same
        # (active + pending) load baseline as the LP, so the two policies are
        # compared under identical capacity, not just identical costs.
        # PER ENGINE, like the LP's; dc_headroom folds in each site's engine
        # count. See policies/common/capacity.py.
        cap_per_engine = 0
        if max_batch_size > 0 and self.bs_cap > 0:
            cap_per_engine = min(max_batch_size, self.bs_cap)
        elif max_batch_size > 0:
            cap_per_engine = max_batch_size
        elif self.bs_cap > 0:
            cap_per_engine = self.bs_cap

        A_link = np.zeros((n_dc, n_var))
        for j in range(n_dc):
            for i in range(n_req):
                A_link[j, i * n_dc + j] = 1.0
            free = dc_headroom(dc_view[dc_ids[j]], cap_per_engine)
            # No guard: a site may take the whole batch, which is what the
            # binary open/closed variable is there to decide.
            cap_j = float(n_req) if free is None else float(free)
            A_link[j, n_x + j] = -cap_j
        cons.append(LinearConstraint(A_link, lb=-np.inf, ub=0.0))

        lb = np.zeros(n_var)
        ub = np.ones(n_var)
        for i in range(n_req):
            for j in range(n_dc):
                if not feasible[i][j]:
                    ub[i * n_dc + j] = 0.0
        integrality = np.zeros(n_var)
        integrality[n_x:] = 1  # only the site variables enter the tree

        bounds = Bounds(lb, ub)
        for relax in (False, True):
            try:
                res = milp(c=c, constraints=cons,
                           integrality=(np.zeros(n_var) if relax else integrality),
                           bounds=bounds,
                           options=(None if relax
                                    else {"time_limit": SOLVE_TIME_LIMIT_S}))
                if res.success and res.x is not None:
                    x = np.asarray(res.x[:n_x]).reshape(n_req, n_dc)
                    out, unplaced = {}, []
                    for i, req in enumerate(requests):
                        j = int(np.argmax(x[i]))
                        if x[i][j] > 0.5:
                            out[req.req_id] = dc_ids[j]
                        else:
                            unplaced.append(req)
                    if unplaced:
                        seated = {}
                        for d in out.values():
                            seated[d] = seated.get(d, 0) + 1
                        fb_view = {}
                        for d, st in dc_view.items():
                            if seated.get(d):
                                proj = inference_pb2.DcStats()
                                proj.CopyFrom(st)
                                proj.active_requests = st.active_requests + seated[d]
                                fb_view[d] = proj
                            else:
                                fb_view[d] = st
                        log.info("facility seated %d/%d; %d over capacity",
                                 len(out), n_req, len(unplaced))
                        out.update(self._fallback_greedy(unplaced, fb_view,
                                                         rtt_matrix))
                    return out
                if not relax:
                    # Measured: on a time limit HiGHS returns res.x = None, no
                    # incumbent, so there is nothing to salvage from the tree.
                    log.warning("facility MILP timed out (%s) — LP relaxation",
                                getattr(res, 'message', '?'))
            except Exception as e:
                log.warning("facility MILP error (relax=%s): %s", relax, e)

        # Last rung. Note this drops the fixed charge ENTIRELY, which is the
        # marginal-only LP -- measured 39% worse than the MILP when carbon
        # intensities are close together, because charging nothing to wake a
        # site makes scattering look free. The relaxation above keeps the term
        # (scaled by the fraction of capacity used) and is preferred; this runs
        # only if that also fails.
        return self._solve_lp(requests, dc_ids, dc_view, cost, feasible,
                              n_req, n_dc, max_batch_size, rtt_matrix)
