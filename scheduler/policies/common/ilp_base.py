"""The machinery both ILP policies run on.

ilp_carbon_sla and iterative_ilp_carbon_sla are two placement STRATEGIES over
one solver: the first solves the batch once, the second re-solves it as
placements land. The solver, the greedy fallback when it cannot, and the
single-request path a batch policy still has to answer on are the same in both.

They used to be siblings only in name -- the iterative one inherited from the
one-shot one, so reading it did not tell you what it did and changing the
one-shot policy changed the iterative one silently. Now both inherit from here
and neither from the other, which is what lets each file be read on its own.
"""

import logging

import inference_pb2
from scheduler.policies.common.capacity import at_cap, dc_headroom
from scheduler.policies.common.base import BasePolicy, PolicyScope
from scheduler.policies.common.estimators import (
    HAS_NUMPY,
    HAS_SCIPY,
    RoundRobinFallback,
    csr_matrix,
    _check_sla,
    _get_carbon_intensity,
    _lookup_rtt_ms,
    linprog,
    np,
)

log = logging.getLogger(__name__)


# This constraint counts CONCURRENT requests, which is what max_num_seqs
# limits: the engine holds 64 sequences whether they emit 60 tokens or 600.
#
# KNOWN LIMITATION, not a bug. A long-decode request holds its seat far longer,
# and over a single window the LP cannot see that -- it allocates the scarce
# clean-datacentre seats by total energy A_i, when the benefit per unit of
# OCCUPANCY is A_i/osl_i. Those orderings invert at the top: (9000 ISL, 60 OSL)
# returns 5.88 J per slot-step against 0.87 for (9000 ISL, 600 OSL), yet ranks
# third by A_i. Heavy prefill with short decode is the best tenant a green site
# can have -- its energy is paid up front in one step and the seat is released
# quickly.
#
# Weighting this row by osl_i/R_j was implemented on 2026-09-01 and REVERTED
# TWICE. Do not try it again without reading this:
#
#   1. It does not address the case it was written for. A prefill-heavy request
#      already wins the scarce green seat, because cost = A_i x CI and prefill
#      energy scales with ISL: at ISL 5000 vs 500, A_i is 216.7 J against 34.8.
#      The weighting is inert whenever the OSLs are equal, which is exactly the
#      comparison that motivated it.
#   2. It breaks integrality. With unit weights the matrix is a transportation
#      problem, totally unimodular, and the LP relaxation is integral -- which
#      is the recorded justification for using HiGHS over a commercial MIP
#      solver. A weight that depends on (i, j) makes it a generalized assignment
#      problem, NP-hard, and the relaxation is no longer integral. Measured: two
#      requests against one seat of headroom returned x = 0.5/0.5, and argmax
#      then rounded it into an assignment consuming 1.33 seats of a 1.0 seat
#      site. A wide 104-datacentre snapshot showed 0 fractional variables and
#      gave false confidence -- capacity has to BIND before the defect appears.
#   3. It lost on the only metric a single window can score, 81.71 -> 81.93
#      mgCO2, while the benefit it was meant to buy is invisible to a
#      one-window evaluation.
#
# The real fix is a multi-window formulation, not a distorted capacity row.


class ILPBase(BasePolicy):
    """Solver, fallback and single-request path. Not registered: not a policy."""

    def __init__(self, scheduler_group: str = "europe_north",
                 ttft_budget_ms: float = 0, tpot_budget_ms: float = 0,
                 analytical_model=None, tp_size: int = 2,
                 bs_cap: int = 0, **kwargs):
        super().__init__(scheduler_group=scheduler_group, **kwargs)
        self.ttft_budget_ms = ttft_budget_ms
        self.tpot_budget_ms = tpot_budget_ms
        self.analytical_model = analytical_model
        self.tp_size = tp_size
        # Hard cap on a datacentre's TOTAL in-flight work: active + pending
        # summed over ALL its engines, plus whatever this solve adds. 0 disables.
        #
        # PER DATACENTRE, NOT PER ENGINE. The name reads like a batch-size cap
        # and invites being set to max_num_seqs, which throttles a site to 1/N
        # of what it can hold, N being its engine count. Measured 2026-09-02 on
        # 8 sites of 8 engines at max_num_seqs=128 -- real capacity 1024 each --
        # with bs_cap=128: every site sat at a median 217-327 in flight, so
        # `effective_cap - current_load` was 0 everywhere, the LP could seat
        # nothing, every request fell through to the round-robin fallback, and
        # ilp_carbon_sla placed 52% of requests on the clean sites against
        # round_robin's 50%. The carbon objective never ran.
        #
        # It binds only as a datacentre's own load approaches it, so a sweep
        # over 104 sites at 30-50 req/s (~4 in flight per site) never notices.
        self.bs_cap = bs_cap
        self.max_num_seqs = int(kwargs.pop('max_num_seqs', 0) or 0)
        # Last rung of the fallback ladder; see _fallback_greedy.
        self._rr_fallback = RoundRobinFallback()

    @property
    def scope(self) -> PolicyScope:
        return PolicyScope.GLOBAL

    def choose_dc(self, dc_view, region_id, **kwargs):
        result = self.choose_dc_extended(dc_view, region_id, **kwargs)
        return result[0] if result else None

    def choose_dc_extended(self, dc_view, region_id, **kwargs):
        # Single-request fallback — greedy carbon with SLA.
        #
        # This used to accept rtt_matrix and never read it, so every candidate
        # was priced as though it were in this region: the SLA check charged no
        # forwarding delay at all. On this topology the widest one-way hop is
        # asia<->south_america at 280 ms, which is 18% of the ISL-128 budget and
        # 13% of the ISL-8192 one, always in the optimistic direction. The batch
        # path had charged it since it was written; this one had not.
        req_isl = kwargs.get('req_isl', 0)
        req_osl = kwargs.get('req_osl', 0)
        rtt_matrix = kwargs.get('rtt_matrix') or {}
        best_dc = None
        best_ci = float("inf")
        for dc_id, stats in dc_view.items():
            # Skip datacentres with no seat left. Without this the ladder never
            # reached its round-robin rung on overflow: every request the batch
            # solve could not seat was handed the single lowest-carbon site,
            # capacity or not, so 60 overflowing requests all went to one full
            # datacentre. Carbon-optimal among sites that can actually take it,
            # then round-robin -- which is the ladder as specified.
            # PER ENGINE; see policies/common/capacity.py for why.
            if at_cap(stats, self.bs_cap):
                continue
            if req_isl > 0:
                sla_ok, _, _ = _check_sla(
                    stats, req_isl,
                    self.ttft_budget_ms, self.tpot_budget_ms,
                    self.analytical_model,
                    # 0, as in choose_dc_batch. The default of 1 made the two
                    # paths of one policy disagree about the same datacentre.
                    bs_offset=0,
                    fwd_ms=_lookup_rtt_ms(rtt_matrix, region_id, dc_id),
                    req_osl=req_osl,
                    max_num_seqs=self.max_num_seqs
                )
            else:
                sla_ok = True
            if not sla_ok:
                continue
            ci = _get_carbon_intensity(stats)
            if ci <= 0:
                ci = 1.0
            if ci < best_ci:
                best_ci = ci
                best_dc = dc_id
        if best_dc is None:
            # No datacentre meets the SLA. This used to pick the fewest
            # active_requests, which is the one thing a carbon policy must not
            # do here: it has been sending work to the low-carbon sites, so
            # those are the loaded ones, and "least loaded" then selects with
            # fair reliability the site no policy wanted -- the carbon-expensive
            # one. It converted an SLA failure into a carbon failure, in exactly
            # the overloaded regime where both matter most. It was also weaker
            # than the least_loaded policy it imitated: no `pending` in the
            # load, no bs_cap.
            #
            # It also made the round-robin rung below _fallback_greedy dead
            # code, because this branch never returned None, so the documented
            # ladder never reached its last step.
            best_dc = self._rr_fallback.pick(dc_view)
        if best_dc is None:
            return None
        return (best_dc, best_dc in self._local_regions)

    def _solve_lp(self, requests, dc_ids, dc_view, cost, feasible,
                  n_req, n_dc, max_batch_size, rtt_matrix=None):
        """Solve the LP and return assignments.

        Capacity constraint:
            Σᵢ x_{i,j}  ≤  max(0, cap_per_engine · engines_j − load_j)

        Where:
            current_load_j = active_requests_j + pending_requests_j
            cap_per_engine = min(max_batch_size, self.bs_cap) if both set,
                             else whichever is > 0

        Both are PER ENGINE and `engines_j` is the site's own engine count, so
        the same configuration means the same thing on a one-node site and a
        four-node one. Comparing a per-engine number against a site-wide load
        is the bug described in policies/common/capacity.py.

        The **new** piece is counting ``pending_requests`` in the load
        baseline. Without it, a DC with 32 active + 40 pending (= 72 in
        flight) reports "headroom = 32 − 32 = 0", which is correct only
        for the active counter — it silently accepts more while 40 more
        are already waiting for a semaphore slot. We instead measure
        total work at the DC (active + pending) and refuse to pile on.
        """
        # Each request goes to AT MOST one DC, not exactly one.
        #
        # With an equality the programme is infeasible the moment the batch is
        # larger than the fleet's free headroom, because "every request must be
        # placed" contradicts the capacity rows. It cannot leave one request
        # out; it fails entirely, and the greedy fallback then places the WHOLE
        # batch with no capacity check at all. Measured: 60 requests into 60
        # seats spreads 20/20/20 correctly, and 61 requests into the same 60
        # seats puts all 61 on one 20-seat datacentre. A cliff, on one request.
        #
        # Relaxed to <= 1 and paired with a reward for placing, so the solver
        # seats as many as capacity allows, at least cost, and hands only the
        # genuine overflow to the fallback. Where the batch does fit, every
        # request is still placed and the result is unchanged -- the reward is
        # constant per placement, so it cannot reorder anything.
        finite = cost[cost < 1e11]
        reward = (float(finite.max()) + 1.0) if finite.size else 1.0
        c = (cost - reward).flatten()

        # Both are PER ENGINE, so the tightest of the two is still per engine
        # and dc_headroom scales it by each site's own engine count. It used to
        # be compared against a site-wide load, which made bs_cap=max_num_seqs
        # an eighth of the real capacity on an 8-engine site -- see
        # policies/common/capacity.py.
        # SPARSE, and this is not an optimisation -- the dense form could not
        # be allocated. The assignment block is n_req rows by n_req*n_dc
        # columns, so its DENSE size grows with the SQUARE of the batch:
        # 11 MB at 300 requests, 213 MB at 1290, and 18.4 GB at 12000, with
        # np.vstack making a second copy of it. On 2026-09-07 a 4800 req/s run
        # reached memory.peak 749 GB on a 755 GB host and was OOM-killed,
        # twice, and the batch got that large because a stalled solver let the
        # buffer fill for ten seconds before the fallback ran.
        #
        # Both blocks hold exactly one 1 per variable, so the real content is
        # 2*n_req*n_dc entries -- LINEAR in the batch. At 12000 requests that
        # is 1.5 MB instead of 18.4 GB. HiGHS takes a sparse A_ub directly.
        #
        # Variable k encodes (request k//n_dc, datacentre k%n_dc), which is
        # what makes both index sets expressible without a Python loop:
        # capacity row j collects every k with k%n_dc == j, assignment row i
        # every k with k//n_dc == i.
        cap_per_engine = 0
        if max_batch_size > 0 and self.bs_cap > 0:
            cap_per_engine = min(max_batch_size, self.bs_cap)
        elif max_batch_size > 0:
            cap_per_engine = max_batch_size
        elif self.bs_cap > 0:
            cap_per_engine = self.bs_cap

        n_var = n_req * n_dc
        k = np.arange(n_var)
        n_cap = n_dc if cap_per_engine > 0 else 0

        if n_cap:
            free = [dc_headroom(dc_view[d], cap_per_engine) for d in dc_ids]
            b_ub = np.concatenate([
                np.array([float(0 if f is None else f) for f in free]),
                np.ones(n_req),
            ])
            rows = np.concatenate([k % n_dc, n_cap + k // n_dc])
            cols = np.concatenate([k, k])
        else:
            b_ub = np.ones(n_req)
            rows = k // n_dc
            cols = k

        A_ub = csr_matrix((np.ones(rows.size), (rows, cols)),
                          shape=(n_cap + n_req, n_var))

        # As an (n, 2) array rather than a list of n tuples: at 192000
        # variables the list alone is tens of megabytes of Python objects.
        bounds = np.zeros((n_var, 2))
        bounds[:, 1] = feasible.reshape(-1).astype(float)

        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                             bounds=bounds, method='highs')

            if result.success:
                x = result.x.reshape(n_req, n_dc)
                assignments = {}
                unplaced = []
                for i, req in enumerate(requests):
                    j_best = int(np.argmax(x[i]))
                    if x[i][j_best] > 0.5:
                        assignments[req.req_id] = dc_ids[j_best]
                    else:
                        unplaced.append(req)
                if unplaced:
                    # Overflow only: the fleet had no seat for these. Hand the
                    # fallback a view that already counts what this solve just
                    # seated -- against the raw snapshot every datacentre still
                    # looks as empty as it was before the batch, so the ladder
                    # sends the whole overflow to the one lowest-carbon site it
                    # has itself just filled. Measured: 60 overflowing requests
                    # all onto a datacentre with 20 free seats.
                    seated = {}
                    for d in assignments.values():
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
                    log.info("ILP seated %d/%d; %d over capacity -> fallback",
                             len(assignments), n_req, len(unplaced))
                    assignments.update(
                        self._fallback_greedy(unplaced, fb_view, rtt_matrix))
                return assignments
            else:
                log.warning("ILP solver failed: %s — falling back to greedy",
                            result.message)
                return self._fallback_greedy(requests, dc_view, rtt_matrix)
        except Exception as e:
            log.warning("ILP solve error: %s — falling back to greedy", e)
            return self._fallback_greedy(requests, dc_view, rtt_matrix)

    def _fallback_greedy(self, requests, dc_view, rtt_matrix=None):
        """Per-request carbon-optimal-under-SLA, then round-robin.

        The ladder is: the batch solve, then this. choose_dc_extended still
        looks for the lowest-carbon datacentre that meets the SLA, so the carbon
        objective survives the solver failing.

        When even that returns nothing -- no datacentre meets the SLA at all --
        the request used to be left out of `assignments` entirely, and the
        server then answered it with an ERROR. That made the ILP policies the
        only ones that can DROP requests under overload, and dropped requests
        leave the latency percentiles while counting against SLO compliance, so
        the policy looked better on the tail the more it failed. Round-robin
        instead: neutral on carbon, unlike least-loaded, which in a carbon
        policy reliably selects whatever the low-carbon datacentres are not.
        """
        assignments = {}
        for req in requests:
            result = self.choose_dc_extended(
                dc_view, self.scheduler_group, req_isl=req.in_tok,
                rtt_matrix=rtt_matrix,
            )
            if result:
                assignments[req.req_id] = result[0]
            else:
                dc_id = self._rr_fallback.pick(dc_view)
                if dc_id is not None:
                    assignments[req.req_id] = dc_id
        return assignments


# ── 3. Iterative ILP ────────────────────────────────────────────────────

