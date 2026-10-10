"""Shared estimators for the policies that reason about cost.

Every policy that scores a placement rather than just picking one needs the
same four things: what a hop costs, what a step costs in time, what it costs in
energy, and whether the result would meet the SLA. These used to live at the
top of ilp_carbon.py, which meant importing a 792-line module containing three
policies to get at them -- and RequestInfo, which the testbed harnesses import
directly, came from the same place.
"""

import logging
import os
from typing import Dict, Tuple

import inference_pb2
from common.network import REGION_TO_GROUP, rtt_pair_key
from scheduler.policies.common.carbon_signal import effective_carbon_intensity
from scheduler.policies.common.latency_estimates import (
    estimate_ttft_ms,
    estimate_tpot_ms,
)

log = logging.getLogger(__name__)


def _lookup_rtt_ms(rtt_matrix: Dict[str, float], my_group: str, dc_id: str) -> float:
    """Return dynamic RTT in ms from my scheduler group to the candidate DC.

    The rtt_matrix is keyed by sorted group-pair strings (see
    common/network.rtt_pair_key). 0 for same-group (no cross-region hop).
    """
    if not rtt_matrix:
        return 0.0
    dc_group = REGION_TO_GROUP.get(dc_id, my_group)
    if dc_group == my_group:
        return 0.0
    key = rtt_pair_key(my_group, dc_group)
    return float(rtt_matrix.get(key, 0.0))

log = logging.getLogger("scheduler.ilp")

try:
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
    HAS_SCIPY = True
except ImportError:
    linprog = None
    csr_matrix = None
    HAS_SCIPY = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


# ── Shared utilities ─────────────────────────────────────────────────────

_STEP_MODEL_CACHE = {}


def _step_model(analytical_model, tp_size):
    """The measured per-step energy model for whatever this DC serves.

    Resolved from the analytical model's name when there is one, otherwise from
    MODEL_NAME, which is what the sweep scripts set. Returns None when that
    (model, TP) was never calibrated -- the caller then falls back to the old
    three-state estimate rather than rescaling somebody else's coefficients.
    """
    name = getattr(analytical_model, "model_name", "") or os.environ.get("MODEL_NAME", "")
    key = (name, tp_size)
    if key not in _STEP_MODEL_CACHE:
        try:
            from datacenter.perf_models.analytical.energy import StepEnergyModel
            _STEP_MODEL_CACHE[key] = StepEnergyModel.load(name, tp_size)
        except Exception:
            _STEP_MODEL_CACHE[key] = None
    return _STEP_MODEL_CACHE[key]


def _estimate_energy_j(osl: int, tpot_ms: float, bs: int,
                       analytical_model=None,
                       req_isl: int = 0, tp_size: int = 2,
                       stats=None) -> float:
    """Energy in joules that placing this request at this DC would add.

    MARGINAL, not total. A scheduling decision asks what the request COSTS
    here, and an engine that is already serving other requests pays the fixed
    per-step cost e0 whether or not this one joins. Charging the request the
    whole step -- which the previous version did, via power_w(bs) x tpot x osl
    -- multiplies that shared cost by the batch size across the batch, and the
    error grows with load, so it biased the policy against busy DCs for a
    reason that is an accounting artifact.

    Per step this request adds
        e_d              its own decoding slot
        e_kv * kv_self   its own cached tokens
    and its prefill adds e_p per prompt token. The cache grows from ISL to
    ISL+OSL over the request's life, so its mean is ISL + OSL/2.

    Marginal costing assumes the steps were running anyway, and there are two
    places where that is false. Both were left out, both made a busy site look
    cheaper than it is, and both are charged here when `stats` is supplied.

    THE PREFILL STEP'S FIXED COST. Only e_p * ISL was charged, but a prefill
    step also costs e0 + e0_p, and e0_plo on top when the pass is below the
    plateau threshold where the device does not saturate. On an engine with an
    empty prefill queue this request is the whole reason that step runs, so it
    causes all of it. With prompts already queued the step happens regardless
    and they fuse, so only the marginal e_p * ISL applies.

    KEEPING THE ENGINE ALIVE. Every decode step costs e0 whether or not this
    request is in it -- unless this request is the reason the step happens,
    which is true past the point where everything else has finished. Charged
    only when the engine has no queue, because a queue means it keeps stepping
    regardless of this request, and only for the steps beyond the batch's own
    drain. The batch owes total_remaining_osl over d decoders, so its mean
    remaining is R/d and, for a spread that is roughly uniform, its LAST
    request finishes near 2R/d. Using the max rather than the mean makes this
    deliberately conservative: it still under-charges when the spread is wider
    than uniform, which is the same direction as the bug it replaces, so it
    cannot overshoot into penalising busy sites for a new accounting reason.

    Neither term can be scored the way the TTFT estimator was. A placement's
    marginal energy is a counterfactual -- what the datacentre would have burnt
    without this request -- and one run does not observe it. They are argued
    from the step model's own decomposition instead.
    """
    return sum(_split_energy_j(osl, tpot_ms, bs, analytical_model,
                               req_isl, tp_size, stats))


def _split_energy_j(osl: int, tpot_ms: float, bs: int,
                    analytical_model=None, req_isl: int = 0,
                    tp_size: int = 2, stats=None):
    """The same estimate, split into (marginal, keepalive).

    Marginal is what the request costs at a datacentre that is already running:
    its prefill tokens, its decode slot, its own KV. Keepalive is the part that
    exists only because this request is the reason the engine steps at all --
    the prefill step's fixed cost when nothing is queued, and e0 for every
    decode step past the point the existing batch would have drained.

    Split because the two behave differently in an optimisation. Marginal is
    linear in the number of requests placed; keepalive is paid ONCE however many
    go there, which is an economy of scale and therefore concave. A linear
    programme cannot express that -- its relaxation wants to spread where the
    truth is to concentrate -- so a policy that wants to model it needs the
    keepalive as a fixed charge on a binary "is this datacentre open" variable,
    which is the capacitated facility location form.

    _estimate_energy_j is the sum, unchanged.
    """
    m = _step_model(analytical_model, tp_size)
    if m is None:
        return (_estimate_energy_j_three_state(osl, tpot_ms, bs,
                                               analytical_model, req_isl,
                                               tp_size), 0.0)

    if stats is not None:
        engines = max(int(getattr(stats, 'num_gpus', 1) or 1), 1)
        pending = int(getattr(stats, 'pending_requests', 0) or 0) / engines
        decoding = (int(getattr(stats, 'decoding_requests', 0) or 0)
                    or int(getattr(stats, 'active_requests', 0) or 0)) / engines
        remaining = int(getattr(stats, 'total_remaining_osl', 0) or 0) / engines
    else:
        pending = decoding = remaining = None

    # e_p is linear in the token count; e_att carries attention, which is
    # quadratic in the sequence's own length. One request is one sequence
    # here, so sum_i(n_i^2) is just isl^2. Zero on artifacts fitted before
    # the term existed.
    prefill_j = (m.e_p * req_isl + m.e_att * req_isl * req_isl
                 if req_isl > 0 else 0.0)
    fixed_j = 0.0
    idle_credit = 0.0
    if req_isl > 0 and pending is not None and pending <= 0:
        fixed_j += m.e0 + m.e0_p
        if m.plateau_tokens and req_isl < m.plateau_tokens:
            fixed_j += m.e0_plo
        # The prefill step displaces idle too, for as long as it runs -- but
        # ONLY on an engine that is otherwise doing nothing. Where a batch is
        # decoding, this step displaces a decode step, not idle, so no credit
        # is due: `decoding > 0` is exactly that test. Measured at 19% of the
        # step's energy across every ISL from 200 to 12000, a share that
        # EXCEEDS the step's own fixed part (e0 + e0_p = 4.87 J), so it cannot
        # be netted out of `fixed` alone. Credited against the whole step
        # below, clamped at zero.
        if (m.idle_w > 0 and analytical_model is not None
                and decoding is not None and decoding <= 0):
            try:
                idle_credit += m.idle_w * (
                    analytical_model.ttft_ms(req_isl, bs=1) / 1000.0)
            except Exception:
                pass

    n_steps = max(0, osl)
    kv_self = req_isl + osl / 2.0
    decode_j = n_steps * (m.e_d + m.e_kv * kv_self)

    # Order matters, and it used to be wrong. `bs` is int(active / engines), so
    # it rounds to 0 whenever a datacentre has fewer active requests than
    # engines -- a LIGHTLY LOADED site, not one with no batch information. The
    # bs <= 0 branch ran first and charged such a site the whole OSL at e0 even
    # when it had a queue of 36, which contradicts this function's own rule
    # that a queue means the engine steps regardless. The effect was a cliff:
    # at 8 engines, active=7 with 36 queued estimated 4317 J and active=8 with
    # the same queue estimated 443 J, a 10x jump for one more request. It made
    # lightly loaded datacentres look ten times more carbon-expensive than they
    # are and pushed all three carbon policies toward busy ones -- the opposite
    # of the bias the marginal accounting was introduced to remove.
    if pending is not None and pending > 0:
        # A queue means the steps happen whether or not this request joins.
        solo_steps = 0.0
    elif pending is not None and decoding > 0:
        solo_steps = max(0.0, n_steps - 2.0 * remaining / decoding)
    elif bs <= 0:
        # No snapshot at all, or an engine with genuinely nothing on it: this
        # request is the only reason the steps happen.
        solo_steps = n_steps
    else:
        solo_steps = 0.0
    # Net out the idle the datacentre pays anyway.
    #
    # e0 is the intercept of the step-energy fit: what a step costs before any
    # request's own work. That is the right number for "what did this engine
    # spend", which is what the simulator asks. It is the wrong number for
    # "what does this engine spend EXTRA if I place here", which is what a
    # scheduler asks, because the counterfactual is not a powered-down engine.
    # It is an idle one, still drawing idle_w.
    #
    # The two branches of this function were answering different questions. On
    # a loaded datacentre the estimate is a difference -- bs+1 against bs -- and
    # checks out exactly: 82.7 J by hand, 82.7 J from here. On an idle one it
    # returned the TOTAL, 1945.3 J, where the difference against 5.56 s of
    # idling is 1127.7 J. A 1.73x overcharge, and always in the direction of
    # making an idle datacentre look worse than it is.
    if solo_steps > 0 and m.idle_w > 0:
        step_s = tpot_ms / 1000.0 if tpot_ms and tpot_ms > 0 else 0.0
        if step_s <= 0 and analytical_model is not None:
            # Callers that price a site rather than a placement pass no step
            # time; derive it rather than silently skip the correction.
            try:
                kv = (req_isl + osl / 2.0) * max(1, bs)
                step_s = analytical_model.tpot_ms(
                    max(1, bs), total_kv=int(kv)) / 1000.0
            except Exception:
                step_s = 0.0
        if step_s > 0:
            idle_credit += solo_steps * m.idle_w * step_s
    fixed_j += solo_steps * m.e0

    # Clamped at zero. The credit can exceed the fixed terms for a long prompt
    # (at ISL 12000 the prefill step displaces 86 J of idle against 4.87 J of
    # fixed cost), and a negative keepalive would tell the facility MILP that
    # opening a site is a reward. The lost accuracy is small -- about 0.5% of
    # the placement's total at ISL 1500 -- and it keeps `fixed` meaning what
    # the formulation needs it to mean: what is paid ONCE, however many
    # requests land here.
    fixed_j = max(0.0, fixed_j - idle_credit)

    return (prefill_j + decode_j, fixed_j)


def _estimate_energy_j_three_state(osl: int, tpot_ms: float, bs: int,
                                   analytical_model=None,
                                   req_isl: int = 0, tp_size: int = 2) -> float:
    """Legacy estimate, for (model, TP) pairs with no measured artifact.

    Kept as written except for one fix: decode power is now TP-scaled the same
    way prefill is. Before, prefill was multiplied by tp_size and decode was
    not, so the two phases of the same request were priced on different
    hardware -- which pushed the policy away from long prompts for no physical
    reason.
    """
    if analytical_model is not None:
        decode_power = analytical_model.power_w(max(1, bs)) * (tp_size / 2.0)
    else:
        decode_power = 300.0 * (tp_size / 2.0)
    decode_energy = decode_power * (tpot_ms / 1000.0) * osl

    prefill_energy = 0.0
    if analytical_model is not None and req_isl > 0:
        prefill_energy = 400.0 * tp_size * analytical_model.ttft_s(req_isl)

    return prefill_energy + decode_energy


def _get_carbon_intensity(stats: inference_pb2.DcStats) -> float:
    """The carbon price of a site: intensity x PUE; see carbon_signal."""
    return effective_carbon_intensity(stats)


def _estimate_ttft(stats, req_isl: int, analytical_model,
                   fwd_ms: float = 0.0, max_num_seqs: int = 0) -> float:
    return estimate_ttft_ms(stats, req_isl, analytical_model, fwd_ms=fwd_ms,
                            max_num_seqs=max_num_seqs)


def _estimate_tpot(stats, analytical_model, bs_offset: int = 1,
                   req_osl: int = 0, req_isl: int = 0) -> float:
    return estimate_tpot_ms(stats, analytical_model, bs_offset=bs_offset,
                            req_osl=req_osl, req_isl=req_isl)


def _check_sla(
    stats,
    req_isl: int,
    ttft_budget_ms: float,
    tpot_budget_ms: float,
    analytical_model,
    bs_offset: int = 1,
    fwd_ms: float = 0.0,
    max_num_seqs: int = 0,
    req_osl: int = 0,
) -> Tuple[bool, float, float]:
    """Check SLA for a (request, DC) pair using analytical model.

    TTFT: includes queue wait from total_pending_isl; compared against the
          FLAT ``ttft_budget_ms``. It is no longer an additive slack over the
          prompt's own prefill -- see the note in the body.
    TPOT: uses projected BS (current + bs_offset) against a flat budget.

    Args:
        bs_offset: +1 for greedy (single request), 0 for ILP (batch projected).
        req_osl: what the placement is priced on. It sets the horizon over
            which the queued prefill backlog is charged against TPOT, and
            without it that term is off -- which left the TPOT estimate 1.66x
            optimistic on the concentrating arms and the gate never rejecting.

    Returns:
        (sla_ok, ttft_ms, tpot_ms)
    """
    ttft_ms = _estimate_ttft(stats, req_isl, analytical_model, fwd_ms=fwd_ms,
                             max_num_seqs=max_num_seqs)
    tpot_ms = _estimate_tpot(stats, analytical_model, bs_offset=bs_offset,
                             req_osl=req_osl, req_isl=req_isl)

    # TTFT_BUDGET is a FLAT ceiling in ms, compared against the estimate
    # directly. It used to be an additive slack over 1.5 x prefill_ms(ISL),
    # which meant the budget grew with the prompt -- 1530 ms at ISL 128 and
    # 3104 ms at ISL 20000 -- so a long prompt could never breach it and the
    # constraint bound nothing: measured 0 of 6656 (request, DC) pairs
    # infeasible at the loads the sweep runs.
    sla_ok = True
    if ttft_budget_ms > 0 and ttft_ms > ttft_budget_ms:
        sla_ok = False
    if tpot_budget_ms > 0 and tpot_ms > tpot_budget_ms:
        sla_ok = False

    return sla_ok, ttft_ms, tpot_ms


# ── Request info for batch policies ──────────────────────────────────────

class RequestInfo:
    """Lightweight container for a pending request's features."""
    __slots__ = ('req_id', 'in_tok', 'out_tok', 'predicted_osl', 'model_id')

    def __init__(self, req_id: str, in_tok: int, out_tok: int,
                 predicted_osl: int = 0, model_id: str = ""):
        self.req_id = req_id
        self.in_tok = in_tok
        self.out_tok = out_tok
        self.predicted_osl = predicted_osl if predicted_osl > 0 else out_tok
        self.model_id = model_id



# ── Last-resort fallback ─────────────────────────────────────────────────

class RoundRobinFallback:
    """What a carbon policy does when no datacentre meets the SLA.

    Deliberately NOT least-loaded, which is what every carbon policy used to do
    here. A carbon policy has been sending its work to the low-carbon
    datacentres, so those are the loaded ones; "least loaded" then selects, with
    fair reliability, the datacentre no policy wanted -- the carbon-expensive
    one. Falling back that way converts an SLA failure into a carbon failure,
    and does it precisely in the overloaded regime where both matter most.

    Round-robin is not smart, but it is neutral: it does not correlate with
    carbon in either direction, and it spreads rather than piling onto whichever
    datacentre the stale snapshot happened to call emptiest.

    One counter per policy instance, so each scheduler rotates independently.
    """

    def __init__(self):
        self._i = 0

    def pick(self, dc_view):
        ids = sorted(dc_view)
        if not ids:
            return None
        dc = ids[self._i % len(ids)]
        self._i += 1
        return dc
