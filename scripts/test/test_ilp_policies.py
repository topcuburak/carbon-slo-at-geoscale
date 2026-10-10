#!/usr/bin/env python3
"""The ILP policies: what the solver actually decides.

These are the paper's headline policies and they were the least tested thing in
the repository -- 14-33% of their lines. The suites elsewhere check that a
policy can be built and returns a datacentre that exists. These check the
decision: does it minimise carbon, does it respect capacity, does it honour an
SLA, and what does it do when it cannot.

Several are regression tests for measured cliffs, named in the source:

  60 requests into 60 seats spread 20/20/20; 61 into the same 60 put all 61 on
  one 20-seat datacentre, because an equality constraint made the programme
  infeasible and the fallback then placed the whole batch with no capacity
  check at all.

  The overflow that did reach the fallback was priced against the raw snapshot,
  in which the datacentre the solve had just filled still looked empty.

Run:  PYTHONPATH=.:proto python3 scripts/test/test_ilp_policies.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "proto"))

import inference_pb2                                              # noqa: E402
from common.network import SCHEDULER_GROUPS                       # noqa: E402
from datacenter.perf_models.analytical.latency import AnalyticalLatencyModel  # noqa: E402
from scheduler.policies import get_policy                         # noqa: E402
from scheduler.policies.common.base import PolicyScope            # noqa: E402
from scheduler.policies.common.estimators import RequestInfo      # noqa: E402

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  ok    {name}")
        except AssertionError as e:
            FAIL.append((name, str(e)))
            print(f"  FAIL  {name}\n          {e}")
        except BaseException as e:                                # noqa: BLE001
            FAIL.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
        return fn
    return deco


_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = AnalyticalLatencyModel("llama3_8B", tp=2)
    return _MODEL


ILPS = ("ilp_carbon_sla", "iterative_ilp_carbon_sla", "facility_ilp_carbon_sla")
EAST = list(SCHEDULER_GROUPS["us_east"])


def ilp(name="ilp_carbon_sla", **kw):
    kw.setdefault("scheduler_group", "us_east")
    kw.setdefault("analytical_model", model())
    kw.setdefault("tp_size", 2)
    kw.setdefault("max_num_seqs", 32)
    return get_policy(name, **kw)


def stats(active=0, pending=0, engines=8, ci=300.0, **kw):
    s = inference_pb2.DcStats(active_requests=active, pending_requests=pending,
                              num_gpus=engines, carbon_intensity=ci)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def reqs(n, isl=256, osl=128):
    return [RequestInfo(req_id=f"r{i}", in_tok=isl, out_tok=osl,
                        predicted_osl=osl) for i in range(n)]


def counts(assignments):
    out = {}
    for d in assignments.values():
        out[d] = out.get(d, 0) + 1
    return out


# ─────────────────────────────── the shape ──────────────────────────────────

def test_every_ilp_declares_itself_global_and_batching():
    for n in ILPS:
        p = ilp(n)
        assert p.scope is PolicyScope.GLOBAL, (n, p.scope)
        assert hasattr(p, "choose_dc_batch"), n


def test_an_empty_batch_or_an_empty_fleet_places_nothing():
    for n in ILPS:
        p = ilp(n)
        assert p.choose_dc_batch([], {EAST[0]: stats()}, "us_east") == {}, n
        assert p.choose_dc_batch(reqs(3), {}, "us_east") == {}, n


def test_a_batch_that_fits_places_every_request_inside_the_view():
    view = {d: stats(ci=100.0 + 50 * i) for i, d in enumerate(EAST[:4])}
    for n in ILPS:
        a = ilp(n).choose_dc_batch(reqs(12), view, "us_east")
        assert len(a) == 12, f"{n} placed {len(a)}/12"
        assert set(a) == {r.req_id for r in reqs(12)}, n
        assert all(d in view for d in a.values()), n


def test_the_assignment_is_integral_not_fractional():
    # The LP relaxation is exact here (the constraint matrix is totally
    # unimodular), which is why HiGHS is used instead of a MILP solver. If that
    # stopped holding, a request would be split across datacentres and the
    # rounding would be silent.
    view = {d: stats(ci=100.0 + 40 * i) for i, d in enumerate(EAST[:5])}
    a = ilp("ilp_carbon_sla").choose_dc_batch(reqs(40), view, "us_east")
    assert len(a) == 40
    assert all(isinstance(d, str) and d for d in a.values()), a


# ────────────────────────── the carbon objective ────────────────────────────

def test_with_no_sla_the_cheapest_carbon_site_is_preferred():
    # Two sites identical but for intensity. Nothing else can break the tie.
    green, dirty = EAST[0], EAST[1]
    view = {green: stats(ci=50.0), dirty: stats(ci=500.0)}
    a = ilp("ilp_carbon_sla").choose_dc_batch(reqs(10), view, "us_east")
    c = counts(a)
    assert c.get(green, 0) > c.get(dirty, 0), \
        f"the dirtier site got at least as much work: {c}"


def test_reversing_the_intensities_reverses_the_choice():
    # The negative control: without it, a policy that always picks the
    # alphabetically first datacentre passes the test above.
    a_dc, b_dc = EAST[0], EAST[1]
    first = counts(ilp().choose_dc_batch(
        reqs(10), {a_dc: stats(ci=50.0), b_dc: stats(ci=500.0)}, "us_east"))
    second = counts(ilp().choose_dc_batch(
        reqs(10), {a_dc: stats(ci=500.0), b_dc: stats(ci=50.0)}, "us_east"))
    assert first.get(a_dc, 0) > first.get(b_dc, 0), first
    assert second.get(b_dc, 0) > second.get(a_dc, 0), \
        f"the choice did not follow the intensity: {second}"


def test_an_unknown_intensity_becomes_the_greenest_site_KNOWN_GAP():
    # MEASURED, and currently true. All three ILPs do `if ci <= 0: ci = 1.0`,
    # so a datacentre that reports no intensity is priced at 1 gCO2/kWh -- ten
    # to nine hundred times cheaper than any real site -- and takes the whole
    # batch. "Unknown" should be pessimistic or excluded, not free.
    #
    # Latent rather than live: every launch path sets CARBON_INTENSITY
    # (datacenter/launch.py from the archive, and the sweep and test scripts
    # explicitly). But common/config.py defaults it to 0.0, so a datacentre
    # started by hand as `python -m datacenter.server` becomes the carbon sink
    # for the whole fleet, silently.
    #
    # Pinned as the current behaviour so a fix flips the assertion rather than
    # quietly moving results.
    unknown, green = EAST[0], EAST[1]
    view = {unknown: stats(ci=0.0), green: stats(ci=10.0)}
    c = counts(ilp().choose_dc_batch(reqs(10), view, "us_east"))
    assert c.get(unknown, 0) == 10, \
        (f"an unknown intensity no longer wins outright ({c}) -- the rule may "
         f"have been fixed. Update this test to assert the new behaviour.")


def test_the_default_carbon_intensity_is_the_thing_that_makes_it_reachable():
    # The other half of the gap above, checked where it lives.
    from common import config as cfg
    assert cfg.CARBON_INTENSITY == 0.0, \
        (f"CARBON_INTENSITY now defaults to {cfg.CARBON_INTENSITY}; if it is a "
         f"real number the unknown-intensity gap above is no longer reachable "
         f"by starting a datacentre by hand.")


# ──────────────────────────────── capacity ──────────────────────────────────

def test_a_cap_limits_what_one_site_can_take():
    green, dirty = EAST[0], EAST[1]
    # ONE engine per site, so a per-engine bs_cap is also the site-wide cap and
    # the arithmetic below reads as written. bs_cap is per engine; see
    # scheduler/policies/common/capacity.py.
    view = {green: stats(ci=10.0, engines=1), dirty: stats(ci=900.0, engines=1)}
    # bs_cap 5: 8 requests cannot all fit on the green one.
    a = ilp(bs_cap=5).choose_dc_batch(reqs(8), view, "us_east")
    c = counts(a)
    assert c.get(green, 0) <= 5, f"the cap was exceeded: {c}"
    assert len(a) == 8, f"placed {len(a)}/8"


def test_pending_work_counts_against_headroom():
    # A datacentre with 5 active and 40 pending is in flight with 45. Counting
    # only the active ones reports headroom it does not have.
    green, other = EAST[0], EAST[1]
    view = {green: stats(active=5, pending=40, ci=10.0, engines=1),
            other: stats(active=0, pending=0, ci=900.0, engines=1)}
    c = counts(ilp(bs_cap=20).choose_dc_batch(reqs(10), view, "us_east"))
    assert c.get(green, 0) < 10, \
        f"pending work was ignored and the green site took everything: {c}"


def test_an_overflowing_batch_does_not_land_on_one_site():
    # The measured cliff: 60 requests into 60 seats spread correctly, 61 into
    # the same 60 put all 61 on one 20-seat datacentre. An equality constraint
    # made the programme infeasible, and the greedy fallback then placed the
    # WHOLE batch with no capacity check.
    three = EAST[:3]
    view = {d: stats(active=0, ci=100.0 + i, engines=1)
            for i, d in enumerate(three)}
    p = ilp(bs_cap=20)
    fits = counts(p.choose_dc_batch(reqs(60), view, "us_east"))
    assert max(fits.values()) <= 20, f"60 into 60 seats overflowed a site: {fits}"

    over = counts(ilp(bs_cap=20).choose_dc_batch(reqs(61), view, "us_east"))
    assert max(over.values()) <= 25, \
        (f"one more request than the fleet holds put {max(over.values())} on a "
         f"20-seat datacentre: {over}")


def test_overflow_is_priced_against_what_the_solve_just_seated():
    # The fallback used to see the raw snapshot, in which the site the solve had
    # just filled still looked empty -- so the whole overflow went to the one
    # lowest-carbon site it had itself just filled.
    green, mid, dirty = EAST[0], EAST[1], EAST[2]
    view = {green: stats(ci=10.0, engines=1), mid: stats(ci=100.0, engines=1),
            dirty: stats(ci=900.0, engines=1)}
    c = counts(ilp(bs_cap=10).choose_dc_batch(reqs(28), view, "us_east"))
    assert c.get(green, 0) <= 15, \
        f"the overflow piled back onto the site the solve had filled: {c}"
    assert len(c) >= 2, c


def test_bs_cap_is_per_engine_not_per_datacentre():
    # The bug this pins: bs_cap used to be compared against a site-wide load,
    # so setting it to max_num_seqs throttled an 8-engine site to an eighth of
    # its capacity, the LP could seat nothing, and every ILP silently ran as
    # round-robin. Two sites differing ONLY in engine count must get
    # proportionally different headroom from the same bs_cap.
    from scheduler.policies.common.capacity import dc_capacity, dc_headroom
    one = stats(active=0, engines=1)
    eight = stats(active=0, engines=8)
    assert dc_capacity(one, 128) == 128, dc_capacity(one, 128)
    assert dc_capacity(eight, 128) == 1024, dc_capacity(eight, 128)
    assert dc_headroom(stats(active=300, engines=8), 128) == 724
    # and a site-wide reading would have called that one full
    assert 300 >= 128, "the old comparison excluded this site"


def test_a_disabled_cap_is_not_a_cap_of_zero():
    # None and 0 are different answers: unconstrained versus constrained-and-
    # full. Collapsing them is how bs_cap=0 would turn into "no seats".
    from scheduler.policies.common.capacity import dc_headroom, at_cap
    s = stats(active=5000, engines=8)
    assert dc_headroom(s, 0) is None
    assert at_cap(s, 0) is False
    assert dc_headroom(s, 128) == 0
    assert at_cap(s, 128) is True


def test_the_same_config_means_the_same_thing_on_a_bigger_site():
    # A per-engine cap has to survive a change of fleet shape. 16 requests into
    # a site with 2 engines and one with 8, same bs_cap: the larger site takes
    # them and the smaller does not.
    green_small = EAST[0]
    green_big = EAST[1]
    view = {green_small: stats(active=0, engines=2, ci=10.0),
            green_big: stats(active=0, engines=8, ci=11.0)}
    c = counts(ilp(bs_cap=4).choose_dc_batch(reqs(16), view, "us_east"))
    assert c.get(green_small, 0) <= 8, f"2 engines x 4 = 8 seats, got {c}"
    assert c.get(green_big, 0) >= 8, f"8 engines x 4 = 32 seats, got {c}"


# ────────────────────────────────── SLA ─────────────────────────────────────

def test_a_site_that_cannot_meet_the_budget_is_avoided():
    slow, ok = EAST[0], EAST[1]
    # `slow` is deep in backlog and green; `ok` is empty and dirty. Without a
    # budget the green one wins; with a tight one it must not.
    view = {slow: stats(active=2000, pending=2000, ci=10.0),
            ok: stats(active=0, ci=900.0)}
    loose = counts(ilp().choose_dc_batch(reqs(6), view, "us_east"))
    tight = counts(ilp(ttft_budget_ms=50, tpot_budget_ms=25).choose_dc_batch(
        reqs(6), view, "us_east"))
    assert loose.get(slow, 0) >= tight.get(slow, 0), (loose, tight)
    assert tight.get(ok, 0) >= loose.get(ok, 0), \
        f"a tight budget did not move work off the backlogged site: {tight}"


def test_no_request_is_dropped_when_nothing_meets_the_sla():
    # Dropped requests leave the latency percentiles while still counting
    # against SLO compliance, so a policy that drops looks BETTER on the tail
    # the more it fails. Every request must come back with a datacentre.
    view = {d: stats(active=5000, pending=5000, ci=300.0) for d in EAST[:3]}
    for n in ILPS:
        a = ilp(n, ttft_budget_ms=1, tpot_budget_ms=1).choose_dc_batch(
            reqs(9), view, "us_east")
        assert len(a) == 9, f"{n} dropped {9 - len(a)} of 9 requests"
        assert all(d in view for d in a.values()), n


def test_the_fallback_is_not_least_loaded():
    # In a carbon policy, least-loaded reliably selects whatever the low-carbon
    # datacentres are not -- it turns a scheduling failure into a carbon
    # failure. The documented fallback is round-robin.
    view = {d: stats(active=5000, pending=5000, ci=300.0) for d in EAST[:4]}
    a = ilp(ttft_budget_ms=1, tpot_budget_ms=1).choose_dc_batch(
        reqs(16), view, "us_east")
    c = counts(a)
    assert len(c) > 1, f"the fallback herded onto one site: {c}"


# ─────────────────────────── the iterative one ──────────────────────────────

def test_the_iterative_solver_respects_its_round_budget():
    view = {d: stats(ci=100.0 + 30 * i) for i, d in enumerate(EAST[:4])}
    seen = []
    for rounds in (1, 3):
        p = ilp("iterative_ilp_carbon_sla", max_iterations=rounds)
        calls = {"n": 0}
        real = p._project_load

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)
        p._project_load = counting
        a = p.choose_dc_batch(reqs(12), view, "us_east")
        assert len(a) == 12, (rounds, len(a))
        seen.append(calls["n"])
    assert seen[0] <= seen[1], seen
    assert seen[1] <= 3, f"3 rounds projected {seen[1]} times"


def test_the_iterative_solver_prices_the_next_round_on_the_last_one():
    # Without the projection every round sees the same empty fleet and the
    # iteration is a no-op.
    green, dirty = EAST[0], EAST[1]
    view = {green: stats(active=0, ci=10.0, engines=1),
            dirty: stats(active=0, ci=900.0, engines=1)}
    p = ilp("iterative_ilp_carbon_sla", max_iterations=3, bs_cap=8)
    a = p.choose_dc_batch(reqs(16), view, "us_east")
    c = counts(a)
    assert len(a) == 16, len(a)
    assert c.get(green, 0) <= 10, \
        f"the rounds never saw the load they had placed: {c}"


def test_one_round_of_the_iterative_matches_the_plain_ilp():
    # With a single round there is nothing to project, so the two must agree.
    view = {d: stats(ci=100.0 + 40 * i) for i, d in enumerate(EAST[:4])}
    plain = ilp("ilp_carbon_sla").choose_dc_batch(reqs(10), view, "us_east")
    once = ilp("iterative_ilp_carbon_sla",
               max_iterations=1).choose_dc_batch(reqs(10), view, "us_east")
    assert counts(plain) == counts(once), (counts(plain), counts(once))


# ─────────────────────────── the facility one ───────────────────────────────

def test_the_facility_solver_places_everything_and_opens_sites():
    view = {d: stats(ci=100.0 + 30 * i) for i, d in enumerate(EAST[:5])}
    a = ilp("facility_ilp_carbon_sla").choose_dc_batch(reqs(20), view, "us_east")
    assert len(a) == 20, len(a)
    assert all(d in view for d in a.values())


def test_the_facility_solver_concentrates_rather_than_spreading():
    # The binary "is this site open" variable is the point: it should prefer
    # filling a green site to lighting up a dirty one.
    view = {EAST[0]: stats(ci=10.0), EAST[1]: stats(ci=800.0),
            EAST[2]: stats(ci=900.0)}
    c = counts(ilp("facility_ilp_carbon_sla").choose_dc_batch(
        reqs(12), view, "us_east"))
    assert c.get(EAST[0], 0) >= max(c.get(EAST[1], 0), c.get(EAST[2], 0)), c


# ─────────────────────────── degrading gracefully ───────────────────────────

def test_a_solver_failure_still_places_every_request():
    # The solver itself failing, not the method around it: the try/except lives
    # inside _solve_lp, so replacing the whole method removes the very
    # protection under test.
    import scheduler.policies.common.ilp_base as base
    view = {d: stats(ci=100.0 + 10 * i) for i, d in enumerate(EAST[:3])}
    real = base.linprog
    base.linprog = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("solver unavailable"))
    try:
        a = ilp("ilp_carbon_sla").choose_dc_batch(reqs(9), view, "us_east")
    finally:
        base.linprog = real
    assert len(a) == 9, f"placed {len(a)}/9 after the solver failed"
    assert all(d in view for d in a.values()), a


def test_without_scipy_the_policy_degrades_instead_of_dying():
    import scheduler.policies.ilp_carbon_sla as mod
    view = {d: stats(ci=100.0 + 10 * i) for i, d in enumerate(EAST[:3])}
    real = mod.HAS_SCIPY
    mod.HAS_SCIPY = False
    try:
        a = ilp("ilp_carbon_sla").choose_dc_batch(reqs(9), view, "us_east")
    finally:
        mod.HAS_SCIPY = real
    assert len(a) == 9, f"placed {len(a)}/9 with no scipy"


def test_a_single_datacentre_is_not_a_special_case():
    view = {EAST[0]: stats(ci=200.0)}
    for n in ILPS:
        a = ilp(n).choose_dc_batch(reqs(5), view, "us_east")
        assert len(a) == 5, (n, len(a))
        assert set(a.values()) == {EAST[0]}, n


def test_a_single_request_is_not_a_special_case():
    view = {d: stats(ci=100.0 + 40 * i) for i, d in enumerate(EAST[:3])}
    for n in ILPS:
        a = ilp(n).choose_dc_batch(reqs(1), view, "us_east")
        assert len(a) == 1, (n, a)


def test_the_constraint_matrix_is_sparse_and_linear_in_the_batch():
    """The dense form could not be allocated, and that is not hyperbole.

    The assignment block is n_req rows by n_req*n_dc columns, so DENSE it grows
    with the SQUARE of the batch: 11 MB at 300 requests, 213 MB at 1290, and
    18.4 GB at 12000 -- with np.vstack making a second copy. On 2026-09-07 a
    4800 req/s run reached memory.peak 749 GB on a 755 GB host and was
    OOM-killed twice before this was found.

    Both blocks hold exactly one 1 per variable, so the honest size is
    2*n_req*n_dc entries. This asserts that directly, on the matrix actually
    handed to the solver, because a future edit back to np.zeros would still
    pass every behavioural test in this file -- it would just fall over at
    4800 req/s, at night, on the machine.
    """
    import scipy.sparse as sp
    import scheduler.policies.common.ilp_base as ilp_base

    seen = {}
    real = ilp_base.linprog

    def spy(c, A_ub=None, b_ub=None, bounds=None, method=None):
        seen["A_ub"] = A_ub
        return real(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)

    view = {d: stats(engines=8, ci=100.0 + 50 * i)
            for i, d in enumerate(EAST[:4])}
    n_req, n_dc = 200, len(view)

    ilp_base.linprog = spy
    try:
        ilp().choose_dc_batch(reqs(n_req), view, "us_east",
                              max_batch_size=32, rtt_matrix={})
    finally:
        ilp_base.linprog = real

    A = seen.get("A_ub")
    assert A is not None, "the solver was never called"
    assert sp.issparse(A), f"the constraint matrix is dense: {type(A)}"
    # n_dc capacity rows + n_req assignment rows, one entry per variable each.
    assert A.shape == (n_dc + n_req, n_req * n_dc), A.shape
    assert A.nnz == 2 * n_req * n_dc, (
        f"{A.nnz} non-zeros, expected {2 * n_req * n_dc}")


def test_a_capacity_cap_is_what_makes_the_lp_more_than_a_greedy_rule():
    """Without a per-engine cap the LP has no constraint coupling requests.

    The problem then SEPARATES by request -- each one independently takes its
    cheapest feasible site -- so the solve is a per-request argmin wearing an
    LP's clothes. That was the state of every ILP run in this project:
    `_batch_loop` never passed max_batch_size and BS_CAP defaults to 0, so
    cap_per_engine was 0 and no A_ub was ever built. It is also why all three
    ILPs tied carbon_aware:0.0, which IS pure-carbon greedy.
    """
    # Two engines a site, so a cap of 3 per engine is 6 seats -- less than the
    # batch, which is what makes the constraint bind.
    view = {EAST[0]: stats(engines=2, ci=40.0),
            EAST[1]: stats(engines=2, ci=300.0),
            EAST[2]: stats(engines=2, ci=500.0)}
    batch = reqs(12)

    loose = ilp().choose_dc_batch(batch, view, "us_east", max_batch_size=0,
                                  rtt_matrix={})
    tight = ilp().choose_dc_batch(batch, view, "us_east", max_batch_size=3,
                                  rtt_matrix={})

    greenest = EAST[0]
    assert counts(loose).get(greenest, 0) == len(batch), (
        "uncapped, the LP should put everything on the cheapest site -- if it "
        f"does not, this test no longer shows what it claims: {counts(loose)}")
    assert counts(tight).get(greenest, 0) <= 6, (
        f"a 3-per-engine cap over 2 engines is 6 seats, but the LP placed "
        f"{counts(tight).get(greenest, 0)} there: {counts(tight)}")
    assert len(counts(tight)) > 1, \
        f"the capped solve still used one site: {counts(tight)}"


def test_the_solver_process_answers_exactly_what_in_process_does():
    """The offload is a move, not a rewrite, so it must not change an answer.

    The batch build is interpreter-bound and shares a GIL with the event loop;
    at 4800 req/s that starved it and 26.5% of requests came back unplaced.
    Moving it to its own process fixes the contention only if the assignment
    is identical for the same input, which is what this checks -- the ILP is
    deterministic given (requests, view, rtt), so anything but equality is a
    defect.
    """
    from scheduler.policies.common.batch_worker import (BatchSolver,
                                                        BatchSolverUnavailable)

    view = {d: stats(active=a, engines=8, ci=c)
            for d, a, c in zip(EAST[:4], (0, 12, 40, 4), (90.0, 300.0, 520.0, 150.0))}
    batch = reqs(24)

    here = ilp().choose_dc_batch(batch, view, "us_east", rtt_matrix={})

    solver = BatchSolver({
        "policy_name": "ilp_carbon_sla",
        "alpha": 0.0,
        "scheduler_group": "us_east",
        "model_name": "llama3_8B",
        "tp_size": 2,
        "policy_kwargs": {"max_num_seqs": 32, "tp_size": 2},
    })
    solver.start()
    try:
        there = solver.solve(batch, view, "us_east", {}, timeout=120.0)
    finally:
        solver.close()

    assert there == here, (
        "the solver process placed requests differently:\n"
        f"  in-process: {counts(here)}\n"
        f"  in-worker : {counts(there)}")


def test_a_dead_solver_is_reported_rather_than_waited_on():
    """A batch policy that cannot reach its solver must degrade, not stop.

    server.py catches this and solves in-process, which is only safe if the
    failure surfaces promptly instead of blocking the batch thread.
    """
    from scheduler.policies.common.batch_worker import (BatchSolver,
                                                        BatchSolverUnavailable)
    import time as _t

    solver = BatchSolver({
        "policy_name": "ilp_carbon_sla", "alpha": 0.0,
        "scheduler_group": "us_east", "model_name": "llama3_8B", "tp_size": 2,
        "policy_kwargs": {"max_num_seqs": 32, "tp_size": 2},
    })
    # Never started, so there is no process to answer.
    t0 = _t.monotonic()
    try:
        solver.solve(reqs(4), {EAST[0]: stats()}, "us_east", {}, timeout=5.0)
        raise AssertionError("a solve against a dead solver returned")
    except BatchSolverUnavailable:
        pass
    assert _t.monotonic() - t0 < 1.0, "it waited instead of failing fast"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        check(t.__name__)(t)
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    sys.exit(1 if FAIL else 0)
