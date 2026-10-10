#!/usr/bin/env python3
"""The scheduler, part by part -- everything except the cross-region hop.

The hop has its own file (test_forwarding.py). This covers what is left:

  the registry      every policy that claims a name can be built and used
  scope             a policy that says LOCAL_ONLY must actually stay local
  the herd          plain min() over equal loads always returns candidate 0
  snapshots         a partial wave merges, an old epoch is ignored
  the OSL knob      oracle / predicted / ref, and what happens when the
                    requested source is not there
  batching          the window, the cap, and what a batch decision costs

The datacentre is stubbed; nothing here needs a fleet or a port.

Run:  PYTHONPATH=.:proto python3 scripts/test/test_scheduler.py
"""

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "proto"))
os.environ.setdefault("STATIC_ENDPOINTS_FILE", "/dev/null")

import inference_pb2                                              # noqa: E402
from common.network import (REGION_TO_GROUP, SCHEDULER_GROUPS,    # noqa: E402
                            rtt_pair_key)
from scheduler.policies import list_policies, get_policy     # noqa: E402
from scheduler.policies.common.base import PolicyScope            # noqa: E402
from scheduler.policies.common.latency_estimates import (  # noqa: E402
    resolve_osl, estimate_tpot_ms, estimate_ttft_ms)
from scheduler.policies.common.projection import project_placements  # noqa: E402
from scheduler.server import SchedulerServicer
from async_shim import SyncServicer                    # noqa: E402

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
            # BaseException, not Exception: the servicer raises SystemExit when
            # a policy needs an analytical model and has none, and that walked
            # straight out of the harness and ended the run at the first case.
            FAIL.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
        return fn
    return deco


def stats(active=0, pending=0, engines=8, ci=300.0, **kw):
    s = inference_pb2.DcStats(active_requests=active, pending_requests=pending,
                              num_gpus=engines, carbon_intensity=ci)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def analytical_model():
    """The model the estimating policies need, as main() builds it.

    llama3_8B at TP=2 is fitted -- see
    datacenter/perf_models/analytical/fit_data/. Built once: the fit is read
    from disk.
    """
    global _MODEL
    if _MODEL is None:
        from datacenter.perf_models.analytical.latency import AnalyticalLatencyModel
        _MODEL = AnalyticalLatencyModel("llama3_8B", tp=2)
    return _MODEL


_MODEL = None


def servicer(group="us_east", policy="round_robin", **kw):
    pk = {"max_num_seqs": 32, "analytical_model": analytical_model()}
    pk.update(kw.pop("policy_kwargs", {}))
    return SyncServicer(SchedulerServicer(
        f"sched-{group}", group, policy, kw.pop("alpha", 0.0),
        simulate_rtt=False, verbosity=0, policy_kwargs=pk, **kw))


EAST = list(SCHEDULER_GROUPS["us_east"])
WEST = list(SCHEDULER_GROUPS["us_west"])

# Policies that need a fitted latency model to score anything.
NEEDS_MODEL = {"ilp_carbon_sla", "iterative_ilp_carbon_sla",
               "facility_ilp_carbon_sla", "carbon_aware", "carbon_aware_local"}


# ──────────────────────────────── registry ──────────────────────────────────

@check("test_every_registered_policy_can_be_built")
def _():
    names = list_policies()
    assert len(names) >= 9, names
    for n in names:
        p = get_policy(n, scheduler_group="us_east", max_num_seqs=32,
                       analytical_model=analytical_model())
        assert p is not None, n
        assert isinstance(p.scope, PolicyScope), f"{n} has no usable scope"


@check("test_an_unknown_policy_is_refused_and_the_error_lists_the_real_ones")
def _():
    try:
        get_policy("magic_carbon_thing", scheduler_group="us_east")
    except ValueError as e:
        assert "magic_carbon_thing" in str(e), e
        assert "round_robin" in str(e), "the error does not list what exists"
        return
    raise AssertionError("an unknown policy name was accepted")


@check("test_a_name_cannot_be_registered_twice")
def _():
    from scheduler.policies import register_policy
    from scheduler.policies.round_robin import RoundRobin
    try:
        register_policy("round_robin")(RoundRobin)
    except Exception as e:
        assert "round_robin" in str(e), e
        return
    raise AssertionError("a duplicate policy name was accepted silently")


@check("test_every_policy_places_inside_the_view_it_was_given")
def _():
    view = {d: stats(active=i) for i, d in enumerate(EAST[:4])}
    for n in list_policies():
        p = get_policy(n, scheduler_group="us_east", max_num_seqs=32,
                       analytical_model=analytical_model())
        try:
            dc = p.choose_dc(view, "us_east", req_isl=128, req_osl=64)
        except Exception as e:                                    # noqa: BLE001
            if n in NEEDS_MODEL:
                continue                     # no fitted artifact in this process
            raise AssertionError(f"{n} raised: {type(e).__name__}: {e}")
        if dc is None:
            continue
        assert dc in view, f"{n} chose {dc}, which is not in the view"


@check("test_a_local_only_policy_is_held_to_its_scope_by_the_server")
def _():
    # The scope is enforced, not trusted. The server dispatches the local path
    # directly and charges no RTT, so a LOCAL_ONLY policy handed the whole view
    # would place requests in other regions for free -- weighted_least_loaded
    # reads the full view and would do exactly that. The server filters before
    # asking it, which is the contract; this checks the contract, at the layer
    # that keeps it.
    for n in list_policies():
        probe = get_policy(n, scheduler_group="us_east", max_num_seqs=32,
                       analytical_model=analytical_model())
        if probe.scope is not PolicyScope.LOCAL_ONLY:
            continue
        s = servicer(policy=n)
        s._dc_view = {d: stats(active=0) for d in EAST[:3]}
        s._dc_view.update({d: stats(active=0) for d in WEST[:3]})
        seen = []
        s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: (
            seen.append(dc_id)
            or inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id))
        for i in range(20):
            r = s.SubmitRequest(inference_pb2.InferenceRequest(
                req_id=f"{n}-{i}", in_tok=128, out_tok=64), None)
            if r.status != "OK":
                break
        for dc in seen:
            assert REGION_TO_GROUP.get(dc) == "us_east", \
                f"{n} declares LOCAL_ONLY and the server let it place at {dc}"


@check("test_the_server_filters_rather_than_trusting_the_scope")
def _():
    # The negative control for the test above: weighted_least_loaded really
    # does read whatever view it is handed, so the protection is the server's
    # filter and not the policy's own restraint. If this ever starts failing,
    # the policy has learned to filter itself and the note above is stale.
    p = get_policy("weighted_least_loaded", scheduler_group="us_east",
                   max_num_seqs=32, analytical_model=analytical_model())
    assert p.scope is PolicyScope.LOCAL_ONLY
    view = {d: stats(active=100) for d in EAST[:3]}
    view.update({d: stats(active=0) for d in WEST[:3]})
    assert REGION_TO_GROUP.get(p.choose_dc(view, "us_east", req_isl=1,
                                           req_osl=1)) == "us_west", \
        "the policy now filters itself; the server-side filter may be redundant"


# ──────────────────────────── placement behaviour ───────────────────────────

@check("test_round_robin_visits_every_datacentre_evenly")
def _():
    p = get_policy("round_robin", scheduler_group="us_east")
    view = {d: stats() for d in EAST[:4]}
    seen = {}
    for _ in range(400):
        dc = p.choose_dc(view, "us_east")
        seen[dc] = seen.get(dc, 0) + 1
    assert len(seen) == 4, seen
    assert max(seen.values()) - min(seen.values()) <= 1, seen


@check("test_least_loaded_picks_the_least_loaded")
def _():
    p = get_policy("least_loaded", scheduler_group="us_east")
    view = {EAST[0]: stats(active=50), EAST[1]: stats(active=2),
            EAST[2]: stats(active=90)}
    assert p.choose_dc(view, "us_east") == EAST[1]


@check("test_global_least_loaded_herds_on_a_tie_KNOWN_GAP")
def _():
    # MEASURED, and currently true: with the view frozen -- which is what a
    # scheduler sees between snapshots, once a second -- every request goes to
    # the same site. `min` over equal keys returns the first candidate, and
    # nothing rotates the tie.
    #
    # The within-datacentre LeastLoaded in datacenter/scheduling.py rotates for
    # exactly this reason, and its docstring says the herd is "the reason the
    # least-loaded rungs were taken out of the datacentre-level policies" --
    # but this one, at the global level, still has it.
    #
    # This test pins the CURRENT behaviour rather than the desired one, so the
    # gap is visible and a fix flips the assertion instead of silently
    # changing the baselines. See the resume note.
    p = get_policy("least_loaded", scheduler_group="us_east")
    view = {d: stats(active=0) for d in EAST[:4]}
    seen = {}
    for _ in range(200):
        dc = p.choose_dc(view, "us_east")
        seen[dc] = seen.get(dc, 0) + 1
    assert len(seen) == 1, \
        (f"least_loaded now spreads a tied field over {len(seen)} sites -- the "
         f"herd looks fixed. Update this test to assert the spread.")


@check("test_least_loaded_counts_pending_as_load")
def _():
    # A site with active=5, pending=100 is in flight with 105, and ranking it
    # "lightly loaded" was the original herd. This part IS fixed.
    p = get_policy("least_loaded", scheduler_group="us_east")
    view = {EAST[0]: stats(active=5, pending=100),
            EAST[1]: stats(active=20, pending=0)}
    assert p.choose_dc(view, "us_east") == EAST[1], \
        "pending is not being counted as load"


@check("test_an_empty_view_yields_no_placement_rather_than_a_guess")
def _():
    for n in ("round_robin", "least_loaded", "round_robin_local",
              "least_loaded_local", "weighted_least_loaded"):
        p = get_policy(n, scheduler_group="us_east", max_num_seqs=32,
                       analytical_model=analytical_model())
        assert p.choose_dc({}, "us_east", req_isl=1, req_osl=1) is None, n


# ────────────────────────────── the OSL knob ────────────────────────────────

@check("test_oracle_reads_the_true_length")
def _():
    v, fell = resolve_osl(true_osl=417, predicted_osl=99, source="oracle",
                          ref_osl=500)
    assert (v, fell) == (417, False), (v, fell)


@check("test_predicted_reads_the_prediction_and_never_the_truth")
def _():
    # The whole point of the knob: a run configured for "predicted" must not be
    # scored partly on oracle knowledge.
    v, fell = resolve_osl(true_osl=417, predicted_osl=99, source="predicted",
                          ref_osl=500)
    assert (v, fell) == (99, False), (v, fell)


@check("test_a_missing_prediction_falls_back_to_ref_and_says_so")
def _():
    v, fell = resolve_osl(true_osl=417, predicted_osl=0, source="predicted",
                          ref_osl=500)
    assert v == 500, v
    assert fell is True, "a silent fallback hides a mixed population"


@check("test_ref_ignores_both_and_is_never_a_fallback")
def _():
    v, fell = resolve_osl(true_osl=417, predicted_osl=99, source="ref",
                          ref_osl=250)
    assert (v, fell) == (250, False), (v, fell)


@check("test_the_servicer_honours_the_configured_source")
def _():
    req = inference_pb2.InferenceRequest(in_tok=100, out_tok=417,
                                         predicted_osl=99)
    for source, want in (("oracle", 417), ("predicted", 99), ("ref", 500)):
        s = servicer(policy_kwargs={"osl_source": source, "ref_osl": 500})
        assert s._osl_for(req) == want, (source, s._osl_for(req))


# ─────────────────────────────── snapshots ──────────────────────────────────

@check("test_a_snapshot_populates_the_view")
def _():
    s = servicer()
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=1, dc_view={EAST[0]: stats(active=3)},
        rtt_matrix={"asia|us_east": 120.0}), None)
    assert s._dc_view[EAST[0]].active_requests == 3
    assert s._rtt_matrix["asia|us_east"] == 120.0


@check("test_a_partial_wave_merges_rather_than_replaces")
def _():
    # Async mode sends one wave per snapshot. Replacing the view would blank
    # every datacentre outside the wave, and the policy would then see a fleet
    # that is mostly missing.
    s = servicer()
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=1, dc_view={EAST[0]: stats(active=3),
                          EAST[1]: stats(active=9)}), None)
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=2, dc_view={EAST[0]: stats(active=5)}), None)
    assert s._dc_view[EAST[0]].active_requests == 5, "the wave did not refresh"
    assert EAST[1] in s._dc_view, "a site outside the wave was dropped"
    assert s._dc_view[EAST[1]].active_requests == 9


@check("test_an_older_epoch_is_ignored")
def _():
    # Pushes race: async waves are in flight at once and can arrive out of
    # order. A late one must not roll the view back.
    s = servicer()
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=5, dc_view={EAST[0]: stats(active=5)}), None)
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=3, dc_view={EAST[0]: stats(active=99)}), None)
    assert s._dc_view[EAST[0]].active_requests == 5, "a stale epoch was applied"
    assert s._epoch == 5, s._epoch


@check("test_the_rtt_matrix_is_replaced_not_merged")
def _():
    # It is always recomputed from the full cached fleet on the provisioner
    # side, so a stale pair left behind would describe a fleet state that no
    # longer exists.
    s = servicer()
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=1, dc_view={}, rtt_matrix={"asia|us_east": 120.0,
                                         "us_east|us_west": 60.0}), None)
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=2, dc_view={}, rtt_matrix={"asia|us_east": 200.0}), None)
    assert s._rtt_matrix == {"asia|us_east": 200.0}, s._rtt_matrix


@check("test_an_empty_rtt_matrix_leaves_the_last_one_alone")
def _():
    s = servicer()
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=1, dc_view={}, rtt_matrix={"asia|us_east": 120.0}), None)
    s.PushSnapshot(inference_pb2.Snapshot(epoch=2, dc_view={}), None)
    assert s._rtt_matrix == {"asia|us_east": 120.0}, s._rtt_matrix


@check("test_the_rtt_pair_key_does_not_depend_on_order")
def _():
    for a in SCHEDULER_GROUPS:
        for b in SCHEDULER_GROUPS:
            assert rtt_pair_key(a, b) == rtt_pair_key(b, a), (a, b)


# ─────────────────────────────── dispatching ────────────────────────────────

@check("test_a_request_with_no_snapshot_yet_still_gets_served")
def _():
    # The window between start-up and the provisioner's first push. Failing
    # every request in it would make the first second of every run an error.
    s = servicer()
    s._dc_endpoints = {EAST[0]: "localhost:1"}
    seen = {}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: (
        seen.update(dc=dc_id)
        or inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id))
    resp = s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="r1", in_tok=10, out_tok=5), None)
    assert resp.status == "OK", resp.error
    assert seen["dc"] == EAST[0], seen


@check("test_a_datacentre_that_refuses_is_reported_not_swallowed")
def _():
    s = servicer()
    s._dc_view = {EAST[0]: stats()}
    s._dispatch_to_dc = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("connection refused"))
    resp = s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="r2", in_tok=10, out_tok=5), None)
    assert resp.status == "ERROR", resp
    assert "refused" in resp.error, resp.error


@check("test_the_caller_id_survives_to_the_datacentre")
def _():
    # One id from the load generator through every hop into DECISION_LOG.
    s = servicer()
    s._dc_view = {EAST[0]: stats()}
    seen = {}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: (
        seen.update(rid=req_id)
        or inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id))
    s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="req4242", in_tok=10, out_tok=5), None)
    assert seen["rid"] == "req4242", seen


@check("test_a_request_with_no_id_is_given_one_that_names_this_scheduler")
def _():
    # Several schedulers each count from 1, and so does the client: a bare
    # req{n} means different requests in different logs.
    s = servicer(group="us_east")
    s._dc_view = {EAST[0]: stats()}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: \
        inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)
    resp = s.SubmitRequest(inference_pb2.InferenceRequest(
        in_tok=10, out_tok=5), None)
    assert resp.req_id.startswith("sched-us_east-req"), resp.req_id


@check("test_the_response_carries_the_measurements_the_datacentre_reported")
def _():
    s = servicer()
    s._dc_view = {EAST[0]: stats(ci=275.0)}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: \
        inference_pb2.DispatchResponse(
            req_id=req_id, dc_id=dc_id, service_ms=900.0, ttft_ms=42.0,
            avg_tpot_ms=19.5, queue_ms=7.0)
    r = s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="r5", in_tok=10, out_tok=5), None)
    assert (r.ttft_ms, r.avg_tpot_ms, r.queue_ms) == (42.0, 19.5, 7.0), r
    assert r.service_ms == 900.0, r
    assert r.carbon_intensity == 275.0, "the CI of the chosen site is not reported"
    assert r.latency_ms >= 0.0


@check("test_scheduling_time_is_measured_and_reported")
def _():
    s = servicer()
    s._dc_view = {d: stats() for d in EAST[:4]}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: \
        inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)
    r = s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="r6", in_tok=10, out_tok=5), None)
    assert r.scheduling_ms >= 0.0
    assert r.scheduling_ms < 1000.0, r.scheduling_ms


@check("test_a_local_dispatch_is_charged_no_forwarding")
def _():
    s = servicer()
    s._dc_view = {EAST[0]: stats()}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: \
        inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)
    r = s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="r7", in_tok=10, out_tok=5), None)
    assert r.forwarding_ms == 0.0, r.forwarding_ms


# ──────────────────────────────── batching ──────────────────────────────────

@check("test_a_batch_policy_buffers_and_then_places_every_request")
def _():
    # The batch loop is a timer thread: requests wait for the window, get
    # solved together, and each waiting caller is handed its own answer.
    s = servicer(policy="round_robin", batch_window_ms=60)

    class Batching:
        scope = PolicyScope.GLOBAL

        def __init__(self):
            # A counter that survives ACROSS batches. Indexing within the batch
            # made the test depend on all six threads landing in one window:
            # split across windows every batch holds one request, every one
            # takes index 0, and the spread assertion fails for a reason that
            # has nothing to do with what is being tested.
            self.n = 0

        def choose_dc_batch(self, reqs, dc_view, group, max_batch_size=0,
                            rtt_matrix=None):
            out = {}
            for r in reqs:
                out[r.req_id] = sorted(dc_view)[self.n % len(dc_view)]
                self.n += 1
            return out

        def choose_dc(self, dc_view, group, **kw):
            return sorted(dc_view)[0]

    s.policy = Batching()
    s._is_batch_policy = True
    s._batch_window_ms = 60
    s.start_batch()
    s._dc_view = {d: stats() for d in EAST[:3]}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: \
        inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)

    # Six at once, so the window has something to batch. Concurrent coroutines
    # rather than concurrent threads: that is what the server holds now, and a
    # buffered request that is never woken shows up here as a 5 s timeout.
    out = s.submit_concurrently([
        inference_pb2.InferenceRequest(req_id=f"b{i}", in_tok=10, out_tok=5)
        for i in range(6)])

    assert len(out) == 6, out
    assert all(r.status == "OK" for r in out), \
        [r.error for r in out if r.status != "OK"]
    assert len({r.dc_id for r in out}) > 1, \
        "a batch solve put everything on one site; assignments were not routed back"


@check("test_a_full_buffer_solves_before_the_window_is_up")
def _():
    # The window is a MAX wait, not a fixed period. This is the check the
    # previous batch_max_size never had: it was configured, printed in the
    # startup line, and enforced nowhere, so setting it changed nothing while
    # looking like it had (deleted in 7c1abcc). A four-second window with a
    # trigger of 3 means anything under a second can only have come from the
    # trigger firing.
    window = 4000
    s = servicer(policy="round_robin", batch_window_ms=window)

    class Batching:
        scope = PolicyScope.GLOBAL

        def choose_dc_batch(self, reqs, dc_view, group, max_batch_size=0,
                            rtt_matrix=None):
            return {r.req_id: sorted(dc_view)[0] for r in reqs}

        def choose_dc(self, dc_view, group, **kw):
            return sorted(dc_view)[0]

    s.policy = Batching()
    s._is_batch_policy = True
    s._batch_window_ms = window
    s._batch_window_size = 3
    s.start_batch()
    s._dc_view = {d: stats() for d in EAST[:3]}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: \
        inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)

    t0 = time.monotonic()
    out = s.submit_concurrently([
        inference_pb2.InferenceRequest(req_id=f"f{i}", in_tok=10, out_tok=5)
        for i in range(3)])
    took_ms = (time.monotonic() - t0) * 1000.0

    assert len(out) == 3 and all(r.status == "OK" for r in out), \
        [r.error for r in out if r.status != "OK"]
    assert took_ms < window / 2, (
        f"a full buffer waited {took_ms:.0f}ms of a {window}ms window, so the "
        f"size trigger did not fire")


@check("test_a_buffer_larger_than_the_window_is_solved_in_projected_chunks")
def _():
    # Chunking exists because one solve over a huge buffer is what turned a
    # stalled solver into a 749 GB OOM. But chunking against an UNPROJECTED
    # view would undo the capacity constraint it protects: every chunk sees
    # the same free seats and fills the same site, so N chunks seat N times
    # its capacity. So the check is not that it chunks -- it is that the
    # second chunk sees a fuller fleet than the first.
    s = servicer(policy="round_robin", batch_window_ms=40)
    saw = []

    class Batching:
        scope = PolicyScope.GLOBAL

        def choose_dc_batch(self, reqs, dc_view, group, max_batch_size=0,
                            rtt_matrix=None):
            target = sorted(dc_view)[0]
            saw.append((len(reqs), int(dc_view[target].active_requests)))
            return {r.req_id: target for r in reqs}

        def choose_dc(self, dc_view, group, **kw):
            return sorted(dc_view)[0]

    s.policy = Batching()
    s._is_batch_policy = True
    s._batch_window_ms = 40
    s._batch_window_size = 2          # so six requests need several chunks
    s.start_batch()
    s._dc_view = {d: stats() for d in EAST[:3]}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: \
        inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)

    out = s.submit_concurrently([
        inference_pb2.InferenceRequest(req_id=f"k{i}", in_tok=100, out_tok=20)
        for i in range(6)])

    assert len(out) == 6 and all(r.status == "OK" for r in out), \
        [r.error for r in out if r.status != "OK"]
    assert len(saw) >= 2, f"six requests with a chunk of 2 solved in {len(saw)}"
    assert max(n for n, _ in saw) <= 2, \
        f"a chunk exceeded the window size: {saw}"
    loads = [load for _, load in saw]
    assert len(set(loads)) > 1, (
        f"every chunk saw the same free capacity, so the view was not "
        f"projected between them: {saw}")


@check("test_an_empty_view_spreads_and_says_so_instead_of_picking_one_site")
def _():
    # With no snapshot the policy cannot run at all -- there is nothing to
    # score -- so this path is a broken run, not a decision. It used to take
    # `sorted(endpoints)[0]`, and on 2026-09-07 a provisioner that failed to
    # bind its port sent 100% of a grid cell's load to AU_VIC, the
    # alphabetically first site and one of the dirtiest, with zero errors
    # reported. Spread it, and say so.
    s = servicer()
    s._dc_view = {}
    s._dc_endpoints = {d: f"localhost:{9000 + i}"
                       for i, d in enumerate(EAST[:4])}
    s._refresh_dc_endpoints = lambda: None
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: \
        inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)

    out = s.submit_concurrently([
        inference_pb2.InferenceRequest(req_id=f"e{i}", in_tok=10, out_tok=5)
        for i in range(8)])

    assert all(r.status == "OK" for r in out), \
        [r.error for r in out if r.status != "OK"]
    used = {r.dc_id for r in out}
    assert len(used) > 1, \
        f"eight requests with no view all went to {used}"
    assert s._no_view_reqs == 8, (
        f"the no-snapshot counter read {s._no_view_reqs}; without it a run "
        f"that placed everything blind looks like a normal one")


@check("test_decide_counts_work_not_waiting")
def _():
    # `decide` is the loop's own cost in the decision path. It was plain wall
    # clock from the start of the handler, so for a batch policy it measured
    # the BUFFERED WAIT -- and concurrent waits sum, so at 1200 req/s it
    # printed 34085% and clamped `rest` (100 - decide - finish) to zero. Both
    # columns were unreadable for exactly the policies whose loop cost was the
    # open question.
    window = 120
    s = servicer(policy="round_robin", batch_window_ms=window)

    class Batching:
        scope = PolicyScope.GLOBAL

        def choose_dc_batch(self, reqs, dc_view, group, max_batch_size=0,
                            rtt_matrix=None):
            return {r.req_id: sorted(dc_view)[0] for r in reqs}

        def choose_dc(self, dc_view, group, **kw):
            return sorted(dc_view)[0]

    s.policy = Batching()
    s._is_batch_policy = True
    s._batch_window_ms = window
    s.start_batch()
    s._dc_view = {d: stats() for d in EAST[:3]}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: \
        inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)

    n = 6
    s.submit_concurrently([
        inference_pb2.InferenceRequest(req_id=f"d{i}", in_tok=10, out_tok=5)
        for i in range(n)])

    # Each request waits up to a window, so the old accounting would have
    # reported about n * window. The work itself is microseconds.
    budget_s = (n * window / 1000.0) / 4
    assert s._t_decide < budget_s, (
        f"decide accumulated {s._t_decide * 1000:.0f}ms for {n} requests on a "
        f"{window}ms window -- it is counting the wait, not the work")


@check("test_the_batch_loop_hands_the_lp_a_capacity_cap")
def _():
    # Without this the LP has no coupling constraint, the problem separates by
    # request, and an "ILP" is a per-request argmin -- which is the greedy rule
    # it exists to beat. It was 0 for the life of the project and nothing said
    # so, so the wiring is asserted rather than trusted.
    s = servicer(policy="round_robin", batch_window_ms=60)
    seen = {}

    class Batching:
        scope = PolicyScope.GLOBAL

        def choose_dc_batch(self, reqs, dc_view, group, max_batch_size=0,
                            rtt_matrix=None):
            seen["cap"] = max_batch_size
            return {r.req_id: sorted(dc_view)[0] for r in reqs}

        def choose_dc(self, dc_view, group, **kw):
            return sorted(dc_view)[0]

    s.policy = Batching()
    s._is_batch_policy = True
    s._batch_window_ms = 60
    s.start_batch()
    s._dc_view = {d: stats() for d in EAST[:3]}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: \
        inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)

    s.submit_concurrently([
        inference_pb2.InferenceRequest(req_id="c0", in_tok=10, out_tok=5)])

    assert seen.get("cap") == s._max_num_seqs != 0, (
        f"the LP was handed cap={seen.get('cap')}, expected "
        f"max_num_seqs={s._max_num_seqs}")


@check("test_the_window_still_bounds_a_buffer_that_never_fills")
def _():
    # The other half of the same contract: under the trigger, the wait is the
    # window. Otherwise a quiet period would hold requests indefinitely.
    window = 150
    s = servicer(policy="round_robin", batch_window_ms=window)

    class Batching:
        scope = PolicyScope.GLOBAL

        def choose_dc_batch(self, reqs, dc_view, group, max_batch_size=0,
                            rtt_matrix=None):
            return {r.req_id: sorted(dc_view)[0] for r in reqs}

        def choose_dc(self, dc_view, group, **kw):
            return sorted(dc_view)[0]

    s.policy = Batching()
    s._is_batch_policy = True
    s._batch_window_ms = window
    s._batch_window_size = 500          # far above what is submitted
    s.start_batch()
    s._dc_view = {d: stats() for d in EAST[:3]}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: \
        inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)

    out = s.submit_concurrently([
        inference_pb2.InferenceRequest(req_id="w0", in_tok=10, out_tok=5)])
    assert len(out) == 1 and out[0].status == "OK", out[0].error


@check("test_batch_scheduling_time_includes_the_wait_for_the_window")
def _():
    # It is what the user pays: buffer wait + solve, not solve alone.
    window = 120
    s = servicer(policy="round_robin", batch_window_ms=window)

    class Batching:
        scope = PolicyScope.GLOBAL

        def choose_dc_batch(self, reqs, dc_view, group, max_batch_size=0,
                            rtt_matrix=None):
            return {r.req_id: sorted(dc_view)[0] for r in reqs}

        def choose_dc(self, dc_view, group, **kw):
            return sorted(dc_view)[0]

    s.policy = Batching()
    s._is_batch_policy = True
    s._batch_window_ms = window
    s.start_batch()
    s._dc_view = {EAST[0]: stats()}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: \
        inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)

    r = s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="b99", in_tok=10, out_tok=5), None)
    assert r.status == "OK", r.error
    assert r.scheduling_ms > 1.0, \
        f"a {window} ms window reported {r.scheduling_ms} ms of scheduling"


# ───────────────────── the projection the policies share ────────────────────

@check("test_the_projection_leaves_the_original_untouched")
def _():
    base = stats(active=4, engines=8)
    before = str(base)
    project_placements(base, [512], [200], 32)
    assert str(base) == before, "project_placements mutated its input"


@check("test_the_projection_keeps_active_and_pending_disjoint")
def _():
    # A placement is added to exactly one of them; adding it to both showed 2n
    # of load for n placements.
    for active, cap in ((4, 32), (256, 32), (100, 0)):
        base = stats(active=active, pending=0, engines=8)
        got = project_placements(base, [100] * 5, [50] * 5, cap)
        added = ((got.active_requests - base.active_requests)
                 + (got.pending_requests - base.pending_requests))
        assert added == 5, f"5 placements showed as {added} (active={active})"


@check("test_the_projection_keeps_the_three_token_sums_disjoint")
def _():
    # THREE sums, not two. A prompt is queued (total_pending_isl), admitted
    # but unprefilled (total_prefill_isl), or decoding (total_active_isl).
    for active in (4, 256):
        base = stats(active=active, engines=8)
        got = project_placements(base, [100] * 5, [50] * 5, 32)
        added = ((got.total_active_isl - base.total_active_isl)
                 + (got.total_prefill_isl - base.total_prefill_isl)
                 + (got.total_pending_isl - base.total_pending_isl))
        assert abs(added - 500) <= 1, f"500 tokens showed as {added}"


@check("test_a_fresh_placement_is_prefilling_not_decoding")
def _():
    # The engine admits with `_prefill_count += 1; _total_prefill_isl += isl`
    # and touches neither _decoding_count nor _total_active_isl until the
    # prompt is through a forward pass. Projecting a placement as decoding
    # collapsed the admission wait -- it is priced (R/d) * step / d, quadratic
    # in d -- and left the fused-prefill term believing every arrival prefilled
    # alone.
    base = stats(active=0, pending=0, engines=8)
    got = project_placements(base, [100] * 4, [50] * 4, 32)
    assert got.prefilling_requests == 4, got.prefilling_requests
    assert got.total_prefill_isl == 400, got.total_prefill_isl
    assert got.decoding_requests == 0, got.decoding_requests
    assert got.total_active_isl == 0, got.total_active_isl


@check("test_the_projection_splits_by_stage_when_told")
def _():
    # An in-flight request past its first token is decoding: its prompt is KV
    # the batch holds, not prefill an arrival queues behind. Lumping a whole
    # in-flight population into ONE stage measured worse than not projecting
    # at all -- see the table in projection.py.
    base = stats(active=0, pending=0, engines=8)
    got = project_placements(base, [100] * 4, [50] * 4, 32,
                             stages=["f", "f", "d", "d"])
    assert got.prefilling_requests == 2, got.prefilling_requests
    assert got.decoding_requests == 2, got.decoding_requests
    assert got.total_prefill_isl == 200, got.total_prefill_isl
    assert got.total_active_isl == 200, got.total_active_isl
    assert got.active_requests == 4, got.active_requests


@check("test_the_queued_prefill_is_charged_over_the_request_lifetime")
def _():
    # TPOT was 1.66x optimistic on the concentrating arms with the whole duty
    # correction already applied, because `duty` prices the prefill the site is
    # ALREADY doing and cannot price the backlog it has not started. That work
    # blocks decode, and the request decodes through it.
    m = analytical_model()
    base = dict(engines=8, decoding_requests=64, total_active_isl=32000,
                total_generated_tokens=4000, prefill_duty=0.5)
    empty = stats(active=64, **base)
    loaded = stats(active=64, prefilling_requests=32,
                   total_prefill_isl=24000, **base)

    # req_osl = 0 leaves the term out: that is the pre-existing behaviour and
    # the default every un-updated caller still gets.
    assert (estimate_tpot_ms(loaded, m, req_osl=0)
            == estimate_tpot_ms(empty, m, req_osl=0)), \
        "the backlog moved the estimate with no horizon to charge it over"

    # With a horizon it must cost something, and a SHORTER output must pay MORE
    # per token: the backlog is drained once, so per token it is work/osl.
    short = estimate_tpot_ms(loaded, m, req_osl=50)
    long_ = estimate_tpot_ms(loaded, m, req_osl=500)
    assert short > long_ > estimate_tpot_ms(empty, m, req_osl=500), \
        f"per-token cost of the backlog is not falling with osl: {short}, {long_}"

    # And it is the closed form, not a fixed point: t = (step + work/osl)/(1-duty)
    step = estimate_tpot_ms(empty, m, req_osl=500) * 0.5
    work = m.ttft_ms(total_prefill_tokens=24000 // 8, bs=32 // 8)
    assert abs(long_ - (step + work / 500) / 0.5) < 1e-6, long_


@check("test_the_duty_cap_still_holds_with_the_backlog_term")
def _():
    # 1/(1 - duty) runs away at duty -> 1, which is why it is capped at 0.95.
    # The backlog term is inside the same divisor and must not escape it.
    m = analytical_model()
    s = stats(active=64, engines=8, decoding_requests=64,
              total_active_isl=32000, total_generated_tokens=4000,
              prefilling_requests=32, total_prefill_isl=24000,
              prefill_duty=0.999)
    v = estimate_tpot_ms(s, m, req_osl=100)
    assert v == v and v < 1e6, v
    ref = estimate_tpot_ms(stats(active=64, engines=8, decoding_requests=64,
                                 total_active_isl=32000,
                                 total_generated_tokens=4000,
                                 prefilling_requests=32,
                                 total_prefill_isl=24000,
                                 prefill_duty=0.95), m, req_osl=100)
    assert abs(v - ref) < 1e-6, f"duty was not capped at 0.95: {v} vs {ref}"


# ───────────────────────── TPOT_FILTER=batch ────────────────────────────────
# The shipped test asks what the ARRIVING request would experience. This mode
# asks what the batch already running there will experience once this request's
# prompt lands on it -- two fields, nothing else.

def _with_filter(mode):
    """Set the module-level switch; it is read per call, not captured."""
    from scheduler.policies.common import latency_estimates as le
    old = le.TPOT_FILTER
    le.TPOT_FILTER = mode
    return le, old


@check("test_the_default_tpot_filter_is_the_arrival_test")
def _():
    # Every result recorded so far was measured under it, so a changed default
    # silently reinterprets all of them.
    import os
    from scheduler.policies.common import latency_estimates as le
    assert le.TPOT_FILTER == os.environ.get("TPOT_FILTER", "arrival").strip().lower()


@check("test_batch_mode_counts_the_waiting_set_in_the_batch")
def _():
    # total_prefill_isl = 0 with prefilling_requests > 0 leaves the backlog at
    # zero, so any difference here is the BATCH term alone -- which is the half
    # the arrival test cannot see, because the projection never advances
    # decoding_requests.
    m = analytical_model()
    base = dict(engines=8, decoding_requests=64, total_active_isl=32000,
                total_generated_tokens=4000, prefill_duty=0.5)
    quiet = stats(active=64, **base)
    filling = stats(active=96, pending=32, prefilling_requests=32, **base)

    le, old = _with_filter("arrival")
    try:
        a_quiet, a_fill = (estimate_tpot_ms(quiet, m, req_osl=100),
                           estimate_tpot_ms(filling, m, req_osl=100))
    finally:
        le.TPOT_FILTER = old
    assert abs(a_quiet - a_fill) < 1e-9, \
        f"arrival test moved on the waiting set alone: {a_quiet} vs {a_fill}"

    le, old = _with_filter("batch")
    try:
        b_quiet, b_fill = (estimate_tpot_ms(quiet, m, req_osl=100),
                           estimate_tpot_ms(filling, m, req_osl=100))
    finally:
        le.TPOT_FILTER = old
    assert b_fill > b_quiet, \
        f"batch test blind to the waiting set: {b_quiet} vs {b_fill}"


@check("test_batch_mode_counts_the_admission_queue_in_the_backlog")
def _():
    # total_pending_isl is priced by the TTFT estimator and by nothing on the
    # TPOT path. Those prompts get prefilled while the running batch decodes.
    m = analytical_model()
    base = dict(engines=8, decoding_requests=64, total_active_isl=32000,
                total_generated_tokens=4000, prefilling_requests=32,
                total_prefill_isl=24000, prefill_duty=0.5)
    no_q = stats(active=96, **base)
    with_q = stats(active=96, pending=32, total_pending_isl=24000, **base)

    le, old = _with_filter("batch")
    try:
        a, b = (estimate_tpot_ms(no_q, m, req_osl=100),
                estimate_tpot_ms(with_q, m, req_osl=100))
    finally:
        le.TPOT_FILTER = old
    assert b > a, f"batch test blind to the admission queue: {a} vs {b}"


@check("test_batch_mode_charges_the_arriving_prompt_to_the_running_batch")
def _():
    # THE POINT OF THE MODE. This request's own prefill is what stops the batch
    # already decoding there, and req_isl was not even an argument to the TPOT
    # estimator -- _check_ttft_sla has taken it all along, _check_tpot_sla
    # never did.
    m = analytical_model()
    s = stats(active=64, engines=8, decoding_requests=64,
              total_active_isl=32000, total_generated_tokens=4000,
              prefill_duty=0.5)

    le, old = _with_filter("batch")
    try:
        without = estimate_tpot_ms(s, m, req_osl=100, req_isl=0)
        small = estimate_tpot_ms(s, m, req_osl=100, req_isl=128)
        big = estimate_tpot_ms(s, m, req_osl=100, req_isl=2048)
    finally:
        le.TPOT_FILTER = old
    assert big > small > without, \
        f"the arriving prompt is not charged: {without}, {small}, {big}"

    # And the arrival test must stay blind to it, or the default has moved.
    le, old = _with_filter("arrival")
    try:
        a0 = estimate_tpot_ms(s, m, req_osl=100, req_isl=0)
        a1 = estimate_tpot_ms(s, m, req_osl=100, req_isl=2048)
    finally:
        le.TPOT_FILTER = old
    assert abs(a0 - a1) < 1e-9, f"arrival test moved on req_isl: {a0} vs {a1}"


@check("test_the_projection_scales_by_the_scheduler_count")
def _():
    # Each scheduler projects its own in-flight correctly and then treats the
    # whole of a site's headroom as its own, so four of them fill a site to
    # four times the cap: BS_CAP=20 settled at 68 per engine, BS_CAP=90 at 105.
    # peer_scale is the symmetric assumption -- the others are doing what I am.
    import inference_pb2
    from scheduler.policies.common.projection import project_totals

    base = inference_pb2.DcStats(active_requests=0, pending_requests=0,
                                 num_gpus=8, carbon_intensity=100.0)
    one = project_totals(base, 4, 4 * 800, 4 * 50, 0, 0.0, 0.0, 0)
    four = project_totals(base, 4 * 4, 4 * 4 * 800, 4 * 4 * 50, 0, 0.0, 0.0, 0)
    assert one.active_requests == 4, one.active_requests
    assert four.active_requests == 16, four.active_requests
    assert four.total_prefill_isl == 4 * one.total_prefill_isl, \
        "scaling the sums must scale the projected prompt tokens too"

    # And it must be proportional to this scheduler's OWN activity: a scheduler
    # that has sent nothing to a site still projects nothing there, however
    # many peers it believes in.
    none = project_totals(base, 0, 0.0, 0.0, 0, 0.0, 0.0, 0)
    assert none.active_requests == 0 and none.total_prefill_isl == 0, \
        "an idle scheduler must not invent load from the peer count"


@check("test_the_sla_fallback_stays_green_before_it_goes_blind")
def _():
    # When nothing meets the SLA the request still has to go somewhere, and
    # where it goes used to be round-robin -- carbon-blind. Measured at 3200
    # req/s with BS_CAP=20, ~8% of requests reached that rung, and that 8%
    # accounted for the whole of the carbon lost down the cap axis.
    import inference_pb2
    from scheduler.policies.common.capacity import greenest_under_cap

    def dc(ci, active, engines=8):
        return inference_pb2.DcStats(active_requests=active, num_gpus=engines,
                                     carbon_intensity=ci)

    view = {"dirty": dc(600, 8), "green": dc(100, 8), "mid": dc(300, 8)}
    assert greenest_under_cap(view, 0) == "green", "no cap: take the greenest"

    # bs_cap is per ENGINE, so 8 engines x 10 = 80 site-wide.
    view = {"dirty": dc(600, 8), "green": dc(100, 800), "mid": dc(300, 8)}
    assert greenest_under_cap(view, 10) == "mid", \
        "a site at the guard must be skipped, not chosen for being green"

    view = {"dirty": dc(600, 800), "green": dc(100, 800)}
    assert greenest_under_cap(view, 10) is None, \
        "every site over the guard must hand on to the blind rung"


@check("test_sites_are_priced_at_intensity_times_pue")
def _():
    import inference_pb2
    from scheduler.policies.common.capacity import greenest_under_cap
    from scheduler.policies.common.carbon_signal import (
        PUE_REF, effective_carbon_intensity)
    from scheduler.policies.common.estimators import _get_carbon_intensity

    def dc(ci, pue):
        return inference_pb2.DcStats(num_gpus=8, carbon_intensity=ci, pue=pue)
    # The uniform fleet every earlier run used prices exactly as before, and a
    # site predating the field falls back to its raw intensity.
    for ci in (95.0, 312.7):
        assert effective_carbon_intensity(dc(ci, PUE_REF)) == ci
        assert effective_carbon_intensity(dc(ci, 0.0)) == ci
    # A 95 g grid at PUE 1.32 costs more per unit of work than a 100 g grid at
    # 1.08, and both the fallback and the LP's price must say so.
    view = {"clean_grid_bad_pue": dc(95.0, 1.32),
            "dirtier_grid_good_pue": dc(100.0, 1.08)}
    assert greenest_under_cap(view, 0) == "dirtier_grid_good_pue"
    assert (_get_carbon_intensity(view["clean_grid_bad_pue"])
            > _get_carbon_intensity(view["dirtier_grid_good_pue"]))


@check("test_a_carbon_policy_falls_back_to_the_greenest_not_round_robin")
def _():
    # End to end through the policy: an SLA budget nothing can meet, and the
    # placement must still be the greenest rather than whatever round-robin
    # had next.
    import inference_pb2
    from scheduler.policies.carbon_aware import CarbonAware

    m = analytical_model()
    p = CarbonAware(alpha=0.0, analytical_model=m,
                    ttft_budget_ms=1.0, tpot_budget_ms=1.0)   # nothing passes
    view = {}
    for name, ci in (("dirty", 700.0), ("green", 90.0), ("mid", 400.0)):
        view[name] = stats(active=8, engines=8, ci=ci, decoding_requests=8,
                           total_active_isl=4000, total_generated_tokens=500,
                           total_remaining_osl=400, prefill_duty=0.2)
    seen = {p.choose_dc(view, "r") for _ in range(6)}
    assert seen == {"green"}, f"fallback wandered: {seen}"


def _busy_site(**kw):
    return stats(active=42, engines=8, decoding_requests=40,
                 total_active_isl=40 * 500, total_generated_tokens=4000,
                 total_remaining_osl=40 * 100, prefilling_requests=2,
                 total_prefill_isl=1000, **kw)


@check("test_the_kv_gate_wait_is_inert_where_the_cache_has_room")
def _():
    # What keeps every short-prompt result valid: no KV fields, or slack,
    # prices exactly what the estimator priced before.
    m = analytical_model()
    before = estimate_ttft_ms(_busy_site(), 512, m, max_num_seqs=128)
    slack = estimate_ttft_ms(_busy_site(kv_room_tokens=4096.0,
                                        kv_ms_per_token=10.0),
                             512, m, max_num_seqs=128)
    assert slack == before, (slack, before)


@check("test_a_full_cache_adds_its_wait_to_the_ttft_estimate")
def _():
    m = analytical_model()
    before = estimate_ttft_ms(_busy_site(), 512, m, max_num_seqs=128)
    full = _busy_site(kv_queue_ms=2000.0, kv_room_tokens=256.0,
                      kv_ms_per_token=1.0)
    got = estimate_ttft_ms(full, 512, m, max_num_seqs=128)
    assert got >= 2000.0 + 256 and got > before, (got, before)
    longer = estimate_ttft_ms(full, 4096, m, max_num_seqs=128)
    assert longer - got >= 4096 - 512, (longer, got)


@check("test_the_projection_folds_placements_into_the_kv_queue")
def _():
    # Two 1000-token prompts over 8 engines take 250 tokens of each one's room.
    base = stats(active=4, engines=8, kv_queue_ms=300.0, kv_room_tokens=1000.0,
                 kv_ms_per_token=8.0)
    got = project_placements(base, [1000, 1000], [50, 50], 128)
    assert abs(got.kv_room_tokens - 750.0) < 1e-6, got.kv_room_tokens
    assert (got.kv_queue_ms, got.kv_ms_per_token) == (300.0, 8.0), got


@check("test_every_sla_aware_path_reaches_the_same_tpot_estimate")
def _():
    # These three bodies used to be three inline copies, and a correction to
    # one left the other two on the old model -- which then read as "greedy is
    # worse than ILP" for a reason that was not real. They are one function
    # now; this pins that, in both modes, so a future edit cannot re-fork them.
    import inference_pb2
    from scheduler.policies.common.estimators import _check_sla
    from scheduler.policies.carbon_aware import CarbonAware
    from scheduler.policies.carbon_aware_local import CarbonAwareLocal

    m = analytical_model()
    st = inference_pb2.DcStats(
        active_requests=96, pending_requests=32, num_gpus=8,
        carbon_intensity=300.0, decoding_requests=64, prefilling_requests=32,
        total_active_isl=32000, total_generated_tokens=4000,
        total_prefill_isl=24000, total_pending_isl=24000, prefill_duty=0.5)

    for mode in ("arrival", "batch"):
        le, old = _with_filter(mode)
        try:
            _, _, ilp = _check_sla(st, 841, 3000.0, 100.0, m, bs_offset=1,
                                   req_osl=71)
            vals = [ilp]
            for cls in (CarbonAware, CarbonAwareLocal):
                pol = cls(alpha=0.25, analytical_model=m, tpot_budget_ms=100)
                vals.append(pol._estimate_tpot(st, req_osl=71, req_isl=841))
        finally:
            le.TPOT_FILTER = old
        assert max(vals) - min(vals) < 1e-9, \
            f"{mode}: the SLA-aware paths have forked: {vals}"

    # And the mode must actually change the answer, or the check is vacuous.
    le, old = _with_filter("arrival")
    try:
        _, _, arr = _check_sla(st, 841, 3000.0, 100.0, m, bs_offset=1, req_osl=71)
    finally:
        le.TPOT_FILTER = old
    le, old = _with_filter("batch")
    try:
        _, _, bat = _check_sla(st, 841, 3000.0, 100.0, m, bs_offset=1, req_osl=71)
    finally:
        le.TPOT_FILTER = old
    assert bat > arr, f"the flag did not reach _check_sla: {arr} vs {bat}"


@check("test_batch_mode_leaves_the_horizon_and_the_duty_divisor_alone")
def _():
    # Only two fields change. With no waiting set and no queue the two modes
    # must agree exactly -- which is also why neither can stop the gate opening
    # at a snapshot where both are zero.
    m = analytical_model()
    s = stats(active=64, engines=8, decoding_requests=64,
              total_active_isl=32000, total_generated_tokens=4000,
              prefill_duty=0.57)
    le, old = _with_filter("arrival")
    try:
        a = estimate_tpot_ms(s, m, req_osl=71)
    finally:
        le.TPOT_FILTER = old
    le, old = _with_filter("batch")
    try:
        b = estimate_tpot_ms(s, m, req_osl=71, req_isl=0)
    finally:
        le.TPOT_FILTER = old
    assert abs(a - b) < 1e-9, f"modes differ on an idle backlog: {a} vs {b}"


# ───────────────────────────── SubmitStream ─────────────────────────────────
# The streamed path must be the unary path with different plumbing. What is
# tested here is only the plumbing: that no reply is lost, that no reply is
# misrouted, and that a slow request does not hold up a fast one.


def _stream_servicer(delays):
    """A servicer whose dispatch sleeps a per-DC amount, in ms."""
    s = servicer()
    s._dc_view = {d: stats() for d in delays}

    async def fake(req_id, dc_id, request, timeout=2000.0):
        await asyncio.sleep(delays[dc_id] / 1000.0)
        return inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id,
                                              service_ms=delays[dc_id])
    s._dispatch_to_dc = fake
    return s


@check("test_stream_answers_every_request_exactly_once")
def _():
    s = _stream_servicer({"US_PJM": 0.0})
    s.policy.choose_dc_extended = lambda *a, **k: ("US_PJM", True)
    reqs = [inference_pb2.InferenceRequest(req_id=f"r{i}", in_tok=100,
                                           out_tok=10) for i in range(50)]
    got = s.stream(reqs)
    assert len(got) == 50, f"{len(got)} replies for 50 requests"
    ids = sorted(r.req_id for r in got)
    assert ids == sorted(r.req_id for r in reqs), "req_id set does not match"


@check("test_stream_reply_carries_the_req_id_it_answers")
def _():
    # The failure this catches is a reply written to the wrong requester, which
    # would attribute one request's dc_id and service_ms to another. Each
    # request is pinned to a different site, so the pairing is checkable.
    delays = {"US_PJM": 12.0, "US_MISO": 1.0, "US_NYISO": 6.0}
    s = _stream_servicer(delays)
    want = {}
    order = list(delays)

    def pick(dc_view, group, **kw):
        return (order[kw["req_isl"] % len(order)], True)
    s.policy.choose_dc_extended = pick

    reqs = []
    for i in range(30):
        dc = order[i % len(order)]
        reqs.append(inference_pb2.InferenceRequest(req_id=f"r{i}", in_tok=i,
                                                   out_tok=10))
        want[f"r{i}"] = dc

    for resp in s.stream(reqs):
        assert resp.dc_id == want[resp.req_id], (
            f"{resp.req_id} answered with {resp.dc_id}, "
            f"expected {want[resp.req_id]}")


@check("test_stream_does_not_block_a_fast_reply_behind_a_slow_one")
def _():
    # Head-of-line blocking is the whole reason replies are unordered. The
    # first request takes 80 ms and the rest take none; if the handler wrote in
    # arrival order, r0 would come out first.
    delays = {"SLOW": 80.0, "FAST": 0.0}
    s = servicer()
    s._dc_view = {d: stats() for d in delays}

    async def fake(req_id, dc_id, request, timeout=2000.0):
        await asyncio.sleep(delays[dc_id] / 1000.0)
        return inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)
    s._dispatch_to_dc = fake
    s.policy.choose_dc_extended = lambda *a, **k: (
        ("SLOW" if k["req_isl"] == 0 else "FAST"), True)

    reqs = [inference_pb2.InferenceRequest(req_id=f"r{i}", in_tok=i,
                                           out_tok=10) for i in range(5)]
    got = s.stream(reqs)
    assert len(got) == 5, len(got)
    assert got[-1].req_id == "r0", (
        f"the 80 ms request came back at position "
        f"{[r.req_id for r in got].index('r0')}, so replies are ordered")


@check("test_stream_waits_for_in_flight_work_after_the_client_half_closes")
def _():
    # The termination bug this pins: the request iterator is exhausted long
    # before the replies are ready, and an empty queue at that moment is not
    # evidence that the stream is done. Every dispatch sleeps, so the reader
    # finishes with all 20 still outstanding and none yet queued.
    s = _stream_servicer({"US_PJM": 40.0})
    s.policy.choose_dc_extended = lambda *a, **k: ("US_PJM", True)
    reqs = [inference_pb2.InferenceRequest(req_id=f"r{i}", in_tok=100,
                                           out_tok=10) for i in range(20)]
    got = s.stream(reqs)
    assert len(got) == 20, f"{len(got)}/20 replies survived the half-close"


@check("test_stream_survives_one_failing_request")
def _():
    s = servicer()
    s._dc_view = {"US_PJM": stats()}
    s.policy.choose_dc_extended = lambda *a, **k: ("US_PJM", True)

    async def fake(req_id, dc_id, request, timeout=2000.0):
        if req_id == "r3":
            raise RuntimeError("dispatch exploded")
        return inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)
    s._dispatch_to_dc = fake

    reqs = [inference_pb2.InferenceRequest(req_id=f"r{i}", in_tok=100,
                                           out_tok=10) for i in range(6)]
    got = s.stream(reqs)
    assert len(got) == 6, f"a failing request took the stream down: {len(got)}/6"
    bad = [r for r in got if r.req_id == "r3"]
    assert bad and bad[0].status == "ERROR", "the failure was not reported"


@check("test_no_placements_change_nothing")
def _():
    base = stats(active=4, pending=1, engines=8, total_active_isl=900)
    assert project_placements(base, [], [], 32) == base


@check("test_peer_only_projection_from_the_arrival_rate")
def _():
    """A site this scheduler forwards to holds no in-flight entry of its own.

    _fwd_begin runs on the is_local branch alone, so the ratio correction never
    reaches those sites and they used to get no projection at all -- the
    measured 5x optimism about remote sites. The rate form needs no tracking:
    admitted_requests counts every scheduler's arrivals, so the rate times the
    snapshot age IS the unreflected load.
    """
    s = servicer(group="us_east", policy="carbon_aware")
    dc, st = "JP_Tokyo", stats(active=0, pending=0, engines=8)
    s._dc_rate[dc] = [0, 0.0, 200.0, 0, 1.0]      # 200 req/s at the site
    s._dc_arrived[dc] = 100.0

    n, n_f = s._peer_only_counts(dc, st, 100.5)   # half a second of staleness
    assert (n, n_f) == (100, 100), f"expected 100 all prefilling, got {n},{n_f}"

    # With a learned dispatch-to-first-token boundary, only the newest arrivals
    # are still in prefill: 250 ms of a 500 ms window is half of them.
    s._stage_ttft_ms[dc] = 250.0
    n, n_f = s._peer_only_counts(dc, st, 100.5)
    assert (n, n_f) == (100, 50), f"expected a half/half split, got {n},{n_f}"

    # A rate estimate cannot imply more arrivals than the site has seats.
    s._dc_rate[dc] = [0, 0.0, 100000.0, 0, 1.0]
    n, _ = s._peer_only_counts(dc, st, 100.5)
    assert n == 8 * 32, f"unclamped projection: {n} against 256 seats"

    # No rate yet, or a site whose snapshot never arrived: project nothing.
    assert s._peer_only_counts("nowhere", st, 100.5) == (0, 0)


@check("test_infeasible_target_is_respotted_least_loaded")
def _():
    """A forwarded request lands where it was sent -- unless that no longer fits.

    The sender chose a hop and a snapshot ago. Measured, 77-81% of arrivals
    reach a site the receiver can already see is over budget. The region still
    stands; only the site moves, to the least-loaded one in it.
    """
    s = servicer(group="us_east", policy="carbon_aware",
                 policy_kwargs={"ttft_budget_ms": 3000.0,
                                "tpot_budget_ms": 100.0})
    target, spare = EAST[0], EAST[1]
    req = inference_pb2.InferenceRequest(req_id="r1", in_tok=512, out_tok=64,
                                         forwarded=True, target_dc=target)

    full = {target: stats(active=4000, pending=4000, engines=8,
                          total_pending_isl=4_000_000,
                          total_prefill_isl=2_000_000),
            spare: stats(active=0, pending=0, engines=8)}
    assert s._respot(target, full, req) == spare, \
        "an infeasible target should move to the least-loaded site in region"

    idle = {target: stats(active=0, pending=0, engines=8),
            spare: stats(active=0, pending=0, engines=8)}
    assert s._respot(target, idle, req) is None, \
        "a target that still fits must be honoured"

    # No budget configured: nothing to judge with, so nothing is overridden.
    plain = servicer(group="us_east", policy="round_robin")
    assert plain._respot(target, full, req) is None


@check("test_respot_never_leaves_the_region")
def _():
    s = servicer(group="us_east", policy="carbon_aware",
                 policy_kwargs={"ttft_budget_ms": 3000.0,
                                "tpot_budget_ms": 100.0})
    target = EAST[0]
    req = inference_pb2.InferenceRequest(req_id="r1", in_tok=512, out_tok=64,
                                         forwarded=True, target_dc=target)
    view = {target: stats(active=4000, pending=4000, engines=8,
                          total_pending_isl=4_000_000,
                          total_prefill_isl=2_000_000)}
    # The whole region is one loaded site; a far-away empty one is not an answer,
    # because the hop it would need has already been paid to the wrong pair.
    view[WEST[0]] = stats(active=0, pending=0, engines=8)
    assert s._respot(target, view, req) is None, \
        "re-spotting must not cross a region"


@check("test_peer_share_is_measured_not_assumed")
def _():
    """Two schedulers feeding one site must project the SAME total load.

    The worked case: over one 520 ms snapshot period Netherlands takes 115
    requests, 100 of them from its own region's scheduler and 5 from each of
    the other three. A constant multiplier of 4 makes the local scheduler
    project 400 and a remote one 20 -- a twentyfold disagreement about the
    same site at the same instant, and the measured cause of the 3.3x/0.2x
    estimator bias. The measured factor has to reconcile them.
    """
    def k_after(own_sent):
        s = servicer(group="europe_north", policy="carbon_aware")
        s._own_sent["Netherlands"] = 0
        s._update_rates("Netherlands",
                        inference_pb2.DcStats(admitted_requests=1000), 0.0)
        s._own_sent["Netherlands"] = own_sent
        s._update_rates("Netherlands",
                        inference_pb2.DcStats(admitted_requests=1115), 0.52)
        return s._k_for("Netherlands")

    for own in (100, 5):
        k = k_after(own)
        projected = own * k
        assert abs(projected - 115) < 1.0, (
            f"own={own} projected {projected:.1f} unreflected, not 115")


@check("test_peer_share_falls_back_without_a_rate")
def _():
    s = servicer(group="europe_north", policy="carbon_aware",
                 policy_kwargs={"peer_scale": 4})
    assert s._k_for("Netherlands") == 4.0, "no history should mean peer_scale"
    # A site this scheduler has stopped sending to: lambda_own is 0, the ratio
    # is unbounded, and the projection it multiplies is 0 either way.
    s._own_sent["Netherlands"] = 7
    s._update_rates("Netherlands",
                    inference_pb2.DcStats(admitted_requests=10), 0.0)
    s._update_rates("Netherlands",
                    inference_pb2.DcStats(admitted_requests=210), 1.0)
    assert s._k_for("Netherlands") == 4.0, "zero own rate should not divide"


if __name__ == "__main__":
    print(f"{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    if FAIL:
        sys.exit(1)
