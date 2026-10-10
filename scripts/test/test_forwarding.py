#!/usr/bin/env python3
"""What happens to a request that crosses a region boundary.

Every cross-region placement now goes through the scheduler that owns the
target group, and that scheduler dispatches where it was told rather than
choosing again. These check the three things that has to be true:

  the choice survives the hop        target_dc is honoured, not re-decided
  it cannot name someone else's site a third group is refused, not dispatched
  the receiver knows what landed     forwarded-in load appears in its own view
                                     until a snapshot that includes it arrives

Run:  PYTHONPATH=.:proto python3 scripts/test/test_forwarding.py
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "proto"))
os.environ.setdefault("STATIC_ENDPOINTS_FILE", "/dev/null")

import inference_pb2                                            # noqa: E402
from scheduler.server import SchedulerServicer
from async_shim import SyncServicer                  # noqa: E402
from common.network import rtt_pair_key                         # noqa: E402
from scheduler.policies.common.projection import project_placements  # noqa: E402

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
        except Exception as e:                                  # noqa: BLE001
            FAIL.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
        return fn
    return deco


def servicer(group="us_east", policy="round_robin", max_num_seqs=32):
    return SyncServicer(SchedulerServicer(
        f"sched-{group}", group, policy, 0.0,
        simulate_rtt=False, verbosity=0,
        policy_kwargs={"max_num_seqs": max_num_seqs}))


def stats(active=0, pending=0, engines=8, ci=300.0, **kw):
    s = inference_pb2.DcStats(active_requests=active, pending_requests=pending,
                              num_gpus=engines, carbon_intensity=ci)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# us_east and us_west both exist in the topology; these are real members.
US_EAST_DC = "US_PJM"
US_WEST_DC = "US_CAISO"


# ─────────────────────────── honouring the choice ───────────────────────────

@check("test_target_dc_is_dispatched_to_without_re_deciding")
def _():
    s = servicer()
    s._dc_view = {d: stats() for d in ("US_PJM", "US_MISO", "US_NYISO")}
    seen = {}

    def fake(req_id, dc_id, request, timeout=2000.0):
        seen["dc"] = dc_id
        return inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id,
                                              service_ms=1.0)
    s._dispatch_to_dc = fake
    # If the local policy is consulted at all, the dispatch goes here instead
    # of to the named site. Asserting only "it went to US_MISO" is not enough:
    # round_robin can pick US_MISO by luck, and then a scheduler that ignores
    # target_dc entirely still passes.
    s.policy.choose_dc_local_only = lambda *a, **k: "RE_DECIDED"

    resp = s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="r1", in_tok=100, out_tok=50, forwarded=True,
        target_dc="US_MISO"), None)
    assert resp.status == "OK", resp.error
    assert seen["dc"] != "RE_DECIDED", "the receiver decided again"
    assert seen["dc"] == "US_MISO", f"dispatched to {seen['dc']}, not US_MISO"
    assert resp.dc_id == "US_MISO", resp.dc_id


@check("test_a_site_in_another_group_is_refused")
def _():
    s = servicer(group="us_east")
    s._dc_view = {US_EAST_DC: stats()}
    s._dispatch_to_dc = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("dispatched a request that names another group"))

    resp = s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="r2", in_tok=100, out_tok=50, forwarded=True,
        target_dc=US_WEST_DC), None)
    assert resp.status == "ERROR", "a third group's site was accepted"
    assert US_WEST_DC in resp.error, resp.error


@check("test_no_target_dc_still_decides_locally")
def _():
    # A caller that predates the field must keep working.
    s = servicer()
    s._dc_view = {d: stats() for d in ("US_PJM", "US_MISO")}
    seen = {}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: (
        seen.update(dc=dc_id)
        or inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id))

    resp = s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="r3", in_tok=100, out_tok=50, forwarded=True), None)
    assert resp.status == "OK", resp.error
    assert seen["dc"] in ("US_PJM", "US_MISO"), seen


@check("test_a_group_with_no_scheduler_falls_back_to_a_direct_dispatch")
def _():
    # A deliberately partial fleet: another group's datacentres are running and
    # the provisioner polls them, so a GLOBAL policy can pick one, but no
    # scheduler owns them. That placement succeeded before this change and must
    # keep succeeding -- and it must still pay the round trip, once out and
    # once back, or forwarding looks free in the one configuration where it is
    # dispatched directly.
    #
    # The matrix value IS the round trip (cloudping.co p50, which is what that
    # site measures), so 120 is the whole cost and each leg is 60. This used to
    # assert 240, treating the table as one-way and charging it twice -- twice
    # the real network on 87% of requests in a concentrating run.
    s = servicer(group="us_east")
    s._dc_view = {US_WEST_DC: stats()}          # only a remote site to choose
    s._rtt_matrix = {rtt_pair_key("us_east", "us_west"): 120.0}
    s.simulate_rtt = True
    s._find_sched_for_dc = lambda dc_id: None   # nobody owns us_west
    seen = {}
    s._dispatch_to_dc = lambda req_id, dc_id, request, timeout=2000.0: (
        seen.update(dc=dc_id)
        or inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id))

    t0 = time.monotonic()
    resp = s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="r20", in_tok=100, out_tok=50), None)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    assert resp.status == "OK", resp.error
    assert seen["dc"] == US_WEST_DC, seen
    assert abs(resp.forwarding_ms - 120.0) < 1e-6, \
        f"charged {resp.forwarding_ms} ms, not the matrix's own 120"
    assert elapsed_ms >= 115.0, \
        f"only slept {elapsed_ms:.0f} ms: one leg of the round trip is missing"
    assert elapsed_ms < 200.0, \
        f"slept {elapsed_ms:.0f} ms: the round trip is being charged twice"


# ──────────────────────────────── the delta ─────────────────────────────────

@check("test_forwarded_in_load_shows_up_in_the_local_view")
def _():
    s = servicer()
    base = stats(active=4, pending=0, engines=8)
    s._dc_view = {US_EAST_DC: base}

    before = s._view_with_forwarded(dict(s._dc_view))[US_EAST_DC]
    assert before.active_requests == 4, before.active_requests

    s._fwd_begin("r10", US_EAST_DC,
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=200))
    after = s._view_with_forwarded(dict(s._dc_view))[US_EAST_DC]
    assert after.active_requests == 5, \
        f"the forwarded request is invisible: {after.active_requests}"
    # Prefilling, not decoding: it was dispatched a moment ago and its prompt
    # has not been through a forward pass.
    assert after.total_prefill_isl == 512, after.total_prefill_isl
    assert after.total_active_isl == 0, after.total_active_isl
    # and the snapshot itself is untouched
    assert s._dc_view[US_EAST_DC].active_requests == 4, "the base view was mutated"


@check("test_a_dispatch_the_snapshot_already_counts_is_not_projected_twice")
def _():
    # The correction may only add what the datacentre's counters cannot show.
    # A request dispatched BEFORE the snapshot arrived is already in those
    # counters, so projecting it again doubles it. Untested until now, and the
    # filter is the kind that fails silently: drop everything and the whole
    # correction quietly becomes a no-op while every test still passes.
    # Driven through PushSnapshot rather than by setting _dc_arrived, because
    # the retirement IS the snapshot's pruning: it drops what it already counts
    # and the projection then works from what is left. Poking _dc_arrived
    # tested a second, redundant filter that no longer exists.
    s = servicer()
    s._dc_view = {US_EAST_DC: stats(active=4, pending=0, engines=8)}

    # dispatched BEFORE the snapshot lands -> the poll's counters include it
    s._fwd_begin("old", US_EAST_DC,
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=200))
    time.sleep(0.002)
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=1, dc_view={US_EAST_DC: stats(active=4, pending=0, engines=8)}),
        None)
    v = s._view_with_forwarded(dict(s._dc_view))[US_EAST_DC]
    assert v.active_requests == 4, \
        f"a request the snapshot already counts was projected again: {v.active_requests}"
    assert v.total_prefill_isl == 0, v.total_prefill_isl

    # dispatched AFTER it -> invisible to those counters, so it must be added
    s._fwd_begin("new", US_EAST_DC,
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=200))
    v = s._view_with_forwarded(dict(s._dc_view))[US_EAST_DC]
    assert v.active_requests == 5, \
        f"a request the snapshot cannot see was dropped: {v.active_requests}"
    assert v.total_prefill_isl == 512, v.total_prefill_isl


@check("test_the_projection_size_reaches_the_decision_log")
def _():
    # "Did the projection fire?" has to be answerable from a run's log alone.
    # Without it a null result is ambiguous between "the correction did not
    # help" and "the correction never ran" -- which is exactly the ambiguity
    # probe_rate4 landed in.
    import time as _t
    from scheduler.server import _proj_note
    s = servicer()
    s._dc_view = {US_EAST_DC: stats(active=4, engines=8)}
    s._dc_arrived[US_EAST_DC] = _t.monotonic() - 1.0
    s._fwd_begin("a", US_EAST_DC,
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=200))
    s._fwd_begin("b", US_EAST_DC,
                 inference_pb2.InferenceRequest(in_tok=256, out_tok=100))
    s._view_with_forwarded(dict(s._dc_view))
    n, nf, _b = _proj_note.by_dc.get(US_EAST_DC, (0, 0, 0.0))
    assert n == 2, f"projection size not recorded: {n}"
    assert nf == 2, f"both were just dispatched, so both are prefilling: {nf}"


@check("test_the_delta_uses_the_same_rule_as_the_ilp")
def _():
    s = servicer(max_num_seqs=32)
    base = stats(active=4, engines=8)
    s._dc_view = {US_EAST_DC: base}
    s._fwd_begin("r11", US_EAST_DC,
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=200))
    mine = s._view_with_forwarded(dict(s._dc_view))[US_EAST_DC]
    theirs = project_placements(base, [512], [200], 32)
    assert mine == theirs, "the delta and project_placements disagree"


@check("test_a_full_site_queues_the_forwarded_request")
def _():
    # 8 engines x 32 = 256 slots, all taken: the placement is pending, not
    # active, or whoever reads this next prices its TTFT against no queue.
    s = servicer(max_num_seqs=32)
    s._dc_view = {US_EAST_DC: stats(active=256, pending=0, engines=8)}
    s._fwd_begin("r12", US_EAST_DC,
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=200))
    v = s._view_with_forwarded(dict(s._dc_view))[US_EAST_DC]
    assert v.active_requests == 256, v.active_requests
    assert v.pending_requests == 1, v.pending_requests
    assert v.total_pending_isl == 512, v.total_pending_isl
    assert v.total_active_isl == 0, v.total_active_isl


@check("test_completion_forgets_it")
def _():
    s = servicer()
    s._dc_view = {US_EAST_DC: stats(active=4)}
    s._fwd_begin("r13", US_EAST_DC,
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=200))
    s._fwd_end("r13", US_EAST_DC)
    v = s._view_with_forwarded(dict(s._dc_view))[US_EAST_DC]
    assert v.active_requests == 4, v.active_requests
    assert not s._fwd_inflight, s._fwd_inflight


@check("test_a_snapshot_that_includes_the_site_retires_the_delta")
def _():
    s = servicer()
    s._dc_view = {US_EAST_DC: stats(active=4)}
    s._fwd_begin("r14", US_EAST_DC,
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=200))
    time.sleep(0.002)   # dispatched strictly before the snapshot arrives
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=1, dc_view={US_EAST_DC: stats(active=5)}), None)
    v = s._view_with_forwarded(dict(s._dc_view))[US_EAST_DC]
    assert v.active_requests == 5, \
        f"double counted: the poll already saw it, got {v.active_requests}"


@check("test_a_partial_wave_leaves_other_sites_corrected")
def _():
    # The async case. A snapshot carrying one wave must not retire the
    # corrections for datacentres it did not refresh -- their counters are
    # still pre-dispatch, and dropping the delta under-counts exactly them.
    s = servicer()
    s._dc_view = {"US_PJM": stats(active=4), "US_MISO": stats(active=4)}
    s._fwd_begin("r15", "US_PJM",
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=200))
    s._fwd_begin("r16", "US_MISO",
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=200))
    time.sleep(0.002)
    # only US_PJM is in this wave
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=1, dc_view={"US_PJM": stats(active=5)}), None)
    v = s._view_with_forwarded(dict(s._dc_view))
    assert v["US_PJM"].active_requests == 5, \
        f"refreshed site double counted: {v['US_PJM'].active_requests}"
    assert v["US_MISO"].active_requests == 5, \
        (f"a site absent from the wave lost its correction: "
         f"{v['US_MISO'].active_requests}")


@check("test_a_request_dispatched_after_the_snapshot_survives_it")
def _():
    s = servicer()
    s._dc_view = {US_EAST_DC: stats(active=4)}
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=1, dc_view={US_EAST_DC: stats(active=4)}), None)
    time.sleep(0.002)
    s._fwd_begin("r17", US_EAST_DC,       # arrived after the poll
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=200))
    v = s._view_with_forwarded(dict(s._dc_view))[US_EAST_DC]
    assert v.active_requests == 5, \
        f"the snapshot cannot know about it, got {v.active_requests}"


@check("test_submitrequest_tracks_a_forwarded_request_while_it_runs")
def _():
    # The wiring, not the bookkeeping: _fwd_begin/_fwd_end are only useful if
    # the dispatch path actually calls them. A mechanism that never fires is
    # worse than none -- it reads as covered.
    s = servicer()
    s._dc_view = {US_EAST_DC: stats(active=4)}
    during = {}

    def fake(req_id, dc_id, request, timeout=2000.0):
        with s._fwd_lock:
            during[req_id] = {d: dict(v) for d, v in s._fwd_inflight.items()}
        return inference_pb2.DispatchResponse(req_id=req_id, dc_id=dc_id)
    s._dispatch_to_dc = fake

    s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="fwd", in_tok=512, out_tok=200, forwarded=True,
        target_dc=US_EAST_DC), None)
    assert during["fwd"].get(US_EAST_DC, {}).get("fwd"), \
        "a forwarded request was not tracked while it ran"
    assert not s._fwd_inflight, f"not released on return: {s._fwd_inflight}"

    # A request this scheduler placed ITSELF is tracked too. It used to be
    # skipped, on the reasoning that the policy already knew about its own
    # placements -- no policy here does: they all read self._dc_view, a
    # snapshot from the last poll, and carry no memory of what they dispatched
    # since. That gap was the whole of the TTFT estimator's error.
    s.SubmitRequest(inference_pb2.InferenceRequest(
        req_id="own", in_tok=512, out_tok=200), None)
    assert during["own"].get(US_EAST_DC, {}).get("own"), \
        "a local placement was not tracked while it ran"
    assert not s._fwd_inflight, f"not released on return: {s._fwd_inflight}"


@check("test_the_delta_never_leaves_the_scheduler")
def _():
    # It is a local correction. Nothing may publish it: the provisioner polls
    # datacentres, and a scheduler that exported its own guesses would feed
    # them back to every other scheduler as though they were measurements.
    s = servicer()
    s._fwd_begin("r18", US_EAST_DC,
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=200))
    src = (ROOT / "scheduler" / "server.py").read_text()
    for sink in ("PushSnapshot(", "Snapshot(dc_view", "PollStats"):
        for line in src.splitlines():
            if "_fwd_inflight" in line and sink.rstrip("(") in line:
                raise AssertionError(f"the delta reaches {sink}: {line.strip()}")


@check("test_the_maintained_sums_match_a_recount_at_every_step")
def _():
    # The six sums are now kept incrementally instead of recomputed, so the
    # failure mode is drift: a subtraction that hits the wrong bucket, an entry
    # aged twice, a snapshot prune that forgets one. None of that shows up as
    # an exception or a wrong request count -- it shows up as a policy quietly
    # deciding against a view that is off by a few hundred tokens.
    #
    # So recount from _fwd_inflight, which is the ground truth, after every
    # kind of mutation.
    s = servicer()
    dcs = [US_EAST_DC, "US_PJM"]
    s._dc_view = {d: stats(active=0) for d in dcs}

    def recount(dc_id):
        at = s._fwd_inflight.get(dc_id, {})
        n_f = sum(1 for e in at.values() if e[3] == "f")
        n_d = sum(1 for e in at.values() if e[3] == "d")
        return [n_f,
                float(sum(e[1] for e in at.values() if e[3] == "f")),
                float(sum(e[2] for e in at.values() if e[3] == "f")),
                n_d,
                float(sum(e[1] for e in at.values() if e[3] == "d")),
                float(sum(e[2] for e in at.values() if e[3] == "d"))]

    def assert_consistent(where):
        for dc_id in dcs:
            got = s._fwd_agg.get(dc_id, [0, 0.0, 0.0, 0, 0.0, 0.0])
            want = recount(dc_id)
            assert [round(x, 6) for x in got] == [round(x, 6) for x in want], \
                f"{where}: {dc_id} sums {got} but a recount gives {want}"

    # 1. dispatches accumulate
    for i in range(40):
        s._fwd_begin(f"q{i}", dcs[i % 2],
                     inference_pb2.InferenceRequest(in_tok=100 + i,
                                                    out_tok=10 + i))
    assert_consistent("after dispatch")

    # 2. ageing moves entries between buckets
    for dc_id in dcs:
        s._stage_ttft_ms[dc_id] = 1.0       # 1 ms: everything is past it
    time.sleep(0.01)
    with s._fwd_lock:
        for dc_id in dcs:
            s._fwd_age(dc_id, time.monotonic())
    assert_consistent("after ageing")
    assert s._fwd_agg[US_EAST_DC][3] > 0, "nothing aged into decoding"

    # 3. completions subtract from the bucket the entry is actually in
    for i in range(0, 40, 3):
        s._fwd_end(f"q{i}", dcs[i % 2])
    assert_consistent("after completion")

    # 4. a snapshot retires everything dispatched before it
    s.PushSnapshot(inference_pb2.Snapshot(
        epoch=1, dc_view={d: stats(active=1) for d in dcs}), None)
    assert_consistent("after snapshot")
    for dc_id in dcs:
        assert s._fwd_agg.get(dc_id, [0])[0] == 0, \
            f"{dc_id} still projects prefills the snapshot already counted"

    # 5. and the projection built from the sums equals the one built from the
    #    per-placement lists, which is what this replaced.
    from scheduler.policies.common.projection import (project_placements,
                                                      project_totals)
    s._fwd_begin("z0", US_EAST_DC,
                 inference_pb2.InferenceRequest(in_tok=512, out_tok=64))
    s._fwd_begin("z1", US_EAST_DC,
                 inference_pb2.InferenceRequest(in_tok=300, out_tok=20))
    base = stats(active=3, pending=0, engines=8)
    a = s._fwd_agg[US_EAST_DC]
    from_totals = project_totals(base, a[0], a[1], a[2], a[3], a[4], a[5],
                                 s._max_num_seqs)
    from_lists = project_placements(base, [512, 300], [64, 20],
                                    s._max_num_seqs, stages=["f", "f"])
    assert from_totals == from_lists, \
        f"totals {from_totals} != lists {from_lists}"


if __name__ == "__main__":
    print(f"{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    if FAIL:
        sys.exit(1)
