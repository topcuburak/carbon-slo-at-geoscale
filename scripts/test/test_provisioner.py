#!/usr/bin/env python3
"""The provisioner, part by part.

It has one job -- poll every datacentre, work out how congested each group is,
push the result to every scheduler -- and three ways of getting it wrong that
have all happened:

  a slow peer          one unanswered poll used to kill the loop for good
  a partial wave       async mode refreshes a subset; the congestion estimate
                       must still see the whole fleet
  a site that leaves   the cache used to keep answering for datacentres that
                       had gone

No gRPC and no fleet: the peer registry is stubbed, so a failure here is the
provisioner's logic and not a port that was busy.

Run:  PYTHONPATH=.:proto python3 scripts/test/test_provisioner.py
"""

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "proto"))

import inference_pb2                                              # noqa: E402
from common.network import (REGION_TO_GROUP, SCHEDULER_GROUPS,    # noqa: E402
                            get_all_base_rtts, rtt_pair_key)
from provisioner.congestion import CongestionModel                # noqa: E402
from provisioner.polling import FleetPoller                       # noqa: E402
from provisioner.snapshot import SnapshotPusher                   # noqa: E402
from provisioner.service import ProvisionerServicer               # noqa: E402
from common.grpc_bind import bind_with_retry                      # noqa: E402

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
        except Exception as e:                                    # noqa: BLE001
            FAIL.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
        return fn
    return deco


# ─────────────────────────────── test doubles ───────────────────────────────

class FakeDcStub:
    """A datacentre that answers, fails, or hangs."""

    def __init__(self, dc_id, active=0, engines=8, ci=300.0,
                 delay=0.0, raises=None):
        self.dc_id, self.active, self.engines, self.ci = dc_id, active, engines, ci
        self.delay, self.raises = delay, raises
        self.calls = 0

    def PollStats(self, request, timeout=None):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        return inference_pb2.PollStatsResponse(
            dc_id=self.dc_id, epoch=request.epoch,
            active_requests=self.active, num_gpus=self.engines,
            carbon_intensity=self.ci, pue=getattr(self, "pue", 0.0))


class FakeSchedStub:
    def __init__(self, delay=0.0, raises=None):
        self.delay, self.raises, self.seen = delay, raises, []

    def PushSnapshot(self, snapshot, timeout=None):
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        self.seen.append(snapshot)
        return inference_pb2.Ack()


class FakePeers:
    def __init__(self, dcs=None, scheds=None):
        self.dcs = dcs or {}
        self.scheds = scheds or {}
        self.dc_endpoints = {d: f"localhost:{50000 + i}"
                             for i, d in enumerate(self.dcs)}
        self.sched_endpoints = {s: f"localhost:{51000 + i}"
                                for i, s in enumerate(self.scheds)}

    def dc_stub(self, dc_id):
        return self.dcs[dc_id]

    def sched_stub(self, sid):
        return self.scheds[sid]


def some_dcs(group, n):
    return list(SCHEDULER_GROUPS[group])[:n]


def stats(active=0, engines=8, ci=300.0):
    return inference_pb2.DcStats(active_requests=active, num_gpus=engines,
                                 carbon_intensity=ci)


# ──────────────────────────── congestion: heat ──────────────────────────────

@check("test_group_heat_is_the_mean_per_engine_batch")
def _():
    m = CongestionModel(beta=0.5, norm="cap", heat_cap=32.0)
    a, b = some_dcs("us_east", 2)
    heat = m.group_heat({a: stats(active=64, engines=8),     # 8 per engine
                         b: stats(active=32, engines=8)})    # 4 per engine
    assert abs(heat["us_east"] - 6.0) < 1e-9, heat["us_east"]


@check("test_a_group_with_no_reachable_site_reads_zero")
def _():
    # "empty" where it means "unknown" -- a known limitation, pinned so it
    # cannot change silently.
    m = CongestionModel()
    heat = m.group_heat({some_dcs("us_east", 1)[0]: stats(active=64)})
    assert heat["asia"] == 0.0, heat["asia"]
    assert set(heat) == set(SCHEDULER_GROUPS), "not every group is reported"


@check("test_heat_is_a_mean_so_group_size_does_not_dilute_it")
def _():
    # One saturated site moves a small group more than a large one. This is
    # deliberate and is SIMULATOR_TODO 16; measured here so it is a decision.
    m = CongestionModel()
    small = some_dcs("middle_east", 7)
    big = some_dcs("asia", 7)
    h1 = m.group_heat({small[0]: stats(active=256)}
                      | {d: stats(active=0) for d in small[1:3]})
    h2 = m.group_heat({big[0]: stats(active=256)}
                      | {d: stats(active=0) for d in big[1:6]})
    assert h1["middle_east"] > h2["asia"], (h1["middle_east"], h2["asia"])


# ──────────────────────── congestion: cap vs peak ───────────────────────────

@check("test_cap_normalisation_responds_to_the_level_of_load")
def _():
    m = CongestionModel(beta=0.5, norm="cap", heat_cap=32.0)
    d = some_dcs("us_east", 1)[0]
    light = m.rtt_matrix({d: stats(active=8, engines=8)})    # 1/32 of cap
    heavy = m.rtt_matrix({d: stats(active=256, engines=8)})  # at cap
    key = next(k for k in light if "us_east" in k and k.split("|")[0]
               != k.split("|")[1])
    assert heavy[key] > light[key] * 1.05, \
        f"load did not move the matrix: {light[key]:.2f} -> {heavy[key]:.2f}"


@check("test_peak_normalisation_is_blind_to_the_level")
def _():
    # The pre-2026-09-02 behaviour, and the reason it was replaced: peak makes
    # the busiest group 1.0 by construction, so four orders of magnitude of
    # load give one matrix. Pinned so nobody restores it by accident thinking
    # it is equivalent.
    m = CongestionModel(beta=0.5, norm="peak")
    d = some_dcs("us_east", 1)[0]
    light = m.rtt_matrix({d: stats(active=1, engines=8)})
    heavy = m.rtt_matrix({d: stats(active=2048, engines=8)})
    assert light == heavy, "peak has started responding to load"


@check("test_an_unknown_normalisation_is_refused_at_construction")
def _():
    try:
        CongestionModel(norm="whatever")
    except ValueError:
        return
    raise AssertionError("a bad norm was accepted")


# ──────────────────────── congestion: the matrix ────────────────────────────

@check("test_an_idle_fleet_leaves_the_base_matrix_alone")
def _():
    m = CongestionModel(beta=0.5, norm="cap")
    got = m.rtt_matrix({d: stats(active=0) for d in some_dcs("us_east", 3)})
    base = get_all_base_rtts()
    assert got == base, "an empty fleet inflated the RTT"


@check("test_beta_bounds_how_far_the_matrix_can_stretch")
def _():
    # Every group at the cap: path_heat is 1, so the ceiling is exactly
    # base * (1 + beta) and nothing can exceed it.
    beta = 0.5
    m = CongestionModel(beta=beta, norm="cap", heat_cap=32.0)
    saturated = {d: stats(active=256, engines=8)
                 for g in SCHEDULER_GROUPS for d in SCHEDULER_GROUPS[g]}
    got = m.rtt_matrix(saturated)
    base = get_all_base_rtts()
    for k, v in got.items():
        assert v <= base[k] * (1 + beta) + 1e-9, (k, v, base[k])
        assert abs(v - base[k] * (1 + beta)) < 1e-6, \
            f"{k}: saturated fleet should sit at the ceiling, got {v}"


@check("test_the_matrix_is_symmetric_in_its_two_groups")
def _():
    m = CongestionModel(beta=0.5, norm="cap")
    got = m.rtt_matrix({d: stats(active=200) for d in some_dcs("asia", 4)})
    for key in got:
        g1, g2 = key.split("|")
        assert got[key] == got[rtt_pair_key(g1, g2)], key
        assert rtt_pair_key(g1, g2) == rtt_pair_key(g2, g1), \
            f"the pair key is not order-free: {g1},{g2}"


@check("test_zero_beta_disables_the_model_entirely")
def _():
    m = CongestionModel(beta=0.0, norm="cap")
    loaded = {d: stats(active=256) for d in some_dcs("us_east", 3)}
    assert m.rtt_matrix(loaded) == get_all_base_rtts()


# ───────────────────────────── polling ──────────────────────────────────────

@check("test_a_poll_reads_every_datacentre_that_answers")
def _():
    dcs = {d: FakeDcStub(d, active=8 + i) for i, d in enumerate(some_dcs("us_east", 4))}
    p = FleetPoller(FakePeers(dcs), stale_after_epochs=3)
    view = p.poll_all()
    assert len(view) == 4, view
    for i, d in enumerate(sorted(dcs)):
        assert view[d].active_requests == dcs[d].active, d


@check("test_a_poll_carries_each_sites_pue_to_the_schedulers")
def _():
    # Carbon per unit of work is intensity x PUE; dropped here, every policy
    # would rank sites on the grid alone.
    dcs = {d: FakeDcStub(d) for d in some_dcs("us_east", 2)}
    for i, d in enumerate(sorted(dcs)):
        dcs[d].pue = 1.08 + 0.2 * i
    view = FleetPoller(FakePeers(dcs), stale_after_epochs=3).poll_all()
    for d in dcs:
        assert abs(view[d].pue - dcs[d].pue) < 1e-9, (d, view[d].pue)


@check("test_one_failing_datacentre_does_not_take_the_others_down")
def _():
    ids = some_dcs("us_east", 4)
    dcs = {d: FakeDcStub(d, active=5) for d in ids}
    dcs[ids[1]].raises = RuntimeError("connection refused")
    p = FleetPoller(FakePeers(dcs), stale_after_epochs=3)
    view = p.poll_all()
    assert ids[1] not in view, "a failing site reported a reading"
    assert len(view) == 3, view


@check("test_one_slow_datacentre_does_not_kill_the_polling_loop")
def _():
    # The defect this exists for: as_completed raises TimeoutError, distinct
    # from a per-future exception, and it used to escape an unsupervised daemon
    # thread. The loop died after a single epoch while the gRPC server went on
    # answering and every scheduler kept its last snapshot forever.
    import provisioner.polling as pol
    ids = some_dcs("us_east", 3)
    dcs = {d: FakeDcStub(d, active=5) for d in ids}
    dcs[ids[0]].delay = 5.0                      # never beats the deadline
    old = pol.POLL_DEADLINE_S
    pol.POLL_DEADLINE_S = 0.25
    try:
        p = FleetPoller(FakePeers(dcs), stale_after_epochs=3)
        first = p.poll_all()                     # must not raise
        second = p.poll_all()                    # and must run again
    finally:
        pol.POLL_DEADLINE_S = old
    assert len(first) == 2, f"the two quick sites should still answer: {first}"
    assert len(second) == 2, "the loop did not survive to a second epoch"
    assert p.epoch == 2, p.epoch


@check("test_a_datacentre_that_leaves_discovery_is_dropped_at_once")
def _():
    ids = some_dcs("us_east", 3)
    dcs = {d: FakeDcStub(d, active=5) for d in ids}
    peers = FakePeers(dcs)
    p = FleetPoller(peers, stale_after_epochs=3)
    p.remember(p.poll_all())
    assert len(p.view) == 3, p.view

    gone = ids[2]
    del peers.dc_endpoints[gone]
    p.remember(p.poll(peers.dc_endpoints.keys()))
    assert gone not in p.view, \
        "a site that left discovery is a wrong reading, not a stale one"


@check("test_a_site_that_stops_answering_is_kept_briefly_then_dropped")
def _():
    ids = some_dcs("us_east", 3)
    dcs = {d: FakeDcStub(d, active=5) for d in ids}
    p = FleetPoller(FakePeers(dcs), stale_after_epochs=2)
    p.remember(p.poll_all())
    quiet = ids[1]
    dcs[quiet].raises = RuntimeError("down")

    p.remember(p.poll_all())
    assert quiet in p.view, "dropped on the first miss; a blip is not a death"
    p.remember(p.poll_all())
    assert quiet in p.view, "dropped before the bound"
    p.remember(p.poll_all())
    assert quiet not in p.view, \
        "kept past stale_after_epochs; it no longer describes anything"


@check("test_the_cache_survives_a_partial_wave")
def _():
    # Async mode: a wave refreshes a subset, and the sites outside it must keep
    # their last reading or the congestion estimate sees a fleet that is mostly
    # empty.
    ids = some_dcs("us_east", 4)
    dcs = {d: FakeDcStub(d, active=100) for d in ids}
    p = FleetPoller(FakePeers(dcs), stale_after_epochs=10)
    p.remember(p.poll_all())
    for d in ids:
        dcs[d].active = 1
    p.remember(p.poll(ids[:2]))                  # only half the fleet
    assert p.view[ids[0]].active_requests == 1, "the polled half did not refresh"
    assert p.view[ids[2]].active_requests == 100, \
        "an unpolled site lost its cached reading"


@check("test_polling_an_empty_set_is_a_no_op")
def _():
    p = FleetPoller(FakePeers({}), stale_after_epochs=3)
    assert p.poll([]) == {}


# ───────────────────────────── snapshots ────────────────────────────────────

@check("test_waves_partition_the_fleet_with_nothing_lost_or_repeated")
def _():
    pusher = SnapshotPusher(FakePeers(), wave_count=4)
    ids = some_dcs("asia", 26)
    waves = pusher.waves(ids)
    assert len(waves) == 4, len(waves)
    flat = [d for w in waves for d in w]
    assert sorted(flat) == sorted(ids), "a wave split lost or duplicated a site"
    assert max(len(w) for w in waves) - min(len(w) for w in waves) <= 1, \
        [len(w) for w in waves]


@check("test_wave_membership_is_stable_across_cycles")
def _():
    # The per-DC refresh period is wave_count x wave_interval only if a site
    # stays in the same wave.
    pusher = SnapshotPusher(FakePeers(), wave_count=4)
    ids = some_dcs("asia", 26)
    assert pusher.waves(ids) == pusher.waves(list(reversed(ids))), \
        "wave membership depends on input order"


@check("test_one_wave_is_the_whole_fleet")
def _():
    pusher = SnapshotPusher(FakePeers(), wave_count=1)
    ids = some_dcs("us_east", 8)
    assert pusher.waves(ids) == [sorted(ids)]


@check("test_every_scheduler_gets_the_snapshot")
def _():
    scheds = {f"sched-{g}": FakeSchedStub() for g in ("us_east", "us_west", "asia")}
    pusher = SnapshotPusher(FakePeers(scheds=scheds), wave_count=1)
    view = {some_dcs("us_east", 1)[0]: stats(active=7)}
    pusher.push(epoch=5, dc_view=view, rtt_matrix={"asia|us_east": 120.0})
    for sid, st in scheds.items():
        assert len(st.seen) == 1, f"{sid} got {len(st.seen)} snapshots"
        assert st.seen[0].epoch == 5, sid
        assert dict(st.seen[0].rtt_matrix) == {"asia|us_east": 120.0}, sid


@check("test_one_unreachable_scheduler_does_not_block_the_others")
def _():
    scheds = {"sched-us_east": FakeSchedStub(),
              "sched-us_west": FakeSchedStub(raises=RuntimeError("refused")),
              "sched-asia": FakeSchedStub()}
    pusher = SnapshotPusher(FakePeers(scheds=scheds), wave_count=1)
    pusher.push(epoch=1, dc_view={}, rtt_matrix={})
    assert len(scheds["sched-us_east"].seen) == 1
    assert len(scheds["sched-asia"].seen) == 1


@check("test_one_slow_scheduler_does_not_hold_the_cycle_open")
def _():
    import provisioner.snapshot as snap
    scheds = {"sched-us_east": FakeSchedStub(),
              "sched-asia": FakeSchedStub(delay=5.0)}
    old = snap.PUSH_DEADLINE_S
    snap.PUSH_DEADLINE_S = 0.25
    try:
        pusher = SnapshotPusher(FakePeers(scheds=scheds), wave_count=1)
        t0 = time.monotonic()
        pusher.push(epoch=1, dc_view={}, rtt_matrix={})   # must not raise
        elapsed = time.monotonic() - t0
    finally:
        snap.PUSH_DEADLINE_S = old
    assert elapsed < 2.0, f"the push waited {elapsed:.1f}s on a slow scheduler"
    assert len(scheds["sched-us_east"].seen) == 1, "the quick one missed out"


# ────────────────────────────── the service ─────────────────────────────────

def holder(peers, poller, period=1.0):
    """The subset of Provisioner the servicer actually reads."""
    class H:
        pass
    h = H()
    h.epoch = poller.epoch
    h.poll_period_s = period
    h.last_poll_duration = poller.last_poll_duration
    h.dc_endpoints = peers.dc_endpoints
    h.sched_endpoints = peers.sched_endpoints
    h.poller = poller
    return h


@check("test_getstatus_reports_the_fleet_it_is_watching")
def _():
    ids = some_dcs("us_east", 3)
    dcs = {d: FakeDcStub(d, active=4, ci=250.0) for d in ids}
    peers = FakePeers(dcs, {"sched-us_east": FakeSchedStub(),
                            "sched-asia": FakeSchedStub()})
    poller = FleetPoller(peers, stale_after_epochs=3)
    poller.remember(poller.poll_all())

    resp = ProvisionerServicer(holder(peers, poller)).GetStatus(
        inference_pb2.Empty(), None)
    assert resp.num_datacenters == 3, resp.num_datacenters
    assert resp.num_schedulers == 2, resp.num_schedulers
    assert resp.current_epoch == poller.epoch, resp.current_epoch
    assert resp.poll_period_s == 1.0, resp.poll_period_s


@check("test_the_carbon_report_covers_every_datacentre_and_totals_them")
def _():
    ids = some_dcs("us_east", 3)
    dcs = {d: FakeDcStub(d, active=4, ci=100.0 + 10 * i)
           for i, d in enumerate(sorted(ids))}
    peers = FakePeers(dcs, {})
    poller = FleetPoller(peers, stale_after_epochs=3)
    poller.remember(poller.poll_all())

    resp = ProvisionerServicer(holder(peers, poller)).GetCarbonReport(
        inference_pb2.Empty(), None)
    assert len(resp.dc_stats) == 3, len(resp.dc_stats)
    assert {r.dc_id for r in resp.dc_stats} == set(ids), \
        {r.dc_id for r in resp.dc_stats}
    assert abs(resp.total_energy_j - sum(r.total_energy_j
                                         for r in resp.dc_stats)) < 1e-9


@check("test_the_carbon_report_skips_a_datacentre_that_will_not_answer")
def _():
    ids = some_dcs("us_east", 3)
    dcs = {d: FakeDcStub(d, active=4) for d in ids}
    dcs[ids[1]].raises = RuntimeError("refused")
    peers = FakePeers(dcs, {})
    poller = FleetPoller(peers, stale_after_epochs=3)
    resp = ProvisionerServicer(holder(peers, poller)).GetCarbonReport(
        inference_pb2.Empty(), None)
    assert len(resp.dc_stats) == 2, len(resp.dc_stats)
    assert ids[1] not in {r.dc_id for r in resp.dc_stats}


# ─────────────────────── the parts, working together ────────────────────────

@check("test_congestion_is_computed_from_the_cache_not_the_wave")
def _():
    # The property the whole async design rests on: a wave refreshes a subset,
    # but the RTT matrix must be built from the full cached fleet or it swings
    # every cycle with whichever sites happened to be polled.
    ids = some_dcs("us_east", 4)
    # Deliberately uneven: if every site carried the same load, a one-site wave
    # would give the same mean as the whole fleet and this would pass whether
    # or not the cache was used.
    dcs = {d: FakeDcStub(d, active=a, engines=8)
           for d, a in zip(sorted(ids), (256, 0, 0, 0))}
    poller = FleetPoller(FakePeers(dcs), stale_after_epochs=10)
    poller.remember(poller.poll_all())
    model = CongestionModel(beta=0.5, norm="cap", heat_cap=32.0)
    full = model.rtt_matrix(poller.view)

    wave = poller.poll(sorted(ids)[:1])          # the loaded site alone
    poller.remember(wave)
    assert model.rtt_matrix(poller.view) == full, \
        "the matrix moved because only part of the fleet was refreshed"
    assert model.rtt_matrix(wave) != full, \
        "the wave alone gives the same matrix; this test proves nothing"


@check("test_the_loop_survives_a_datacentre_that_never_answers")
def _():
    # End to end over the parts, with the failure that killed it in production.
    import provisioner.polling as pol
    ids = some_dcs("us_east", 4)
    dcs = {d: FakeDcStub(d, active=32) for d in ids}
    dcs[ids[0]].delay = 5.0
    scheds = {"sched-us_east": FakeSchedStub()}
    peers = FakePeers(dcs, scheds)
    old = pol.POLL_DEADLINE_S
    pol.POLL_DEADLINE_S = 0.2
    try:
        poller = FleetPoller(peers, stale_after_epochs=5)
        pusher = SnapshotPusher(peers, wave_count=1)
        model = CongestionModel(beta=0.5, norm="cap")
        for _ in range(3):
            view = poller.poll_all()
            poller.remember(view)
            pusher.push(poller.epoch, view, model.rtt_matrix(poller.view))
    finally:
        pol.POLL_DEADLINE_S = old
    assert poller.epoch == 3, f"the loop stopped at epoch {poller.epoch}"
    assert len(scheds["sched-us_east"].seen) == 3, \
        f"only {len(scheds['sched-us_east'].seen)} snapshots reached the scheduler"


# ═══════════════════════════ the peer registry ══════════════════════════════

@check("test_discovery_splits_datacentres_from_schedulers")
def _():
    import provisioner.peers as peers_mod
    from provisioner.peers import PeerRegistry
    old_dc, old_s = peers_mod.discover_datacenters, peers_mod.discover_schedulers
    peers_mod.discover_datacenters = lambda: {"US_PJM": "localhost:1",
                                              "US_MISO": "localhost:2"}
    peers_mod.discover_schedulers = lambda: {"sched-us_east": "localhost:3"}
    try:
        r = PeerRegistry()
        r.refresh()
    finally:
        peers_mod.discover_datacenters, peers_mod.discover_schedulers = old_dc, old_s
    assert set(r.dc_endpoints) == {"US_PJM", "US_MISO"}, r.dc_endpoints
    assert set(r.sched_endpoints) == {"sched-us_east"}, r.sched_endpoints


@check("test_discovery_failing_leaves_the_last_map_rather_than_emptying_it")
def _():
    # A discovery blip must not look like a fleet that vanished: an empty map
    # means nothing gets polled and every scheduler keeps its last snapshot.
    import provisioner.peers as peers_mod
    from provisioner.peers import PeerRegistry
    old_dc, old_s = peers_mod.discover_datacenters, peers_mod.discover_schedulers
    peers_mod.discover_datacenters = lambda: {"US_PJM": "localhost:1"}
    peers_mod.discover_schedulers = lambda: {"sched-us_east": "localhost:3"}
    try:
        r = PeerRegistry()
        r.refresh()
        peers_mod.discover_datacenters = lambda: (_ for _ in ()).throw(
            RuntimeError("api down"))
        r.refresh()                                  # must not raise
    finally:
        peers_mod.discover_datacenters, peers_mod.discover_schedulers = old_dc, old_s
    assert r.dc_endpoints == {"US_PJM": "localhost:1"}, \
        f"a discovery failure emptied the fleet: {r.dc_endpoints}"


@check("test_a_channel_is_opened_once_and_reused")
def _():
    from provisioner.peers import PeerRegistry
    r = PeerRegistry()
    r.dc_endpoints = {"US_PJM": "localhost:59999"}
    r.sched_endpoints = {"sched-us_east": "localhost:59998"}
    a, b = r.dc_stub("US_PJM"), r.dc_stub("US_PJM")
    assert r._dc_channels["US_PJM"] is not None
    assert len(r._dc_channels) == 1, "a second channel was opened for one peer"
    r.sched_stub("sched-us_east")
    assert len(r._sched_channels) == 1
    r.close()


# ══════════════════════════════ the loop ════════════════════════════════════
#
# The parts above are tested one at a time. This is the thing that turns them:
# how often, in what order, and what happens when a cycle throws.

from provisioner.loop import Provisioner                            # noqa: E402


def provisioner(dcs, scheds, **kw):
    """A Provisioner wired to fakes instead of real peers."""
    kw.setdefault("poll_period_s", 0.05)
    kw.setdefault("discovery_refresh_s", 1e9)     # never re-discover by default
    p = Provisioner(**kw)
    fake = FakePeers(dcs, scheds)
    fake.refresh = lambda: None
    p.peers = fake
    p.poller.peers = fake
    p.pusher.peers = fake
    return p


def run_for(p, seconds):
    t = threading.Thread(target=p.run, daemon=True)
    t.start()
    time.sleep(seconds)
    p.stop()
    t.join(timeout=10)
    assert not t.is_alive(), "the loop did not stop when asked"


@check("test_a_mistyped_mode_is_refused_rather_than_falling_through_to_sync")
def _():
    # Running sync while believing it is async makes the comparison the mode
    # exists for meaningless, and leaves no trace.
    try:
        Provisioner(poll_period_s=1, discovery_refresh_s=10, poll_mode="asnyc")
    except ValueError as e:
        assert "asnyc" in str(e), e
        return
    raise AssertionError("a mistyped poll_mode was accepted")


@check("test_sync_polls_the_whole_fleet_every_period")
def _():
    ids = some_dcs("us_east", 4)
    dcs = {d: FakeDcStub(d, active=8) for d in ids}
    scheds = {"sched-us_east": FakeSchedStub()}
    p = provisioner(dcs, scheds, poll_mode="sync", poll_period_s=0.05)
    run_for(p, 0.35)
    assert p.epoch >= 3, f"only {p.epoch} epochs in 0.35 s at 50 ms"
    for d in ids:
        assert dcs[d].calls >= p.epoch - 1, \
            f"{d} was polled {dcs[d].calls} times over {p.epoch} epochs"
    snaps = scheds["sched-us_east"].seen
    assert len(snaps) >= 3, len(snaps)
    assert all(len(s.dc_view) == 4 for s in snaps), \
        "a sync snapshot did not carry the whole fleet"


@check("test_async_pushes_one_wave_at_a_time")
def _():
    ids = some_dcs("asia", 8)
    dcs = {d: FakeDcStub(d, active=8) for d in ids}
    scheds = {"sched-asia": FakeSchedStub()}
    p = provisioner(dcs, scheds, poll_mode="async", wave_count=4,
                    wave_interval_s=0.02)
    run_for(p, 0.4)
    snaps = scheds["sched-asia"].seen
    assert len(snaps) >= 4, len(snaps)
    sizes = {len(s.dc_view) for s in snaps}
    assert sizes == {2}, \
        f"8 datacentres over 4 waves should push 2 at a time, saw {sizes}"


@check("test_async_still_reaches_every_datacentre")
def _():
    # A scheduler merges rather than replaces, so a complete view is reached
    # across a cycle even though no single push carries one.
    ids = some_dcs("asia", 8)
    dcs = {d: FakeDcStub(d, active=8) for d in ids}
    scheds = {"sched-asia": FakeSchedStub()}
    p = provisioner(dcs, scheds, poll_mode="async", wave_count=4,
                    wave_interval_s=0.02)
    run_for(p, 0.4)
    covered = set()
    for s in scheds["sched-asia"].seen:
        covered |= set(s.dc_view)
    assert covered == set(ids), f"missed {set(ids) - covered}"


@check("test_the_stale_bound_clears_a_whole_async_cycle")
def _():
    # In async a site is polled once every wave_count epochs. A bound shorter
    # than a cycle would drop every datacentre for being quiet when it is
    # simply not this wave's turn.
    for waves in (1, 4, 8):
        p = Provisioner(poll_period_s=1, discovery_refresh_s=10,
                        poll_mode="async", wave_count=waves)
        assert p.poller.stale_after_epochs > waves, \
            (waves, p.poller.stale_after_epochs)


@check("test_a_snapshot_is_priced_against_the_whole_fleet_not_the_wave")
def _():
    # The property async rests on: the RTT matrix must come from the cache, or
    # it swings every wave with whichever sites happened to be polled.
    ids = some_dcs("us_east", 8)
    dcs = {d: FakeDcStub(d, active=a, engines=8)
           for d, a in zip(sorted(ids), (256, 0, 0, 0, 0, 0, 0, 0))}
    scheds = {"sched-us_east": FakeSchedStub()}
    p = provisioner(dcs, scheds, poll_mode="async", wave_count=4,
                    wave_interval_s=0.02)
    run_for(p, 0.5)
    seen = scheds["sched-us_east"].seen
    assert len(seen) >= 8, len(seen)
    # The first cycle is the cache filling: wave 1 has seen 2 sites, wave 2 has
    # seen 4, and the mean load really is different each time. That is warm-up,
    # not a bug -- and it is also what makes the check below meaningful.
    first_cycle = [dict(x.rtt_matrix) for x in seen[:4]]
    assert len({tuple(sorted(m.items())) for m in first_cycle}) > 1, \
        "the matrix did not move while the cache was still filling; this test " \
        "would pass even if the matrix were constant"
    # Once every site has been seen once, the cache is complete and the matrix
    # must stop moving -- it is priced on the cache, not on the wave.
    settled = [dict(x.rtt_matrix) for x in seen[4:]]
    assert len(settled) >= 4, len(settled)
    assert all(m == settled[0] for m in settled), \
        "the RTT matrix keeps moving after the cache is full; it is being " \
        "priced on the wave"


@check("test_a_cycle_that_throws_does_not_end_the_loop")
def _():
    # run() catches BaseException per cycle on purpose: a polling thread that
    # dies leaves the gRPC server answering and every scheduler frozen on its
    # last snapshot, which looks like a healthy run.
    ids = some_dcs("us_east", 3)
    dcs = {d: FakeDcStub(d, active=4) for d in ids}
    scheds = {"sched-us_east": FakeSchedStub()}
    p = provisioner(dcs, scheds, poll_mode="sync", poll_period_s=0.05)

    calls = {"n": 0}
    real = p.poller.poll_all

    def flaky():
        calls["n"] += 1
        if calls["n"] in (2, 3):
            raise RuntimeError("boom")
        return real()
    p.poller.poll_all = flaky

    run_for(p, 0.4)
    assert calls["n"] >= 5, f"the loop stopped after {calls['n']} cycles"
    assert len(scheds["sched-us_east"].seen) >= 3, \
        "the loop did not recover after the failing cycles"


@check("test_an_empty_fleet_is_skipped_rather_than_pushed_as_empty")
def _():
    # Pushing an empty view would tell every scheduler the fleet is idle.
    scheds = {"sched-us_east": FakeSchedStub()}
    p = provisioner({}, scheds, poll_mode="sync", poll_period_s=0.05)
    run_for(p, 0.25)
    assert not scheds["sched-us_east"].seen, \
        "an empty fleet was pushed as a snapshot"


@check("test_discovery_is_refreshed_on_its_own_interval")
def _():
    ids = some_dcs("us_east", 2)
    dcs = {d: FakeDcStub(d, active=1) for d in ids}
    scheds = {"sched-us_east": FakeSchedStub()}
    p = provisioner(dcs, scheds, poll_mode="sync", poll_period_s=0.05,
                    discovery_refresh_s=0.1)
    hits = {"n": 0}

    def refresh():
        hits["n"] += 1
    p.peers.refresh = refresh
    run_for(p, 0.45)
    assert hits["n"] >= 3, \
        f"discovery ran {hits['n']} times in 0.45 s at a 0.1 s interval"


@check("test_stop_ends_the_loop_promptly")
def _():
    ids = some_dcs("us_east", 2)
    p = provisioner({d: FakeDcStub(d) for d in ids},
                    {"sched-us_east": FakeSchedStub()},
                    poll_mode="sync", poll_period_s=5.0)
    t = threading.Thread(target=p.run, daemon=True)
    t.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    p.stop()
    t.join(timeout=10)
    assert not t.is_alive(), "stop() did not end the loop"
    assert time.monotonic() - t0 < 2.0, \
        "stop() waited out the poll period instead of interrupting it"


@check("test_the_epoch_the_servicer_reports_is_the_pollers")
def _():
    ids = some_dcs("us_east", 2)
    p = provisioner({d: FakeDcStub(d) for d in ids},
                    {"sched-us_east": FakeSchedStub()},
                    poll_mode="sync", poll_period_s=0.05)
    run_for(p, 0.25)
    resp = ProvisionerServicer(p).GetStatus(inference_pb2.Empty(), None)
    assert resp.current_epoch == p.poller.epoch, (resp.current_epoch, p.epoch)
    assert resp.num_datacenters == 2, resp.num_datacenters
    assert resp.poll_period_s == 0.05, resp.poll_period_s


class FlakyServer:
    """add_insecure_port that fails like a port still in TIME_WAIT."""

    def __init__(self, failures):
        self.failures, self.calls = failures, []

    def add_insecure_port(self, address):
        self.calls.append(address)
        if len(self.calls) <= self.failures:
            raise RuntimeError(f"Failed to bind to address {address}")
        return int(address.rsplit(":", 1)[1])


@check("test_bind_waits_out_a_port_still_coming_free")
def _():
    # 2026-09-10: a provisioner that died here left sixteen datacentres
    # serving, and the next two sweeps never started a cell.
    s = FlakyServer(failures=2)
    bind_with_retry(s, 50300, tries=5, gap_s=0)
    assert s.calls == ["[::]:50300"] * 3, s.calls


@check("test_bind_gives_up_when_the_port_never_frees")
def _():
    s = FlakyServer(failures=10)
    try:
        bind_with_retry(s, 50300, tries=3, gap_s=0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a port that never frees must still fail the start")
    assert len(s.calls) == 3, s.calls


if __name__ == "__main__":
    print(f"{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    if FAIL:
        sys.exit(1)
