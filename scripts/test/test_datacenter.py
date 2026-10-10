#!/usr/bin/env python3
"""What the Datacenter must hold, method by method.

The datacentre is a summing layer with two jobs: place a request on one of its
nodes, and answer for what its nodes did. Both are easy to get subtly wrong in
ways nothing else notices -- a total that double counts, a split that does not
add up, an aggregate taken by a route that disagrees with the direct one -- and
the consumers of those numbers are the ILP cost function, the provisioner's RTT
heat and the carbon result.

So most of these check the SAME quantity by two routes: what the datacentre
reports, against what its engines actually hold. They pass trivially while the
two paths are the same code, and stop passing the moment one of them drifts.

    python scripts/test/test_datacenter.py

Companion to test_inference_engine.py, which covers the engine.
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import inference_pb2                                             # noqa: E402
from datacenter.meta_dc import Datacenter                       # noqa: E402
from datacenter.meta_node import HOST_POWER_W                     # noqa: E402
from datacenter.perf_models.analytical.latency import AnalyticalLatencyModel  # noqa: E402
from datacenter.scheduling import available_policies             # noqa: E402

MODEL, TP = "llama3_8B", 2
CI = 400.0            # gCO2eq/kWh


def dc(nodes=2, cards=4, max_bs=32, pue=1.1, ci=CI,
       within_dc_policy="round_robin", within_node_policy="round_robin"):
    """A datacentre of `nodes` nodes, each `cards // TP` engines."""
    return Datacenter("t", num_nodes=nodes, num_gpus_per_node=cards, lookup=None,
                      tp_size=TP, max_batch_size=max_bs,
                      analytical_model=AnalyticalLatencyModel(MODEL, tp=TP),
                      latency_mode="analytical", model_name=MODEL,
                      max_model_len=32768, carbon_intensity=ci, pue=pue,
                      within_dc_policy=within_dc_policy, within_node_policy=within_node_policy)


def engines(d):
    return [g for n in d.nodes for g in n.inference_engines]


def drive(d, reqs):
    """Send reqs as (isl, osl) concurrently and wait for all of them."""
    out, lk = [], threading.Lock()

    def one(isl, osl):
        r = d.dispatch(isl, osl)
        with lk:
            out.append(r)

    ths = [threading.Thread(target=one, args=(a, b)) for a, b in reqs]
    t0 = time.monotonic()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return out, t0, time.monotonic()


# ── __init__ ─────────────────────────────────────────────────────────────

def test_topology_is_what_was_asked_for():
    d = dc(nodes=3, cards=4)
    assert len(d.nodes) == 3, len(d.nodes)
    assert all(len(n.inference_engines) == 4 // TP for n in d.nodes)
    assert d.stats()["num_gpus"] == 3 * (4 // TP), d.stats()["num_gpus"]
    assert d.stats()["num_nodes"] == 3


def test_host_power_is_charged_once_per_node():
    """Not once per card and not once per engine: HOST_POWER_W is the box."""
    for cards in (2, 4, 8):
        d = dc(nodes=2, cards=cards)
        for n in d.nodes:
            share = sum(g._host_power_share_w for g in n.inference_engines)
            assert abs(share - HOST_POWER_W) < 1e-6, (cards, share)


def test_every_node_gets_its_own_placement_policy():
    """Shared state here would couple nodes that are meant to be independent."""
    d = dc(nodes=3)
    ids = [id(n._within_node_scheduler) for n in d.nodes]
    assert len(set(ids)) == len(ids), ids
    assert id(d._within_dc_scheduler) not in ids


def test_bad_topology_is_refused_at_construction():
    try:
        dc(nodes=1, cards=TP - 1)
    except ValueError:
        return
    raise AssertionError("a node with no engines was accepted")


# ── next_node / dispatch ─────────────────────────────────────────────────

def test_round_robin_visits_every_node_evenly():
    d = dc(nodes=4)
    seen = [d.next_node().node_id for _ in range(4 * 5)]
    counts = {k: seen.count(k) for k in set(seen)}
    assert len(counts) == 4, counts
    assert set(counts.values()) == {5}, counts


def test_dispatch_reports_where_the_request_actually_ran():
    d = dc(nodes=2)
    out, _, _ = drive(d, [(128, 4)] * 8)
    by_node = {}
    for r in out:
        assert r["dc_id"] == "t", r["dc_id"]
        assert r["gpu_id"].startswith(r["node_id"]), (r["node_id"], r["gpu_id"])
        by_node[r["node_id"]] = by_node.get(r["node_id"], 0) + 1
    assert len(by_node) == 2, by_node
    assert max(by_node.values()) - min(by_node.values()) <= 1, by_node


def test_every_registered_policy_places_at_both_levels():
    """A policy that cannot read its metric off a node or an engine is broken."""
    for pol in available_policies():
        d = dc(nodes=2, within_dc_policy=pol, within_node_policy=pol)
        out, _, _ = drive(d, [(128, 4)] * 8)
        assert len(out) == 8, (pol, len(out))
        served = sum(g._total_requests for g in engines(d))
        assert served == 8, (pol, served)


# ── stats ────────────────────────────────────────────────────────────────

def test_counters_match_a_direct_walk_over_the_engines():
    """The reported total, against the engines themselves, mid-flight.

    The datacentre sums nodes and the node sums engines. This takes the third
    route -- straight to the engines -- so a mistake in either summing layer
    shows up instead of cancelling.
    """
    d = dc(nodes=2, max_bs=4)
    stop = threading.Event()
    bad = []

    def check():
        while not stop.is_set():
            s = d.stats()
            gs = engines(d)
            direct = {
                "total_active_requests": sum(g.active_count for g in gs),
                "total_pending_requests": sum(g.pending_count for g in gs),
                "total_prefilling_requests": sum(g.prefill_count for g in gs),
                "total_decoding_requests": sum(g.decoding_count for g in gs),
            }
            # active and pending are read at slightly different instants by the
            # two routes, so only the identity that must hold within one
            # snapshot is asserted: the split of active.
            if s["total_active_requests"] != (s["total_prefilling_requests"]
                                              + s["total_decoding_requests"]):
                bad.append(("split", s))
            if s["num_gpus"] != len(gs):
                bad.append(("num_gpus", s["num_gpus"], len(gs)))
            time.sleep(0.01)
    threading.Thread(target=check, daemon=True).start()
    drive(d, [(256, 8)] * 24)
    stop.set()
    time.sleep(0.05)
    assert not bad, bad[:3]


def test_counters_drain_to_zero():
    d = dc(nodes=2, max_bs=4)
    drive(d, [(256, 8)] * 24)
    time.sleep(0.2)
    s = d.stats()
    for k in ("total_active_requests", "total_pending_requests",
              "total_prefilling_requests", "total_decoding_requests",
              "total_pending_isl", "total_active_isl", "total_remaining_osl"):
        assert s[k] == 0, (k, s[k])


def test_empty_engine_count_is_reported_and_bounded():
    """The term a fleet mean throws away. Must never exceed the engine count."""
    d = dc(nodes=2)
    s = d.stats()
    assert s["empty_engines"] == s["num_gpus"], s
    assert s["max_engine_active"] == 0, s
    out = []
    stop = threading.Event()
    seen = []

    def sampler():
        while not stop.is_set():
            x = d.stats()
            seen.append((x["empty_engines"], x["max_engine_active"],
                         x["num_gpus"]))
            time.sleep(0.01)
    threading.Thread(target=sampler, daemon=True).start()
    drive(d, [(256, 8)] * 16)
    stop.set()
    time.sleep(0.05)
    assert seen, "no samples"
    for empty, mx, n in seen:
        assert 0 <= empty <= n, (empty, n)
        assert mx >= 0, mx
    assert any(e < n for e, _, n in seen), "no engine ever looked busy"


def test_health_and_diagnostic_counters_are_present_and_start_clean():
    d = dc()
    s = d.stats()
    for k in ("engine_faults", "osl_clamped", "kv_blocked_steps"):
        assert s[k] == 0, (k, s[k])
    assert s["kv_capacity_tokens"] > 0, s["kv_capacity_tokens"]
    assert s["kv_tokens"] == 0, s["kv_tokens"]


def test_avg_batch_size_is_the_total_over_the_engine_count():
    d = dc(nodes=2)
    s = d.stats()
    assert s["avg_batch_size"] == 0.0
    stop = threading.Event()
    bad = []

    def sampler():
        while not stop.is_set():
            x = d.stats()
            want = x["total_active_requests"] / max(x["num_gpus"], 1)
            if abs(x["avg_batch_size"] - want) > 1e-9:
                bad.append(x)
            time.sleep(0.01)
    threading.Thread(target=sampler, daemon=True).start()
    drive(d, [(256, 8)] * 16)
    stop.set()
    time.sleep(0.05)
    assert not bad, bad[:2]


def test_a_dead_engine_gives_its_counters_back():
    """The failure path, which the happy-path drain test cannot reach.

    When an engine's loop died it used to clear its queues and fail its
    requests but keep the counters, so the site published its crash-time load
    for the rest of the run with nothing left to decrement it. Every consumer
    reads those -- avg_batch_size, the ILP's dc_bs, the provisioner's RTT heat
    -- so a dead engine made its whole datacentre look permanently busy.
    """
    d = dc(nodes=1, cards=4, max_bs=8)
    drive(d, [(128, 4)] * 4)                    # give every engine a live loop
    victim = d.nodes[0].inference_engines[1]
    saved = victim._get_tpot

    def boom(*a, **k):
        raise RuntimeError("injected engine fault")
    victim._get_tpot = boom

    failed = 0
    lk = threading.Lock()

    def one():
        nonlocal failed
        try:
            d.dispatch(256, 8)
        except Exception:
            with lk:
                failed += 1

    ths = [threading.Thread(target=one) for _ in range(16)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    victim._get_tpot = saved
    time.sleep(0.3)

    assert failed > 0, "the injected fault never reached a request"
    s = d.stats()
    assert s["engine_faults"] > 0, s["engine_faults"]
    for k in ("total_active_requests", "total_decoding_requests",
              "total_prefilling_requests", "total_active_isl",
              "total_remaining_osl", "total_generated_tokens"):
        assert s[k] == 0, (k, s[k], "counters kept after a fault")
    rows = d.nodes[0].stats()["per_engine"]
    assert sum(r["faults"] for r in rows) == s["engine_faults"], rows


# ── energy ───────────────────────────────────────────────────────────────

def test_energy_matches_the_sum_over_engines():
    d = dc(nodes=2)
    _, t0, t1 = drive(d, [(256, 8)] * 16)
    d.compute_energy(t0, t1)
    s = d.stats()
    direct = sum(g.energy_j() for g in engines(d))
    assert abs(s["total_energy_j"] - direct) < 1e-6, (s["total_energy_j"], direct)
    assert direct > 0, direct


def test_the_energy_split_adds_up():
    """work = total - idle - host must not be negative, and idle must be real."""
    d = dc(nodes=2)
    _, t0, t1 = drive(d, [(256, 8)] * 16)
    d.compute_energy(t0, t1)
    s = d.stats()
    work = s["total_energy_j"] - s["total_idle_j"] - s["total_host_j"]
    assert work >= -1e-6, (work, s)
    assert s["total_idle_j"] >= 0 and s["total_host_j"] > 0, s


def test_host_energy_is_the_node_count_times_the_window():
    d = dc(nodes=2)
    _, t0, t1 = drive(d, [(256, 8)] * 8)
    d.compute_energy(t0, t1)
    want = len(d.nodes) * HOST_POWER_W * (t1 - t0)
    got = d.stats()["total_host_j"]
    assert abs(got - want) < 1e-3 * want, (got, want)


def test_computing_energy_twice_gives_the_same_answer():
    """It assigns rather than accumulates; a += here would double every run."""
    d = dc(nodes=2)
    _, t0, t1 = drive(d, [(256, 8)] * 12)
    d.compute_energy(t0, t1)
    first = d.stats()["total_energy_j"]
    d.compute_energy(t0, t1)
    assert abs(d.stats()["total_energy_j"] - first) < 1e-9, first


def test_an_idle_datacentre_is_charged_its_floor_and_no_more():
    """Never dispatched to. It still drew idle and host for the whole window."""
    d = dc(nodes=2)
    d.compute_energy_from_duration(10.0)
    s = d.stats()
    gs = engines(d)
    want_idle = sum(g._idle_power_w for g in gs) * 10.0
    want_host = len(d.nodes) * HOST_POWER_W * 10.0
    assert abs(s["total_idle_j"] - want_idle) < 1e-3 * want_idle, (s["total_idle_j"], want_idle)
    assert abs(s["total_host_j"] - want_host) < 1e-3 * want_host, (s["total_host_j"], want_host)
    assert abs(s["total_energy_j"] - (want_idle + want_host)) < 1e-3 * (want_idle + want_host)


def test_the_window_before_the_first_request_is_charged():
    """The head gap: an engine idle before its first mark still drew power."""
    d = dc(nodes=1, cards=2)
    t0 = time.monotonic()
    time.sleep(0.4)                      # powered, empty
    drive(d, [(128, 4)] * 4)
    t1 = time.monotonic()
    d.compute_energy(t0, t1)
    head_only = dc(nodes=1, cards=2)
    head_only.compute_energy(t0, t1)     # same window, no work at all
    busy = d.stats()["total_energy_j"]
    floor = head_only.stats()["total_energy_j"]
    assert busy > floor, (busy, floor)
    # and the busy one is not missing the head: its idle must cover the
    # 0.4 s nothing was running, so it cannot be far below the pure floor's.
    assert d.stats()["total_idle_j"] > 0.3 * head_only.stats()["total_idle_j"], (
        d.stats()["total_idle_j"], head_only.stats()["total_idle_j"])


def test_time_at_bs_profile_agrees_with_the_engines():
    d = dc(nodes=2)
    _, t0, t1 = drive(d, [(256, 8)] * 16)
    d.compute_energy(t0, t1)
    prof = d.time_at_bs_profile()
    direct = {}
    for g in engines(d):
        for bs, secs in g.time_at_bs_snapshot().items():
            direct[int(bs)] = direct.get(int(bs), 0.0) + secs
    assert set(prof) == set(direct), (sorted(prof), sorted(direct))
    for bs in prof:
        assert abs(prof[bs] - direct[bs]) < 1e-9, (bs, prof[bs], direct[bs])
    assert abs(sum(prof.values()) - sum(direct.values())) < 1e-9


# ── carbon ───────────────────────────────────────────────────────────────

def test_carbon_is_energy_times_pue_times_ci():
    d = dc(nodes=2, pue=1.35, ci=512.0)
    _, t0, t1 = drive(d, [(256, 8)] * 12)
    d.compute_energy(t0, t1)
    e = d.stats()["total_energy_j"]
    want = e * 1.35 * 512.0 / 3_600_000.0
    assert abs(d.carbon_emissions_g() - want) < 1e-9, (d.carbon_emissions_g(), want)


def test_pue_scales_carbon_and_leaves_energy_alone():
    """Two identical sites at different PUE: same joules, carbon in the ratio."""
    a, b = dc(nodes=2, pue=1.1), dc(nodes=2, pue=1.6)
    a.compute_energy_from_duration(5.0)
    b.compute_energy_from_duration(5.0)
    ea, eb = a.stats()["total_energy_j"], b.stats()["total_energy_j"]
    assert abs(ea - eb) < 1e-6, (ea, eb)
    ratio = b.carbon_emissions_g() / a.carbon_emissions_g()
    assert abs(ratio - 1.6 / 1.1) < 1e-9, ratio


def test_carbon_is_zero_before_energy_is_computed():
    """PollStats reports it every epoch; it is only meaningful after finalize."""
    d = dc(nodes=2)
    drive(d, [(128, 4)] * 8)
    assert d.carbon_emissions_g() == 0.0
    assert d.stats()["total_energy_j"] == 0.0


# ── layering ─────────────────────────────────────────────────────────────

def test_the_datacentre_reaches_no_further_than_its_nodes():
    """Static check: no Datacenter method may name an engine-only member."""
    import ast
    import inspect
    import datacenter.meta_dc as mod

    tree = ast.parse(inspect.getsource(mod))
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Datacenter")
    engine_only = {
        "inference_engines", "active_count", "pending_count", "prefill_count",
        "decoding_count", "energy_j", "idle_energy_j", "host_energy_j",
        "time_at_bs_snapshot", "_time_series", "kv_tokens", "process",
        "engine_faults", "kv_utilisation", "_get_kv_capacity_tokens",
    }
    breaches = [(f.name, a.attr, a.lineno)
                for f in cls.body if isinstance(f, ast.FunctionDef)
                for a in ast.walk(f)
                if isinstance(a, ast.Attribute) and a.attr in engine_only]
    assert not breaches, breaches


def test_every_in_flight_prompt_is_in_exactly_one_token_sum():
    """The accounting that was broken until 2026-09-03.

    A request passes through three stages and each must carry its prompt
    somewhere:

        waiting on the semaphore   pending_requests    total_pending_isl
        admitted, awaiting prefill prefilling_requests total_prefill_isl
        prefilled, decoding        decoding_requests   total_active_isl

    The middle row used to have no token sum at all: at admission the request
    left total_pending_isl and it joined total_active_isl only once its
    prefill landed, so in between its prompt was in none of them. The TTFT
    estimator needs exactly that volume -- it is what a new arrival's prefill
    fuses with -- and could not see it, which is why it certified 85.5% of
    placements as inside a 3000 ms budget when 25.8% were.

    What is checked is the weaker but reliable form: a set holding requests
    must report tokens, and an empty set must report none. Comparing against
    the exact dispatched ISLs would race the sampler.
    """
    d = dc(nodes=2, cards=4, max_bs=8)
    isls = [512, 1024, 2048, 4096]
    bad, checked = [], 0
    stop = threading.Event()

    def load(k):
        for i in range(6):
            if stop.is_set():
                return
            d.dispatch(isls[(k + i) % 4], 6, dataset="cnn")

    ts = [threading.Thread(target=load, args=(k,), daemon=True) for k in range(8)]
    for t in ts:
        t.start()
    deadline = time.monotonic() + 120
    while any(t.is_alive() for t in ts) and time.monotonic() < deadline:
        s = d.stats()
        if (s["total_pending_requests"] + s["total_prefilling_requests"]
                + s["total_decoding_requests"]) > 0:
            checked += 1
            for cnt, isl, name in (
                    (s["total_pending_requests"], s["total_pending_isl"], "pending"),
                    (s["total_prefilling_requests"], s["total_prefill_isl"], "prefill"),
                    (s["total_decoding_requests"], s["total_active_isl"], "active")):
                if (cnt == 0) != (isl == 0):
                    bad.append(f"{name}: {cnt} requests but {isl} tokens")
        time.sleep(0.01)
    stop.set()
    for t in ts:
        t.join(timeout=60)

    assert checked > 20, f"only {checked} in-flight samples; the probe saw nothing"
    assert not bad, bad[:3]
    f = d.stats()
    assert f["total_prefill_isl"] == 0, f["total_prefill_isl"]
    assert f["total_pending_isl"] == 0 and f["total_active_isl"] == 0, f


def test_the_prefill_token_sum_tracks_what_is_in_prefill():
    # The direct check: n requests of the same length in prefill report n x
    # that length. A count alone cannot price a prefill; tokens can.
    d = dc(nodes=1, cards=2, max_bs=32)
    seen = []
    stop = threading.Event()

    def load():
        for _ in range(4):
            if stop.is_set():
                return
            d.dispatch(4096, 8, dataset="cnn")

    ts = [threading.Thread(target=load, daemon=True) for _ in range(4)]
    for t in ts:
        t.start()
    deadline = time.monotonic() + 120
    while any(t.is_alive() for t in ts) and time.monotonic() < deadline:
        s = d.stats()
        if s["total_prefilling_requests"] > 0:
            seen.append((s["total_prefilling_requests"], s["total_prefill_isl"]))
        time.sleep(0.01)
    stop.set()
    for t in ts:
        t.join(timeout=60)

    assert seen, "nothing was ever observed in prefill"
    for cnt, isl in seen:
        assert isl == cnt * 4096, f"{cnt} requests of 4096 reported {isl} tokens"


# ══════════════════════════ the RPC surface ════════════════════════════════
#
# Everything above tests the datacentre object. These test the three handlers a
# scheduler and the provisioner actually call, because a correct Datacenter
# behind a handler that drops a field is still a wrong answer on the wire.

from datacenter.server import DatacenterServicer                  # noqa: E402


def servicer(**kw):
    return DatacenterServicer(dc(**kw), verbosity=0)


def test_dispatch_returns_where_it_ran_and_what_it_cost():
    sv = servicer()
    resp = sv.Dispatch(inference_pb2.DispatchRequest(
        req_id="r1", in_tok=128, out_tok=8, dataset="cnn"), None)
    assert resp.req_id == "r1", resp.req_id
    assert resp.dc_id == "t", resp.dc_id
    assert resp.gpu_id, "the response does not say which engine ran it"
    assert resp.service_ms > 0, resp.service_ms
    assert resp.ttft_ms > 0, resp.ttft_ms
    assert resp.finish_ms >= resp.start_ms, (resp.start_ms, resp.finish_ms)


def test_dispatch_blocks_for_about_as_long_as_it_reports():
    # The whole chain is synchronous: a scheduler thread is parked here for the
    # request's life. If the handler returned early the simulated time and the
    # wall clock would part company and every latency upstream would be wrong.
    sv = servicer()
    t0 = time.monotonic()
    resp = sv.Dispatch(inference_pb2.DispatchRequest(
        req_id="r2", in_tok=128, out_tok=32), None)
    wall_ms = (time.monotonic() - t0) * 1000.0
    assert wall_ms >= resp.service_ms * 0.8, \
        f"reported {resp.service_ms:.0f} ms but returned after {wall_ms:.0f} ms"


def test_queue_time_is_reported_separately_from_service():
    # queue_ms is arrival -> admission to prefill; service_ms starts there. They
    # must not overlap, or the paper's user-TTFT composition double counts.
    sv = servicer(nodes=1, cards=2, max_bs=1)      # one engine, one slot
    out = {}

    def one(i):
        out[i] = sv.Dispatch(inference_pb2.DispatchRequest(
            req_id=f"q{i}", in_tok=256, out_tok=16), None)

    ts = [threading.Thread(target=one, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    assert len(out) == 3, out
    assert max(r.queue_ms for r in out.values()) > 0, \
        "three requests through one slot and nothing queued"


def test_pollstats_reports_the_same_numbers_the_datacentre_holds():
    sv = servicer()
    sv.Dispatch(inference_pb2.DispatchRequest(req_id="p0", in_tok=64,
                                              out_tok=4), None)
    resp = sv.PollStats(inference_pb2.PollStatsRequest(src="t", epoch=9), None)
    s = sv.dc.stats()
    assert resp.epoch == 9, resp.epoch
    assert resp.dc_id == "t", resp.dc_id
    assert resp.num_gpus == s["num_gpus"], (resp.num_gpus, s["num_gpus"])
    assert resp.num_nodes == s["num_nodes"], resp.num_nodes
    assert resp.active_requests == s["total_active_requests"]
    assert resp.pending_requests == s["total_pending_requests"]
    assert resp.carbon_intensity == sv.dc.carbon_intensity
    assert resp.pue == sv.dc.pue and resp.pue > 0, (resp.pue, sv.dc.pue)


def test_pollstats_carries_the_load_fields_while_work_is_running():
    # A field the handler forgets reads as 0 upstream, which is a plausible
    # number rather than an error -- the failure mode is a policy silently
    # scoring against an empty datacentre. So poll while requests are actually
    # in flight and require the numbers to be there.
    sv = servicer(nodes=2, cards=4, max_bs=32)
    stop = threading.Event()

    def load():
        while not stop.is_set():
            sv.Dispatch(inference_pb2.DispatchRequest(
                req_id="L", in_tok=512, out_tok=64), None)

    ts = [threading.Thread(target=load, daemon=True) for _ in range(8)]
    for t in ts:
        t.start()
    try:
        seen = {}
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and len(seen) < 4:
            r = sv.PollStats(inference_pb2.PollStatsRequest(src="t", epoch=1),
                             None)
            for f in ("active_requests", "total_active_isl",
                      "decoding_requests", "total_remaining_osl"):
                if getattr(r, f) > 0:
                    seen[f] = getattr(r, f)
            time.sleep(0.05)
    finally:
        stop.set()
        for t in ts:
            t.join(timeout=30)

    for f in ("active_requests", "total_active_isl", "decoding_requests",
              "total_remaining_osl"):
        assert f in seen, \
            f"PollStats never reported {f} while 8 threads were dispatching"


def test_pollstats_carries_the_energy_split_after_finalising():
    sv = servicer()
    sv.Dispatch(inference_pb2.DispatchRequest(req_id="s1", in_tok=256,
                                              out_tok=16), None)
    sv.FinalizeEnergy(inference_pb2.FinalizeEnergyRequest(
        sim_duration_s=10.0), None)
    r = sv.PollStats(inference_pb2.PollStatsRequest(src="t", epoch=1), None)
    st = sv.dc.stats()
    assert r.idle_energy_j > 0, "the idle floor is not reported"
    assert r.host_energy_j > 0, "host power is not reported"
    assert abs(r.idle_energy_j - st["total_idle_j"]) < 1e-6
    assert abs(r.host_energy_j - st["total_host_j"]) < 1e-6
    assert r.idle_energy_j + r.host_energy_j <= r.total_energy_j + 1e-6, \
        "the floor exceeds the total"


def test_polling_an_idle_datacentre_reports_it_idle():
    resp = servicer().PollStats(
        inference_pb2.PollStatsRequest(src="t", epoch=1), None)
    assert resp.active_requests == 0 and resp.pending_requests == 0, resp
    assert resp.num_gpus > 0, "an idle datacentre reported no engines"


def test_energy_is_zero_until_it_is_finalised():
    # The window is not known until the run ends, so the handler must not
    # invent one. A non-zero reading here would be integrated over nothing.
    sv = servicer()
    sv.Dispatch(inference_pb2.DispatchRequest(req_id="e0", in_tok=64,
                                              out_tok=4), None)
    resp = sv.PollStats(inference_pb2.PollStatsRequest(src="t", epoch=1), None)
    assert resp.total_energy_j == 0.0, resp.total_energy_j
    assert resp.carbon_emissions_g == 0.0, resp.carbon_emissions_g


def test_finalize_energy_reports_and_then_pollstats_agrees():
    sv = servicer()
    sv.Dispatch(inference_pb2.DispatchRequest(req_id="e1", in_tok=256,
                                              out_tok=16), None)
    fin = sv.FinalizeEnergy(
        inference_pb2.FinalizeEnergyRequest(sim_duration_s=10.0), None)
    assert fin.dc_id == "t", fin.dc_id
    assert fin.total_energy_j > 0, fin.total_energy_j
    assert fin.carbon_emissions_g > 0, fin.carbon_emissions_g
    after = sv.PollStats(inference_pb2.PollStatsRequest(src="t", epoch=2), None)
    assert abs(after.total_energy_j - fin.total_energy_j) < 1e-6, \
        "the two handlers disagree about the same joules"
    assert abs(after.carbon_emissions_g - fin.carbon_emissions_g) < 1e-9


def test_finalising_a_longer_window_costs_more_idle():
    # The floor is real power for real seconds; a longer window must charge
    # more even with identical work.
    def energy(window):
        sv = servicer()
        sv.Dispatch(inference_pb2.DispatchRequest(req_id="w", in_tok=64,
                                                  out_tok=4), None)
        return sv.FinalizeEnergy(inference_pb2.FinalizeEnergyRequest(
            sim_duration_s=window), None).total_energy_j
    short, long = energy(5.0), energy(20.0)
    assert long > short, (short, long)


def test_finalising_twice_gives_the_same_answer():
    # scripts/sweep/collect_energy.py calls it on every site and may be run
    # more than once against a live fleet.
    sv = servicer()
    sv.Dispatch(inference_pb2.DispatchRequest(req_id="t1", in_tok=64,
                                              out_tok=8), None)
    a = sv.FinalizeEnergy(inference_pb2.FinalizeEnergyRequest(
        sim_duration_s=10.0), None).total_energy_j
    b = sv.FinalizeEnergy(inference_pb2.FinalizeEnergyRequest(
        sim_duration_s=10.0), None).total_energy_j
    assert abs(a - b) < 1e-6, (a, b)


def test_the_handler_serves_concurrent_dispatches():
    sv = servicer(nodes=2, cards=4, max_bs=32)
    out, errs = {}, []

    def one(i):
        try:
            out[i] = sv.Dispatch(inference_pb2.DispatchRequest(
                req_id=f"c{i}", in_tok=128, out_tok=8), None)
        except Exception as e:                                    # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=one, args=(i,)) for i in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=120)
    assert not errs, errs[:2]
    assert len(out) == 16, len(out)
    assert len({r.gpu_id for r in out.values()}) > 1, \
        "16 concurrent requests all ran on one engine"
    assert sv.dc.stats()["total_active_requests"] == 0, "counters did not drain"


# ───────────────────── prefill_duty is windowed ─────────────────────────────
# It used to be cumulative since the engine started, so one reading averaged
# over everything before it -- the ramp most of all. Measured on
# carbon_aware:0.0 at 1200 req/s the cumulative ratio reads 0.56-0.58 where the
# plateau is 0.65-0.68, and every consumer divides by (1 - duty), so the SLA
# filter and the three ILPs were all 20-30% optimistic on TPOT.

def _duty_probe(age_s, window_s=10.0):
    """A stand-in carrying only what prefill_duty touches."""
    import collections
    from datacenter.inference_engine import GPU

    class P:
        DUTY_WINDOW_S = window_s
        prefill_duty = GPU.prefill_duty

        def __init__(self):
            self._lock = threading.Lock()
            self._duty_samples = collections.deque()
            self._t_origin = time.monotonic() - age_s
            self._prefill_busy_s = 0.0

        def pass_done(self, at, busy):
            """One prefill pass, as the engine loop records it."""
            self._prefill_busy_s += busy
            self._duty_samples.append((at, self._prefill_busy_s))
            edge = at - self.DUTY_WINDOW_S
            while (len(self._duty_samples) > 1
                   and self._duty_samples[1][0] <= edge):
                self._duty_samples.popleft()

    return P()


def test_prefill_duty_reads_the_window_not_the_whole_run():
    # 90 s near-idle, then 10 s busy. Cumulative says 0.26; what a scheduler
    # needs to know is that the site is at 0.80 right now.
    p = _duty_probe(age_s=100.0)
    t0 = p._t_origin
    for i in range(1, 101):
        p.pass_done(t0 + i, 0.2 if i <= 90 else 0.8)
    assert abs(p._prefill_busy_s / 100.0 - 0.26) < 1e-6, "fixture drifted"
    d = p.prefill_duty()
    assert abs(d - 0.80) < 0.02, f"windowed duty read {d}, not the last 10 s"


def test_prefill_duty_falls_when_the_site_goes_quiet():
    # The mirror case, and the one a cumulative counter cannot do at all: busy
    # first, idle since. A filter that cannot see a site recover will keep
    # rejecting it.
    p = _duty_probe(age_s=100.0)
    t0 = p._t_origin
    for i in range(1, 91):
        p.pass_done(t0 + i, 0.8)
    d = p.prefill_duty()
    assert d < 0.05, f"duty stayed at {d} with no prefill for 10 s"


def test_prefill_duty_of_a_young_engine_is_its_whole_life():
    # Shorter-lived than the window: the denominator must be the engine's age,
    # not the window, or a site that just came up reads artificially idle.
    p = _duty_probe(age_s=3.0)
    t0 = p._t_origin
    for i in (1, 2, 3):
        p.pass_done(t0 + i, 0.5)
    d = p.prefill_duty()
    assert abs(d - 0.50) < 0.02, f"young engine read {d}, not its own average"


def test_duty_samples_do_not_grow_without_bound():
    # One entry per prefill pass, pruned to the window plus the one before it.
    # At a few passes a second over a long run this is the difference between
    # a bounded deque and a leak.
    p = _duty_probe(age_s=600.0)
    t0 = p._t_origin
    for i in range(1, 6001):
        p.pass_done(t0 + i * 0.1, 0.05)
    assert len(p._duty_samples) <= 120, len(p._duty_samples)


def test_prefill_duty_is_still_capped_below_one():
    # The caller divides by (1 - duty); a saturated window must not produce an
    # infinite estimate.
    p = _duty_probe(age_s=100.0)
    t0 = p._t_origin
    for i in range(1, 101):
        p.pass_done(t0 + i, 1.0)
    assert p.prefill_duty() <= 0.95, p.prefill_duty()


def test_the_kv_queue_terms_combine_engine_forecasts():
    from datacenter.meta_dc import kv_queue_terms
    assert kv_queue_terms([]) == (0.0, 0.0, 0.0), "no known cache, no wait"
    # (ahead_ms, room_tokens, ms_per_token, ref_isl) per engine. A mean-length
    # prompt waits 1000 + 0 on the first and 0 + 500 * 4 = 2000 on the second.
    f = [(1000.0, 800.0, 2.0, 600.0), (0.0, 100.0, 4.0, 600.0)]
    assert kv_queue_terms(f, "min") == (1000.0, 800.0, 2.0)
    assert kv_queue_terms(f, "mean") == (500.0, 450.0, 3.0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except BaseException as e:                                # noqa: BLE001
            # Anything else used to end the run at the first case, silently:
            # the remaining tests never reported and the count still looked
            # plausible.
            fails += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
