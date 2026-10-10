#!/usr/bin/env python3
"""Invariants the inference engine must keep, whatever its concurrency
architecture is.

Written before replacing thread-per-request with a step loop. These are not
accuracy checks -- accuracy is measured by replaying the real traces in
testbed/phase2_endtoend/within_node_sim.py against the hardware runs, which
takes minutes. These run in seconds and catch the things a rewrite breaks
first, so the slow check is only worth starting once they pass.

    python scripts/test/test_inference_engine.py
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datacenter.inference_engine import GPU
from datacenter.meta_dc import Datacenter                    # noqa: E402
from datacenter.perf_models.analytical.latency import AnalyticalLatencyModel  # noqa: E402

MODEL, TP = "llama3_8B", 1


def engine(max_bs=64, n_gpus=1):
    return Datacenter("t", 1, n_gpus, None, tp_size=TP, max_batch_size=max_bs,
                      analytical_model=AnalyticalLatencyModel(MODEL, tp=TP),
                      latency_mode="analytical", model_name=MODEL)


def drive(dc, reqs, spacing=0.0):
    """Send reqs as (isl, osl), spacing seconds apart, and collect the results."""
    out, lk = [], threading.Lock()

    def one(i, isl, osl):
        if spacing:
            time.sleep(i * spacing)
        r = dc.dispatch(isl, osl)
        with lk:
            out.append(r)

    ths = [threading.Thread(target=one, args=(i, a, b))
           for i, (a, b) in enumerate(reqs)]
    t0 = time.monotonic()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return out, t0, time.monotonic()


def test_tokens_are_conserved():
    """Every requested output token is emitted exactly once, and nothing leaks.

    The counters are what the energy model divides by to get its step count, so
    a leak here is not a cosmetic bug -- it silently rescales energy.
    """
    dc = engine()
    reqs = [(512, 24)] * 12
    drive(dc, reqs)
    g = dc.nodes[0].inference_engines[0]
    assert g._cum_generated == sum(o for _, o in reqs), g._cum_generated
    assert g.kv_tokens == 0, g.kv_tokens
    for name in ("_active_count", "_decoding_count", "_prefill_count",
                 "_pending_count", "_total_active_isl",
                 "_total_generated_tokens"):
        assert getattr(g, name) == 0, (name, getattr(g, name))


def test_counters_are_consistent_while_running():
    """decoding <= active, and neither goes negative, at every instant."""
    dc = engine()
    g = dc.nodes[0].inference_engines[0]
    bad, stop = [], threading.Event()

    def watch():
        while not stop.is_set():
            a, d, p = g._active_count, g._decoding_count, g._prefill_count
            if d > a or min(a, d, p) < 0:
                bad.append((a, d, p))
            time.sleep(0.001)

    threading.Thread(target=watch, daemon=True).start()
    drive(dc, [(512, 20)] * 16, spacing=0.004)
    stop.set()
    assert not bad, bad[:5]


def test_energy_stays_under_the_device_ceiling():
    """Mean power over the run cannot exceed the engine's TDP.

    The cheapest check that the energy and latency paths agree: energy is
    counted per step from cache growth, elapsed time comes from the latency
    model, and if they disagree the quotient leaves the envelope the hardware
    can physically occupy.
    """
    dc = engine()
    _, t0, t1 = drive(dc, [(1024, 32)] * 16)
    dc.compute_energy(t0, t1)
    g = dc.nodes[0].inference_engines[0]
    watts = g.energy_j() / (t1 - t0)
    ceiling = 400.0 * TP + g._host_power_share_w
    assert 0 < watts < ceiling, (watts, ceiling)


def test_long_prompts_are_split_at_the_cap():
    """A prompt longer than the prefill budget is processed over several steps.

    The engine bounds a forward pass by max_num_batched_tokens, and the energy
    model was fitted up to 32768 tokens in a step; letting one prompt through
    whole would extrapolate past the measured envelope.
    """
    dc = engine()
    g = dc.nodes[0].inference_engines[0]
    n = g.max_prefill_tokens * 2 + 1696
    drive(dc, [(n, 4)])
    tokens = sum(e[2] for e in g._prefill_events)
    assert tokens == n, (tokens, n)
    assert len(g._prefill_events) >= 3, len(g._prefill_events)


def test_step_time_matches_the_model_when_uncontended():
    """One request alone should see the model's own numbers.

    With nothing else running there is no queueing, no batching and no prefill
    interrupting decode, so any gap here is the harness rather than the load.
    """
    dc = engine()
    m = AnalyticalLatencyModel(MODEL, tp=TP)
    isl, osl = 512, 40
    out, _, _ = drive(dc, [(isl, osl)])
    r = out[0]
    want_ttft = m.ttft_ms(isl, bs=1)
    want_tpot = m.tpot_ms(1, total_kv=isl + osl / 2)
    assert abs(r["ttft_ms"] - want_ttft) / want_ttft < 0.25, (r["ttft_ms"], want_ttft)
    assert abs(r["avg_tpot_ms"] - want_tpot) / want_tpot < 0.15, (r["avg_tpot_ms"], want_tpot)


def test_admission_bound_is_respected():
    """No more requests decode at once than max_batch_size allows."""
    cap = 4
    dc = engine(max_bs=cap)
    g = dc.nodes[0].inference_engines[0]
    peak, stop = [0], threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], g._active_count)
            time.sleep(0.001)

    threading.Thread(target=watch, daemon=True).start()
    drive(dc, [(256, 16)] * 20, spacing=0.002)
    stop.set()
    assert peak[0] <= cap, peak[0]


def test_requests_spread_across_engine_slots():
    """A node with several slots uses them; work does not pile onto one."""
    dc = engine(n_gpus=3)
    drive(dc, [(512, 16)] * 12)
    served = [g._total_requests for g in dc.nodes[0].inference_engines]
    assert min(served) > 0, served
    assert max(served) - min(served) <= 1, served


def test_kv_capacity_is_the_headroom_after_weights():
    """capacity = KV_HEADROOM * (tp x HBM - weight_bytes) / kv_bytes_per_token

    A deliberate idealisation, and more generous than the hardware: the
    profiled capacities in KV_CAPACITY_MEASURED come in 20-48% lower because a
    real vLLM reserves memory for activations, CUDA graph buffers and
    fragmentation that this does not model. Pinned so the choice stays visible
    -- and because a larger cache widens the KV region the decode fit does not
    cover, which is the term the TTFT estimator now extrapolates.
    """
    from datacenter.inference_engine import GPU
    from datacenter.perf_models.model_specs import weight_bytes, kv_bytes_per_token
    for model, tp in (("llama3_8B", 1), ("llama3_8B", 2), ("qwen2.5_7B", 2)):
        g = GPU(f"probe-{model}-{tp}", None, max_batch_size=32,
                analytical_model=AnalyticalLatencyModel(model, tp=tp),
                latency_mode="analytical", model_name=model, tp_size=tp,
                max_model_len=32768)
        free = tp * GPU.HBM_BYTES_PER_GPU - weight_bytes(model)
        want = int(free * GPU.KV_HEADROOM / kv_bytes_per_token(model))
        assert g.kv_capacity_tokens == want, (model, tp, g.kv_capacity_tokens, want)
        measured = GPU.KV_CAPACITY_MEASURED.get((model, tp))
        assert g.kv_capacity_tokens > measured, \
            (f"{model} TP={tp}: the idealisation is no longer above the "
             f"profiled {measured:,}; if that is intended, update this test "
             f"and say so in the paper")


def test_measured_kv_capacity_is_selectable_and_refuses_an_unmeasured_pair():
    """KV_CAPACITY_SOURCE=measured swaps in vLLM's profiled capacity.

    Long prompts are where it decides the answer: Mixtral TP=4 fills at ~26
    sequences on the measured 293,984 tokens and ~51 on the derived 568,227.
    A pair with no measurement must fail loudly, not fall back quietly -- and
    with the variable unset nothing changes, so every earlier run stays valid.
    """
    import os
    from datacenter.inference_engine import GPU
    from datacenter.perf_models.model_specs import weight_bytes, kv_bytes_per_token

    def build():
        return GPU("probe-mixtral-4", None, max_batch_size=32,
                   analytical_model=AnalyticalLatencyModel("mixtral_8x7b", tp=4),
                   latency_mode="analytical", model_name="mixtral_8x7b",
                   tp_size=4, max_model_len=32768)

    old_env = os.environ.get("KV_CAPACITY_SOURCE")
    saved = dict(GPU.KV_CAPACITY_MEASURED)
    try:
        os.environ["KV_CAPACITY_SOURCE"] = "measured"
        assert build().kv_capacity_tokens == 293_984, build().kv_capacity_tokens

        GPU.KV_CAPACITY_MEASURED.pop(("mixtral_8x7b", 4))
        try:
            build()
        except ValueError:
            pass
        else:
            raise AssertionError("an unmeasured (model, TP) must refuse to start")
        GPU.KV_CAPACITY_MEASURED.update(saved)

        os.environ.pop("KV_CAPACITY_SOURCE")
        free = 4 * GPU.HBM_BYTES_PER_GPU - weight_bytes("mixtral_8x7b")
        want = int(free * GPU.KV_HEADROOM / kv_bytes_per_token("mixtral_8x7b"))
        assert build().kv_capacity_tokens == want, (build().kv_capacity_tokens, want)
    finally:
        GPU.KV_CAPACITY_MEASURED.clear()
        GPU.KV_CAPACITY_MEASURED.update(saved)
        if old_env is None:
            os.environ.pop("KV_CAPACITY_SOURCE", None)
        else:
            os.environ["KV_CAPACITY_SOURCE"] = old_env


def test_an_unknown_model_gets_no_cache_gate_rather_than_an_infinite_one():
    from datacenter.inference_engine import GPU
    g = GPU("probe-unknown", None, max_batch_size=32,
            analytical_model=AnalyticalLatencyModel("llama3_8B", tp=2),
            latency_mode="analytical", model_name="", tp_size=2,
            max_model_len=32768)
    assert g.kv_capacity_tokens in (None, 0), g.kv_capacity_tokens


def test_the_async_path_answers_the_same_as_the_calibrated_one():
    """`process_async` is a copy of `process`, so it can drift from it.

    `process` is the path the latency model was fitted through, and the async
    copy exists only so a queued request costs a coroutine instead of one of
    the datacentre's worker threads. Everything it reports must still match.
    Timing fields cannot match exactly -- this is a real-time simulation -- so
    they are compared as a ratio, and the structural fields exactly.
    """
    reqs = [(600, 40), (900, 60), (1200, 30), (700, 50)]

    def sync_run():
        g = GPU("sync", None, max_batch_size=8,
                analytical_model=AnalyticalLatencyModel(MODEL, tp=TP),
                latency_mode="analytical", model_name=MODEL, tp_size=TP)
        out, lk = [], threading.Lock()

        def one(isl, osl):
            r = g.process(isl, osl)
            with lk:
                out.append(r)
        ths = [threading.Thread(target=one, args=r) for r in reqs]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        return out

    async def async_run():
        import asyncio
        g = GPU("async", None, max_batch_size=8,
                analytical_model=AnalyticalLatencyModel(MODEL, tp=TP),
                latency_mode="analytical", model_name=MODEL, tp_size=TP)
        return await asyncio.gather(*[g.process_async(a, b) for a, b in reqs])

    import asyncio
    a = sorted(sync_run(), key=lambda r: r["service_ms"])
    b = sorted(asyncio.run(async_run()), key=lambda r: r["service_ms"])

    assert len(a) == len(b) == len(reqs), (len(a), len(b))
    assert set(a[0]) == set(b[0]), (
        f"the two paths report different columns: "
        f"{set(a[0]) ^ set(b[0])}")

    # prefill_bs is NOT compared for equality, and asserting that was a flaw in
    # this test rather than in the engine: it counts how many requests the
    # engine fused into one prefill pass, which depends on how closely their
    # arrivals coincide in real time. The synchronous path staggers them across
    # thread starts, the async path submits them in one loop iteration and
    # fuses more. It differs between repeats of a single path too -- caught by
    # running this three times, not once. What must hold is that both stay in
    # range.
    for r in a + b:
        assert 1 <= r["prefill_bs"] <= len(reqs), r["prefill_bs"]

    # Medians rather than per-request pairs, for the same reason: different
    # fusion moves an individual request's service time around. A real
    # divergence between the two paths would move the middle of the
    # distribution, which this still catches.
    def mid(rows, k):
        v = sorted(r[k] for r in rows)
        return v[len(v) // 2]

    for k in ("service_ms", "ttft_ms", "avg_tpot_ms"):
        lo, hi = sorted((mid(a, k), mid(b, k)))
        assert hi <= lo * 1.5 + 5.0, \
            f"{k} median: sync {mid(a, k):.2f} vs async {mid(b, k):.2f}"


def test_a_queued_request_is_charged_to_queue_ms_on_the_async_path():
    """The reason the async path exists.

    With more requests than seats, the ones that wait must have that wait in
    `queue_ms`. If it lands anywhere else it lands nowhere -- that is exactly
    the hole `dc_queue` had to be added to `where_time_goes.py` to expose,
    where a concentrating policy spent 94% of a request's life invisibly.
    """
    import asyncio

    async def go():
        g = GPU("q", None, max_batch_size=2,
                analytical_model=AnalyticalLatencyModel(MODEL, tp=TP),
                latency_mode="analytical", model_name=MODEL, tp_size=TP)
        # Eight requests, two seats: six of them have to wait for one.
        return await asyncio.gather(*[g.process_async(800, 60)
                                      for _ in range(8)])

    out = asyncio.run(go())
    queued = [r for r in out if r["queue_ms"] > 1.0]
    assert len(queued) >= 5, \
        f"only {len(queued)} of 8 recorded a seat wait with 2 seats"
    assert max(r["queue_ms"] for r in out) > 10.0, \
        "the seat wait was recorded as ~zero, so it went somewhere unmeasured"


def test_a_blocked_prefill_owes_a_decode_run():
    """DECODE_STEPS_PER_PREFILL must actually reorder steps when the gate bites.

    It was added, run at 3200 req/s, and changed NOTHING -- prefill_duty came
    back 0.509 against 0.509. The reading was that the cache gate never fires
    at that load, so the debt is never raised; but that is an inference about
    a run, not evidence that the mechanism works. So force the gate here with a
    cache far too small for the batch, and check the engine spends more of its
    steps decoding with the cadence on than with it off.
    """
    from datacenter.inference_engine import GPU

    def run(n_steps):
        g = GPU("cad", None, max_batch_size=32,
                analytical_model=AnalyticalLatencyModel(MODEL, tp=TP),
                latency_mode="analytical", model_name=MODEL, tp_size=TP)
        # Small enough that a handful of prompts fills it, so prefill blocks
        # and only decode can free it.
        g.kv_capacity_tokens = 3000
        g._decode_steps_per_prefill = n_steps
        out, lk = [], threading.Lock()

        def one(i):
            r = g.process(700, 40)
            with lk:
                out.append(r)
        ths = [threading.Thread(target=one, args=(i,)) for i in range(12)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(timeout=120)
        return g

    off = run(0)
    on = run(8)

    assert off.kv_blocked_steps > 0, (
        "the cache gate never fired, so this test cannot show anything -- "
        "lower kv_capacity_tokens")
    # The cadence pays several decodes per block instead of one, so the same
    # work is reached with fewer blocked attempts.
    assert on.kv_blocked_steps < off.kv_blocked_steps, (
        f"blocked steps {on.kv_blocked_steps} with the cadence vs "
        f"{off.kv_blocked_steps} without: the debt is not being raised")


def test_the_kv_gate_wait_is_queue_time_not_ttft():
    """A prompt held at the KV gate is queueing, and is reported as such.

    Seats are plentiful here (32 for 12 requests), so under the old boundary
    queue_ms was ~0 and the whole cache wait sat inside ttft_ms. With queue
    ending at prefill admission, ttft_ms is the prompt's own prefill step and
    the wait shows up in queue_ms.
    """
    from datacenter.inference_engine import GPU
    g = GPU("kvq", None, max_batch_size=32,
            analytical_model=AnalyticalLatencyModel(MODEL, tp=TP),
            latency_mode="analytical", model_name=MODEL, tp_size=TP)
    g.kv_capacity_tokens = 3000
    out, lk = [], threading.Lock()

    def one(i):
        r = g.process(700, 40)
        with lk:
            out.append(r)
    ths = [threading.Thread(target=one, args=(i,)) for i in range(12)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout=120)

    assert g.kv_blocked_steps > 0, "the cache gate never fired"
    assert len(out) == 12, len(out)
    assert max(r["queue_ms"] for r in out) > 100.0, (
        "prompts waited on a full cache with seats free, and none of it "
        f"reached queue_ms: {sorted(round(r['queue_ms']) for r in out)}")
    # Three 700-token prompts overrun the 2048-token pass, so one is chunked
    # across two prefill steps: ttft is at most a couple of steps -- never the
    # cache wait, which runs to seconds here.
    longest_step = max(r["step_ms"] for r in out)
    for r in out:
        assert r["ttft_ms"] <= 2.0 * longest_step + 50.0, (
            f"ttft {r['ttft_ms']:.0f} ms against a {longest_step:.0f} ms "
            "longest prefill step: the gate wait is still inside ttft_ms")
        assert r["start_ms"] + r["service_ms"] <= r["finish_ms"] + 1e-6, r


def test_the_kv_forecast_winds_the_queue_in_finish_order():
    """Stages 1-2 by hand, and OSL_SOURCE deciding who finishes first.

    Cache 2500. Running: A (1000 + 90 of 100 out) and B (1000 + 10 of 500),
    so 2100 used and 400 free. Waiting: one 900-token prompt, blocked. A
    finishes first, after 10 steps in which the two grow the cache by 20; it
    hands back 1100, leaving 1480, and the prompt goes in with 580 to spare.
    """
    from datacenter.inference_engine import GPU, _Req

    def engine(pred_a):
        g = GPU("fc", None, max_batch_size=32,
                analytical_model=AnalyticalLatencyModel(MODEL, tp=TP),
                latency_mode="analytical", model_name=MODEL, tp_size=TP)
        g.kv_capacity_tokens = 2500
        a, b = _Req(1000, 100, "", 0.0), _Req(1000, 500, "", 0.0)
        a.generated, b.generated = 90, 10
        a.pred_osl, b.pred_osl = pred_a, 500
        g._running = [a, b]
        g._total_active_isl, g._total_generated_tokens = 2000, 100
        g._waiting = [_Req(900, 50, "", 0.0)]
        g._total_requests, g._cum_isl = 3, 2900
        return g

    step_s = GPU.STEP_SCALE * engine(0)._get_tpot(2, total_kv=2110,
                                                   avg_isl=1055)
    pre_s = engine(0)._forecast_prefill_s(900)
    ahead, room, per_tok, ref = engine(0).kv_forecast()
    assert room == 580.0, room
    assert abs(ahead - 1000 * (10 * step_s + pre_s)) < 0.05 * ahead, \
        (ahead, step_s, pre_s)
    assert per_tok > 0 and ref > 0, (per_tok, ref)
    # Told A has 310 tokens left rather than 10, the forecast waits for it.
    GPU.OSL_SOURCE = "predicted"
    try:
        later, room_p, _, _ = engine(400).kv_forecast()
    finally:
        GPU.OSL_SOURCE = "oracle"
    assert later > 10 * ahead, (later, ahead)


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
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
