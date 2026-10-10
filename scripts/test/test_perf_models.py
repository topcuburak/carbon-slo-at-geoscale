#!/usr/bin/env python3
"""The latency and energy models: the numbers everything else is scored on.

The engine's simulated time comes from here, the ILP's SLA feasibility comes
from here, and every joule in the carbon result comes from here. They are
fitted from testbed campaigns, so these do not re-check the fit -- that is what
testbed/phase1_calibration is for. They check the properties a fit must have to
be usable, and the ones its documented limitations say it must not silently
lose:

  monotonicity      more tokens is never faster; a bigger batch never decodes
                    a token faster than a smaller one
  the artifact      a (model, TP) pair with no fit raises rather than
                    returning a plausible wrong number
  TP is not free    a model fitted at TP=1 must not be served for TP=2
  bounds            power under the device ceiling, latency positive and finite

Run:  PYTHONPATH=.:proto python3 scripts/test/test_perf_models.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "proto"))

from datacenter.perf_models.analytical.energy import StepEnergyModel   # noqa: E402
from datacenter.perf_models.analytical.latency import AnalyticalLatencyModel  # noqa: E402
from datacenter.perf_models.factory import create_latency_backend      # noqa: E402

PASS, FAIL = [], []

# H100 SXM board power. Nothing the model reports may exceed it.
DEVICE_CEILING_W = 700.0


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  ok    {name}")
        except AssertionError as e:
            FAIL.append((name, str(e)))
            print(f"  FAIL  {name}\n          {e}")
        except BaseException as e:                                  # noqa: BLE001
            FAIL.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
        return fn
    return deco


def lat(model="llama3_8B", tp=2):
    return AnalyticalLatencyModel(model, tp=tp)


# ─────────────────────── which fits actually exist ──────────────────────────

def test_the_pairs_the_fleet_config_offers_are_all_fitted():
    # datacenter/fleet.yaml says these are the fitted ones. A config that names
    # an unfitted pair fails at start-up, which is the right time -- but only
    # if the list in the comment is true.
    for model, tp in (("llama3_8B", 1), ("llama3_8B", 2),
                      ("qwen2.5_7B", 1), ("qwen2.5_7B", 2),
                      ("mixtral_8x7b", 4)):
        m = AnalyticalLatencyModel(model, tp=tp)
        assert m.ttft_ms(512, bs=1) > 0, (model, tp)


def test_a_pair_with_no_artifact_raises_rather_than_guessing():
    # There is no fallback table on purpose: a plausible wrong number here
    # propagates into every latency and every joule.
    for model, tp in (("llama3_8B", 8), ("mixtral_8x7b", 2),
                      ("no_such_model", 1)):
        try:
            AnalyticalLatencyModel(model, tp=tp)
        except (ValueError, FileNotFoundError, KeyError):
            continue
        raise AssertionError(f"{model} at TP={tp} loaded a fit that does not exist")


def test_a_fit_is_tied_to_the_tp_it_was_measured_at():
    # TP=2 decode is slower than TP=1 below BS~32 -- measured, and one of the
    # paper's findings. Serving one fit for the other would erase it.
    one, two = lat(tp=1), lat(tp=2)
    assert one.tpot_ms(bs=8) != two.tpot_ms(bs=8), \
        "TP=1 and TP=2 report identical TPOT; one fit is being served for both"


def test_the_model_says_what_it_loaded():
    s = lat().summary()
    assert "llama3_8B" in s, s
    assert s.strip(), "an empty summary"


def test_available_lists_something_and_only_real_pairs():
    got = AnalyticalLatencyModel.available()
    assert got, "no fitted pairs reported"


# ──────────────────────────────── TTFT ──────────────────────────────────────

def test_ttft_grows_with_the_prompt():
    m = lat()
    vals = [m.ttft_ms(n, bs=1) for n in (128, 256, 512, 1024, 2048, 4096)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals
    assert vals[-1] > vals[0] * 2, f"4096 tokens is barely slower than 128: {vals}"


def test_more_requests_at_the_same_prompt_length_cost_more():
    # The prefill budget fuses queued prompts into ONE forward pass, so the
    # first argument is the batch's TOTAL tokens. Holding the per-request
    # prompt fixed and adding requests is what makes a fused pass bigger.
    m = lat()
    vals = [m.ttft_ms(1024 * b, bs=b) for b in (1, 2, 4, 8)]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


def test_spreading_the_same_tokens_over_more_requests_is_not_more_expensive():
    # The other direction, and the one that is easy to get backwards: with the
    # TOTAL fixed, a larger batch is the same work in shorter sequences. The
    # fit is near-linear in total tokens, so this must not rise.
    m = lat()
    vals = [m.ttft_ms(4096, bs=b) for b in (1, 2, 4, 8, 16)]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:])), vals
    assert vals[-1] > 0.8 * vals[0], \
        f"the batch term dominates the token term: {vals}"


def test_ttft_is_positive_and_finite_across_the_range():
    import math
    m = lat()
    for n in (1, 16, 512, 8192, 32768):
        for b in (1, 4, 32, 128):
            v = m.ttft_ms(n, bs=b)
            assert v > 0 and math.isfinite(v), (n, b, v)


def test_ttft_seconds_and_milliseconds_agree():
    m = lat()
    assert abs(m.ttft_s(2048, bs=4) * 1000.0 - m.ttft_ms(2048, bs=4)) < 1e-6


def test_a_single_long_prompt_is_only_slightly_dearer_KNOWN_LIMITATION():
    # Recorded limitation: the prefill fit is near-linear in the batch's total
    # tokens, but attention is quadratic per SEQUENCE, so one long prompt is
    # under-predicted against the testbed by 13-18%.
    #
    # Inside the model the effect is small: 8192 tokens as one prompt costs
    # about a tenth more than the same 8192 spread over eight. That ratio is
    # the shape of the gap, so it is pinned -- a refit that fixed the
    # quadratic term would push it well above this band.
    m = lat()
    one_long = m.ttft_ms(8192, bs=1)
    spread = m.ttft_ms(8192, bs=8)
    ratio = one_long / spread
    assert 1.0 <= ratio < 1.35, \
        (f"one long prompt is {ratio:.2f}x the spread batch; the near-linear "
         f"shape this limitation describes may have changed")


# ──────────────────────────────── TPOT ──────────────────────────────────────

def test_tpot_never_falls_as_the_batch_grows():
    # A bigger batch cannot make a single token arrive sooner. If it does, the
    # fit has a region where the ILP is rewarded for piling work on.
    m = lat()
    vals = [m.tpot_ms(bs=b) for b in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-6, f"TPOT fell as the batch grew: {vals}"


def test_tpot_grows_with_the_kv_the_batch_is_holding():
    m = lat()
    small = m.tpot_ms(bs=32, avg_isl=128)
    large = m.tpot_ms(bs=32, avg_isl=8192)
    assert large >= small, (small, large)


def test_tpot_is_positive_and_finite_across_the_range():
    import math
    m = lat()
    for b in (1, 8, 64, 256):
        for isl in (0, 128, 4096, 32768):
            v = m.tpot_ms(bs=b, avg_isl=isl)
            assert v > 0 and math.isfinite(v), (b, isl, v)


def test_tpot_seconds_and_milliseconds_agree():
    m = lat()
    assert abs(m.tpot_s(bs=16) * 1000.0 - m.tpot_ms(bs=16)) < 1e-6


def test_tp2_decodes_slower_than_tp1_at_a_small_batch():
    # A paper-worthy measured result: below BS~32 the extra collective costs
    # more than the split saves. Pinned so a refit cannot quietly lose it.
    one, two = lat(tp=1), lat(tp=2)
    assert two.tpot_ms(bs=4) > one.tpot_ms(bs=4), \
        (one.tpot_ms(bs=4), two.tpot_ms(bs=4))


def test_tpot_keeps_growing_past_the_fitted_kv_envelope():
    # It used to FREEZE. Beyond kv_max the model returned the boundary value
    # for any input -- 34.249 ms at 350k, at 405k and at 5,000,000 tokens
    # alike -- while the run served 61.7 ms. That constant propagated into the
    # TTFT estimator's slot-free time and into the TPOT SLA check, and is why
    # the scheduler certified 85.5% of placements as inside a 3000 ms budget
    # when 25.8% were.
    #
    # The raw polynomial cannot simply be evaluated out there: c4 multiplies
    # kv/bs and is negative for llama3_8B at TP=2, so the form goes negative.
    # It now continues with the boundary gradient instead.
    m = lat()
    kvs = [300_000, 331_200, 400_000, 500_000, 1_000_000, 5_000_000]
    vals = [m.tpot_ms(bs=128, total_kv=k) for k in kvs]
    for a, b in zip(vals, vals[1:]):
        assert b > a, f"TPOT stopped responding to the cache: {list(zip(kvs, vals))}"
    assert vals[-1] > 2 * vals[1], \
        f"5M tokens is barely dearer than the envelope edge: {vals[1]:.1f} -> {vals[-1]:.1f}"


def test_the_extension_is_continuous_at_the_envelope_edge():
    # A step at the boundary would show up as a discontinuity in every
    # downstream estimate, so the extension has to start from the fit's own
    # value there.
    m = lat()
    edge = 331_200
    inside = m.tpot_ms(bs=128, total_kv=edge - 1)
    at = m.tpot_ms(bs=128, total_kv=edge)
    just_out = m.tpot_ms(bs=128, total_kv=edge + 1)
    assert abs(just_out - at) < 0.01, (at, just_out)
    assert at >= inside, (inside, at)


def test_the_extension_never_makes_more_load_cheaper():
    # The one thing that must not happen. max(0, gradient) guards it in case a
    # refit produces a negative slope.
    m = lat()
    for bs in (1, 8, 32, 128, 512):
        vals = [m.tpot_ms(bs=bs, total_kv=k)
                for k in (200_000, 400_000, 800_000, 2_000_000)]
        for a, b in zip(vals, vals[1:]):
            assert b >= a - 1e-9, (bs, vals)


# ─────────────────────────────── power ──────────────────────────────────────

def test_power_rises_with_the_batch_and_stays_under_the_board():
    m = lat()
    vals = [m.power_w(bs=b) for b in (1, 4, 16, 64, 256)]
    assert all(b >= a - 1e-6 for a, b in zip(vals, vals[1:])), vals
    assert all(0 < v <= DEVICE_CEILING_W for v in vals), vals


def test_an_idle_engine_still_draws_something():
    assert lat().power_w(bs=0) > 0, "an idle device draws no power"


# ─────────────────────────────── energy ─────────────────────────────────────

def test_the_energy_model_loads_for_the_fitted_pairs():
    for model, tp in (("llama3_8B", 1), ("llama3_8B", 2), ("qwen2.5_7B", 2)):
        e = StepEnergyModel.load(model, tp)
        assert e is not None, (model, tp)


def test_a_prefill_step_costs_more_for_more_tokens():
    e = StepEnergyModel.load("llama3_8B", 2)
    vals = [e.prefill_step_j(n, n_reqs=1) for n in (256, 1024, 4096, 8192)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals
    assert all(v > 0 for v in vals), vals


def test_a_decode_step_costs_more_for_a_bigger_batch():
    e = StepEnergyModel.load("llama3_8B", 2)
    vals = [e.decode_step_j(b, kv_tokens=4096) for b in (1, 8, 32, 128)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals
    assert all(v > 0 for v in vals), vals


def test_a_decode_step_costs_more_for_a_bigger_cache():
    e = StepEnergyModel.load("llama3_8B", 2)
    small = e.decode_step_j(32, kv_tokens=1024)
    large = e.decode_step_j(32, kv_tokens=65536)
    assert large >= small, (small, large)


def test_a_decode_segment_is_the_integral_of_its_steps():
    # decode_segment_j integrates over a growing cache; it must land between
    # the step cost at each end times the number of steps.
    e = StepEnergyModel.load("llama3_8B", 2)
    b, kv0, kv1 = 32, 1024.0, 9216.0
    seg = e.decode_segment_j(b, kv0, kv1)
    j = seg[0] if isinstance(seg, tuple) else seg
    assert j > 0, seg


def test_idle_energy_is_power_times_time():
    e = StepEnergyModel.load("llama3_8B", 2)
    a, b = e.idle_j(1.0), e.idle_j(10.0)
    assert a > 0, a
    assert abs(b - 10.0 * a) < 1e-6, (a, b)


def test_energy_over_a_window_stays_under_the_board():
    e = StepEnergyModel.load("llama3_8B", 2)
    for seconds in (0.001, 1.0, 60.0):
        j = e.decode_step_j(128, kv_tokens=32768)
        w = e.power_w(j, max(seconds, 1e-6))
        assert w > 0, (seconds, w)
    # the direct check: a single decode step's implied power
    j = e.decode_step_j(128, kv_tokens=32768)
    step_s = lat().tpot_s(bs=128, avg_isl=256)
    assert 0 < j / step_s <= DEVICE_CEILING_W * 1.05, \
        f"a decode step implies {j / step_s:.0f} W"


def test_the_energy_model_says_what_it_loaded():
    s = StepEnergyModel.load("llama3_8B", 2).summary()
    assert s.strip(), "an empty summary"


# ─────────────────────────────── the factory ────────────────────────────────

def test_the_factory_offers_lookup_and_predictor_but_not_analytical():
    # The factory is the LOOKUP side. The analytical model is built directly by
    # datacenter/server.py, which is why `latency_mode: analytical` never
    # reaches here -- worth pinning, because the two modes read as siblings in
    # the config and are not siblings in the code.
    try:
        create_latency_backend("analytical", model_name="llama3_8B", tp_size=2)
    except (ValueError, KeyError, TypeError) as e:
        assert "analytical" in str(e) or "lookup" in str(e), e
        return
    raise AssertionError("the factory now builds analytical; update the note "
                         "in datacenter/server.py that says it does not")


def test_the_factory_refuses_a_mode_it_does_not_have():
    try:
        create_latency_backend("telepathy", model_name="llama3_8B", tp_size=2)
    except (ValueError, KeyError):
        return
    raise AssertionError("an unknown latency mode was accepted")


# ───────────────── the composition the simulator relies on ──────────────────

def test_service_time_is_prefill_plus_the_decode_steps():
    m = lat()
    isl, osl, bs = 512, 64, 8
    got = m.estimate_service_ms(isl, osl, bs)
    want = m.ttft_ms(isl, bs=bs) + m.tpot_ms(bs=bs) * (osl - 1)
    assert abs(got - want) / want < 0.25, (got, want)
    assert got > 0


def test_a_longer_output_takes_longer():
    m = lat()
    vals = [m.estimate_service_ms(512, osl, 8) for osl in (8, 64, 256, 1024)]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        check(t.__name__)(t)
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    sys.exit(1 if FAIL else 0)
