"""Analytical latency prediction models derived from A100 benchmarks.

TTFT (prefill) and TPOT (decode per step) formulas grounded in hardware
behavior: compute-bound vs memory-bound regimes.

TTFT = f(model, ISL):
    - Short ISL: constant overhead (kernel launch, attention setup)
    - Long ISL: power law ~ISL^1.1-1.2 (attention + linear layers)

TPOT = f(model, BS, avg_ISL):
    - Low BS: memory-bound, weak ISL dependence
    - High BS, short ISL: compute-bound, BS scaling
    - High BS, long ISL: memory bandwidth saturated, ISL + BS scaling

All formulas validated against real A100 benchmarks with batch-drain
simulation across 5 datasets (MBPP, HumanEval, ShareGPT, Arxiv, GovReport).

Usage:
    from datacenter.latency_analytical import AnalyticalLatencyModel

    model = AnalyticalLatencyModel("llama3_8B")
    ttft_ms = model.ttft_ms(isl=5000)
    tpot_ms = model.tpot_ms(bs=16, avg_isl=5000)
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

_log = logging.getLogger(__name__)


# ── Lookup-table loader (used for MoE models in the scheduler) ──────────
def _load_binned_lookup(path: str, edge_key: str = "kv_edges") -> Dict[int, Dict[str, Any]]:
    """Load a per-BS binned lookup JSON in the format produced by
    perf_models/lookup/campaign/build_lookups.py. Returns
    {bs: {edges, medians_ms}}.

    edge_key is "kv_edges" for TPOT lookups or "total_prefill_edges" for TTFT.
    """
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        payload = json.load(f)
    table: Dict[int, Dict[str, Any]] = {}
    for bs_str, cell in payload.get("table", {}).items():
        if edge_key not in cell:
            continue
        table[int(bs_str)] = {
            "edges":      np.asarray(cell[edge_key], dtype=float),
            "medians_ms": np.asarray(cell["medians_ms"], dtype=float),
        }
    return table


def _binned_lookup_value(table: Dict[int, Dict[str, Any]], bs: int, x: float) -> Optional[float]:
    """Given a per-BS binned table and a query (bs, x), return median_ms or None."""
    if not table:
        return None
    if bs not in table:
        bs = min(table.keys(), key=lambda b: abs(b - bs))
    cell = table[bs]
    edges = cell["edges"]; meds = cell["medians_ms"]
    if len(meds) == 0:
        return None
    idx = int(np.digitize(x, edges) - 1)
    idx = max(0, min(idx, len(meds) - 1))
    return float(meds[idx])


_ARTIFACT_DIR = Path(__file__).resolve().parent / "fit_data"


def _load_latency_artifact(model_name: str, tp) -> Optional[dict]:
    """Coefficients fitted to the engine-step blocks for exactly this (model, TP).

    Preferred over the tables below wherever one exists, for three reasons the
    tables cannot fix in place.

    **The tables carry no TP index.** They were fitted at TP=2, so a TP=1 run
    silently got TP=2 coefficients. The artifacts are per (model, TP) for the
    same reason the energy artifacts are: the coefficients do not rescale by
    device count, and that was measured rather than assumed.

    **The tables and the energy model disagree.** The shipped prefill slope
    a1 = 0.047790 ms/token puts the measured e_p at 684 W against an 800 W
    engine TDP -- it claims prefill runs 14.5% under a device the power
    campaign watched saturate. Refitting from the blocks the energy model was
    fitted to gives 809 W (+1.1%), so the two models now agree on what the
    hardware did.

    **The linear prefill form cannot bend.** Its residual changes sign around
    the point where the batch fills the device, which no slope removes. The
    piecewise form here takes prefill error on an unseen dataset from 17.5% to
    1.3-1.6%.

    Returns None when no artifact exists, and the constructor then RAISES.
    There is no table to fall back to any more: the old one was TP-blind and
    predated the attention term, so a missing artifact used to produce a
    plausible wrong number rather than an error. Fitted pairs: llama3_8B and
    qwen2.5_7B at TP=1 and TP=2, mixtral_8x7b at TP=4.
    """
    if not model_name or not tp:
        return None
    f = _ARTIFACT_DIR / f"latency_fit_tp{int(tp)}_{model_name}.json"
    if not f.exists():
        return None
    try:
        a = json.loads(f.read_text())
    except Exception:
        return None
    return a if ("ttft" in a and "tpot" in a) else None


# ── TTFT coefficients per model ──────────────────────────────────────────
# Format: (isl_threshold, overhead_ms, a, b)
# if ISL <= threshold: TTFT = overhead_ms
# if ISL > threshold:  TTFT = a * ISL^b  (in seconds)

# The TP=2-only fallback tables that used to live here are gone. They carried
# no TP index, so a TP=1 or TP=4 run silently received TP=2 coefficients, and
# they predated the attention term -- `piecewise_linear` with no b5, which
# under-reads a long single sequence by 13-18%. Two of their entries were for
# configurations that cannot exist on this hardware at all: mixtral_8x7b needs
# ~94 GB of weights in fp16 and therefore at least four A100-40GBs, so its
# TP=1 and TP=2 entries described runs nobody can perform.
#
# A (model, TP) pair without a fitted artifact now raises instead of quietly
# being served stale coefficients. Deleted with them:
# _llama3_8B_legacy_piecewise, the deepseek_moe_16b entries, and the eight
# orphaned ttft_fit_*/tpot_fit_* JSONs under fit_data/ that nothing
# loaded.


# ── Power coefficients per model ─────────────────────────────────────────
# power_w = p0 + p1 * log2(BS)
# Fitted from A100 benchmark power.csv data. Log-linear in BS.

_POWER_PARAMS = {
    "llama3_8B": {
        "p0": 308.7,    # avg of GovReport (332) and MBPP (285)
        "p1": 6.63,     # avg of GovReport (6.27) and MBPP (7.00)
    },
}


class AnalyticalLatencyModel:
    """Analytical TTFT/TPOT/Power predictor for a specific LLM model.

    Args:
        model_name: a model with a fitted artifact for this exact TP.
            ``available()`` lists them.
    """

    def __init__(self, model_name: str, tp=None):
        art = _load_latency_artifact(model_name, tp)
        if art is None:
            raise ValueError(
                f"No fitted latency artifact for {model_name!r} at TP={tp}. "
                f"Fitted: {self.available()}. There is deliberately no "
                f"fallback table: the old one was TP-blind and predated the "
                f"attention term, so falling back returned a plausible wrong "
                f"number instead of an error."
            )
        self.model_name = model_name
        self.tp = tp
        # How many decode steps were priced at the edge of the fit rather than
        # inside it. Read it at the end of a run; a non-zero count is a caveat
        # the results have to carry.
        self._envelope_clamps = 0
        # The denominator. Without it the clamp count says an event happened,
        # not how often -- and "at least once" and "69% of decisions" are the
        # difference between a footnote and a result that does not hold.
        self._tpot_evaluations = 0
        self._envelope_warned = False
        self._seqlen_warned = False
        self.fit_source = art["source"]
        self._ttft = art["ttft"]
        self._tpot = art["tpot"]
        self.holdout_mape = art.get("holdout_mape")
        self.validation = art.get("validation")
        # Absent for a model that was fitted for latency but not for the
        # log-linear power curve. That curve is now a fallback on both call
        # paths -- the measured StepEnergyModel is preferred -- so a missing
        # entry is only fatal if something actually asks for it.
        self._power = _POWER_PARAMS.get(model_name)

        # Lazy-load any binned lookup tables referenced by lookup-form params.
        # All active models now use closed-form fits in the scheduler
        # (LLaMA-8B: linear; MoE models: piecewise_linear / linear_curvature).
        # The lookup branches remain for any future model that needs them
        # and for the legacy archive entries.
        self._ttft_lookup_table: Dict[int, Dict[str, Any]] = {}
        self._tpot_lookup_table: Dict[int, Dict[str, Any]] = {}
        if self._ttft.get("form") == "lookup":
            self._ttft_lookup_table = _load_binned_lookup(
                self._ttft["lookup_json"], edge_key="total_prefill_edges"
            )
            if not self._ttft_lookup_table:
                raise ValueError(
                    f"TTFT lookup JSON missing or empty: {self._ttft['lookup_json']}"
                )
        if self._tpot.get("form") == "lookup":
            self._tpot_lookup_table = _load_binned_lookup(
                self._tpot["lookup_json"], edge_key="kv_edges"
            )
            if not self._tpot_lookup_table:
                raise ValueError(
                    f"TPOT lookup JSON missing or empty: {self._tpot['lookup_json']}"
                )

    def ttft_ms(self, total_prefill_tokens: int, bs: int = 1,
                sum_sq: float = None) -> float:
        """Predict batched prefill wall time in milliseconds.

        Args:
            total_prefill_tokens: Sum of input-sequence lengths in the
                batch being prefilled. For a single request at BS=1 this
                is just its ISL.
            bs: Number of requests sharing the single-pass prefill.
        """
        p = self._ttft
        if p.get("form") in ("piecewise_linear", "piecewise_linear_attn"):
            n = max(0, total_prefill_tokens)
            b = max(1, bs)
            v = (p["b0"]
                 + p["b1"] * n
                 + p["b2"] * max(0.0, n - p["k1"])
                 + p["b3"] * max(0.0, n - p["k2"])
                 + p["b4"] * b)
            # A form with no attention term cannot price a long single
            # sequence: every term reads the step's total, so it answers the
            # same for one 8192-token prompt and eight 1024-token ones, which
            # differ by 13-18%. Say so rather than return the number quietly.
            # sqrt(sum_sq) bounds the longest sequence in the step.
            if not p.get("b5") and not self._seqlen_warned:
                cap = p.get("seq_len_max")
                longest = (sum_sq ** 0.5 if sum_sq else n / b)
                if cap and longest > cap * 1.05:
                    self._seqlen_warned = True
                    _log.warning(
                        "%s TP=%s: pricing a %.0f-token sequence with a fit "
                        "whose longest single sequence was %.0f and which has "
                        "no attention term. Measured error at 8k on a "
                        "comparable fit: 13-18%% LOW. Refit from a campaign "
                        "with long single sequences before trusting this.",
                        self.model_name, self.tp, longest, cap)
            if p.get("b5"):
                # Attention, which every term above is blind to. They read the
                # step's TOTAL tokens, so one 12288-token sequence and two
                # 6144-token ones look identical -- and measured on our testbed they
                # differ by 9.2%, because attention is quadratic WITHIN a
                # sequence. sum_sq is sum_i(n_i^2) over the sequences in the
                # step; without it the caller is assumed to have `bs` equal
                # ones, which is exact for a single request and for a
                # calibration block, and an approximation for a step that fused
                # a long prompt with short ones. Pass the real thing where the
                # lengths are known.
                v += p["b5"] * (total_prefill_tokens ** 2 / b
                                if sum_sq is None else sum_sq)
            return v
        if p.get("form") == "lookup":
            v = _binned_lookup_value(self._ttft_lookup_table, max(1, bs),
                                      max(1, total_prefill_tokens))
            if v is not None:
                return v
            # Fallback if BS is way out of range or table empty
            return float("inf")
        raise ValueError(f"{self.model_name} TP={self.tp}: unknown ttft form "
                         f"{p.get('form')!r}")

    def ttft_s(self, total_prefill_tokens: int, bs: int = 1,
               sum_sq: float = None) -> float:
        """Predict batched prefill wall time in seconds."""
        return self.ttft_ms(total_prefill_tokens, bs, sum_sq) / 1000.0

    # ── SLA budget helpers ──────────────────────────────────────────────
    # The ISL-aware TTFT budget lived here: budget(ISL) = slack + 1.5 x
    # prefill_ms(ISL), with a 1500 ms additive slack for what the SCHEDULER
    # adds (queue, forwarding, contention) over the prefill compute it cannot
    # control. Defensible, and removed anyway: it made the budget grow with the
    # prompt -- 1530 ms at ISL 128, 3104 ms at ISL 20000 -- so a long request
    # could not breach it and the constraint bound nothing. Measured 0 of 6656
    # (request, DC) pairs infeasible at the loads the sweep runs.
    #
    # TTFT_BUDGET is now a flat ceiling, compared against the estimate directly
    # in _check_sla.

    @property
    def envelope_clamps(self):
        """(clamped, evaluated) decode-step prices. A non-zero first element is
        a caveat the results carry: the fit was extrapolated."""
        return self._envelope_clamps, self._tpot_evaluations

    def tpot_ms(self, bs: int, avg_isl: float = 0.0,
                total_kv: int = 0) -> float:
        """Predict per-engine-step decode wall time in milliseconds.

        Preferred call: pass ``total_kv`` directly (sum of KV cache lengths
        across all active decoding requests). Legacy call:
        ``tpot_ms(bs, avg_isl)``; in that case ``total_kv`` is approximated
        as ``bs * avg_isl`` (treats decode as starting at step 0 with no
        generation yet — accurate only for the first decode step of a batch).
        """
        self._tpot_evaluations += 1
        bs = max(1, int(bs))
        p = self._tpot

        if p.get("form") == "linear":
            if total_kv <= 0:
                total_kv = int(bs * max(1.0, avg_isl))
            return p["c0"] + p["c1"] * bs + p["c2"] * total_kv

        if p.get("form") == "linear_curvature":
            if total_kv <= 0:
                total_kv = int(bs * max(1.0, avg_isl))
            # Bound the point the fit is EVALUATED at, because the raw
            # polynomial goes negative outside its envelope. What happens
            # beyond the bound is handled after the evaluation, by extending
            # with the boundary gradient -- see below. It used to stop here,
            # which froze the prediction at the boundary value.
            #
            # Plateauing is safe but it is not free: past the envelope TPOT
            # stops growing with the cache, so a run that spends time out
            # there is optimistic by an unknown amount. On the testbed that
            # never happened -- KV sat at a third of the fit's kv_max. The
            # paper's trace runs an order of magnitude closer to it (mean
            # in-flight ~153k tokens against kv_max 298496 for llama3_8B at
            # TP=2), so count the clamps and say so once rather than let a
            # sweep quietly leave the measured region.
            bs_eval = min(bs, p.get("bs_max", bs))
            kv_eval = max(0, min(total_kv, p.get("kv_max", total_kv)))
            # Optional: clamp on KV-PER-SEQUENCE instead of on the step total.
            #
            # c4 multiplies avg_kv = kv/bs and is negative, so it is that term,
            # not kv itself, that sends the prediction below zero outside the
            # fit -- at bs=1, kv=298496 the raw form reads -6.6 ms. Clamping kv
            # keeps the sign safe but also refuses to extrapolate where the fit
            # HAS data: pooled over the two decode campaigns avg_kv was covered
            # from 70 to 12995, and GovReport at rate 2.5 sits at avg_kv 9546 --
            # inside that range, yet clamped because its step total of 305481
            # exceeds kv_max by 2.3%. Bounding avg_kv instead lets bs and kv
            # grow together while still refusing the small-batch/huge-cache
            # corner that is physically unreachable anyway.
            # Bound the PER-SEQUENCE cache too, at whatever the fit actually
            # covered -- the artifact records it as avg_kv_max, 12995 for
            # llama3_8B at TP=2 against a 32768 context. This was an
            # env-gated option defaulting to off; it is now on, because c4
            # multiplies avg_kv and is negative, so evaluating the polynomial
            # at an avg_kv the campaign never saw is what sent it below zero:
            # at bs=1, kv=331200 the form reads -31.9 ms and only the 1 ms
            # floor hid it. That corner is unreachable anyway (one sequence
            # cannot hold more than max_model_len tokens), but the estimator
            # should not depend on a floor to stay sane.
            avg_max = float(os.environ.get("TPOT_AVG_KV_MAX", "0") or 0)
            if avg_max <= 0:
                avg_max = float(p.get("avg_kv_max", 0) or 0)
            if avg_max > 0 and bs_eval > 0:
                kv_eval = max(0, min(kv_eval, avg_max * bs_eval))
            if bs_eval < bs or kv_eval < total_kv:
                self._envelope_clamps += 1
                if not self._envelope_warned:
                    self._envelope_warned = True
                    _log.warning(
                        "%s TP=%s: decode step outside the fitted envelope "
                        "(bs=%d vs bs_max=%s, total_kv=%d vs kv_max=%s). "
                        "Extending with the boundary gradient from here; still "
                        "extrapolation, and the fit should be swept to the "
                        "device-memory KV limit instead.",
                        self.model_name, self.tp, bs, p.get("bs_max"),
                        total_kv, p.get("kv_max"))
            avg_kv = kv_eval / max(bs_eval, 1)
            pred = (p["c0"]
                    + p["c1"] * bs_eval
                    + p["c2"] * kv_eval
                    + p["c3"] * (bs_eval * bs_eval)
                    + p["c4"] * avg_kv
                    + p["c5"] * (bs_eval * kv_eval))

            # Beyond the envelope, CONTINUE WITH THE BOUNDARY GRADIENT rather
            # than return the boundary value.
            #
            # Freezing was the safe half of the answer and the wrong half. The
            # raw polynomial cannot simply be evaluated out there -- c4
            # multiplies kv/bs and is negative for llama3_8B at TP=2, so at
            # bs=1, kv=298496 the form reads -6.6 ms -- but returning a
            # CONSTANT means the prediction stops responding to load entirely.
            # Measured: past kv_max the value froze at 34.249 ms for 350k, for
            # 405k, and for 5,000,000 tokens alike, while the run served
            # 61.7 ms. That flat answer propagated into the TTFT estimator's
            # slot-free time and into the TPOT SLA check, and it is why the
            # scheduler certified 85.5% of placements as inside a 3000 ms
            # budget when 25.8% were.
            #
            # A first-order extension from the boundary is continuous, is
            # exactly the fit inside the envelope, and stays monotone: the
            # gradient is positive for every fitted pair at every batch size
            # checked (llama3_8B TP=2 gives +1.7e-5 to +4.7e-5 ms/token from
            # bs=32 to 256). The max(0, .) guards the one thing that must not
            # happen -- more load reading as less latency -- if a future refit
            # produces a negative slope.
            #
            # It is still extrapolation and should be reported as such. The
            # honest fix is a campaign that sweeps each pair's decode fit to
            # its device-memory KV limit; llama3_8B at TP=2 is fitted to
            # 331,200 of a reachable 461,553.
            if kv_eval < total_kv or bs_eval < bs:
                d_kv = p["c2"] + p["c4"] / max(bs_eval, 1) + p["c5"] * bs_eval
                d_bs = (p["c1"] + 2.0 * p["c3"] * bs_eval
                        + p["c5"] * kv_eval
                        - p["c4"] * kv_eval / max(bs_eval * bs_eval, 1))
                pred += max(0.0, d_kv) * (total_kv - kv_eval)
                pred += max(0.0, d_bs) * (bs - bs_eval)

            # Final guard: floor at 1 ms (numerical safety; should never
            # trigger inside the envelope for sane fits).
            return max(pred, 1.0)

        if p.get("form") == "lookup":
            if total_kv <= 0:
                total_kv = int(bs * max(1.0, avg_isl))
            v = _binned_lookup_value(self._tpot_lookup_table, bs, total_kv)
            if v is not None:
                return v
            return float("inf")

        # Legacy piecewise path (archived llama3_8B entry only)
        avg_isl = max(1.0, avg_isl)
        bs_thresh = p["bs_threshold"]
        isl_thresh = p.get("isl_threshold")
        if isl_thresh is not None:
            if bs <= bs_thresh and avg_isl <= isl_thresh:
                return self._eval_region(p["regions"]["low_bs_short_isl"], bs, avg_isl)
            elif bs <= bs_thresh and avg_isl > isl_thresh:
                return self._eval_region(p["regions"]["low_bs_long_isl"], bs, avg_isl)
            elif bs > bs_thresh and avg_isl <= isl_thresh:
                return self._eval_region(p["regions"]["high_bs_short_isl"], bs, avg_isl)
            else:
                return self._eval_region(p["regions"]["high_bs_long_isl"], bs, avg_isl)
        else:
            if bs <= bs_thresh:
                return self._eval_region(p["regions"]["low_bs"], bs, avg_isl)
            else:
                return self._eval_region(p["regions"]["high_bs"], bs, avg_isl)

    def tpot_s(self, bs: int, avg_isl: float = 0.0,
               total_kv: int = 0) -> float:
        """Predict per-engine-step decode wall time in seconds."""
        return self.tpot_ms(bs, avg_isl=avg_isl, total_kv=total_kv) / 1000.0

    def power_w(self, bs: int) -> float:
        """Predict GPU power draw in watts for a given batch size.

        power = p0 + p1 * log2(BS)

        Fitted from A100 benchmark power measurements.
        """
        if self._power is None:
            raise ValueError(
                f"No log-linear power curve for {self.model_name!r}. Use the "
                f"measured StepEnergyModel (datacenter/energy_model.py), which "
                f"covers this model, rather than reviving this fallback.")
        log_bs = math.log2(max(bs, 1))
        return max(0.0, self._power["p0"] + self._power["p1"] * log_bs)

    @staticmethod
    def available():
        """(model, tp) pairs that have a fitted latency artifact."""
        out = []
        for f in sorted(_ARTIFACT_DIR.glob("latency_fit_tp*_*.json")):
            try:
                a = json.loads(f.read_text())
                out.append((a["model"], int(a["tp"])))
            except Exception:
                continue
        return out

    @staticmethod
    def _eval_region(region: dict, bs: float, avg_isl: float) -> float:
        """Evaluate a TPOT region formula. Returns ms."""
        t = region["type"]

        if t == "isl_power":
            # TPOT = a * ISL^beta + c
            return region["a"] * (avg_isl ** region["beta"]) + region["c"]

        elif t == "bs_power":
            # TPOT = a * BS^alpha + c
            return region["a"] * (bs ** region["alpha"]) + region["c"]

        elif t == "multiplicative":
            # TPOT = a * BS^alpha * ISL^beta + c
            return (region["a"] * (bs ** region["alpha"])
                    * (avg_isl ** region["beta"]) + region["c"])

        elif t == "saturating":
            # TPOT = a * (1 - exp(-b*BS)) * ISL^gamma
            return (region["a"]
                    * (1.0 - math.exp(-region["b"] * bs))
                    * (avg_isl ** region["gamma"]))

        else:
            raise ValueError(f"Unknown region type {t!r}")

    def estimate_service_ms(self, isl: int, osl: int, bs: int,
                            avg_isl: float = 0.0,
                            total_kv: int = 0) -> float:
        """Estimate total service time for a request in milliseconds.

            service_time ≈ ttft(bs, isl) + tpot(bs, total_kv) * osl

        Note: simplified — actual decode time depends on batch drain
        dynamics (other requests leaving during decode). Use ``total_kv``
        for the most accurate prediction; ``avg_isl`` is a legacy fallback
        that approximates ``total_kv ≈ bs * avg_isl``.
        """
        ttft = self.ttft_ms(isl, bs=bs)
        tpot = self.tpot_ms(bs, avg_isl=avg_isl, total_kv=total_kv)
        return ttft + tpot * osl

    def summary(self) -> str:
        """Human-readable summary of the model."""
        lines = [f"AnalyticalLatencyModel({self.model_name}):"]

        # ── TTFT ──
        p = self._ttft
        if p.get("form") == "linear":
            lines.append(
                f"  TTFT: prefill_ms ≈ {p['a0']:.3f} + "
                f"{p['a1']:.6f}·total_ISL + {p['a2']:.4f}·BS  (linear)"
            )
        elif p.get("form") in ("piecewise_linear", "piecewise_linear_attn"):
            # Every fitted artifact is piecewise_linear_attn, and it was the one
            # form with no branch here: summary() fell through to the legacy
            # power-law line and raised KeyError('a') on every calibrated model.
            lines.append(
                f"  TTFT: prefill_ms ≈ {p['b0']:.3f} + {p['b1']:.6f}·N "
                f"+ {p['b2']:.6f}·max(0,N-{p['k1']}) "
                f"+ {p['b3']:.6f}·max(0,N-{p['k2']}) "
                f"+ {p['b4']:.4f}·BS"
                + (f" + {p['b5']:.3e}·Σnᵢ²  (piecewise-linear + attention)"
                   if p.get("b5") else "  (3-segment piecewise-linear)")
            )
        elif p.get("form") == "lookup":
            lines.append(f"  TTFT: binned lookup (BS × N) — see {p['lookup_json']}")
        else:
            lines.append(f"  TTFT: unknown form {p.get('form')!r}")

        # ── TPOT ──
        tp = self._tpot
        if tp.get("form") == "linear":
            lines.append(
                f"  TPOT: step_ms ≈ {tp['c0']:.3f} + "
                f"{tp['c1']:.4f}·BS + {tp['c2']:.2e}·total_KV  (linear)"
            )
        elif tp.get("form") == "linear_curvature":
            lines.append(
                f"  TPOT: step_ms ≈ {tp['c0']:.3f} + {tp['c1']:.4f}·BS "
                f"+ {tp['c2']:.2e}·total_KV + {tp['c3']:.2e}·BS² "
                f"+ {tp['c4']:.2e}·(total_KV/BS) + {tp['c5']:.2e}·(BS·total_KV)  "
                f"(linear+curvature)"
            )
        elif tp.get("form") == "lookup":
            lines.append(f"  TPOT: binned lookup (BS × total_KV) — see {tp['lookup_json']}")
        else:
            lines.append(
                f"  TPOT regions (bs_thresh={tp['bs_threshold']}, "
                f"isl_thresh={tp.get('isl_threshold', 'N/A')}):"
            )
            for name, region in tp["regions"].items():
                lines.append(f"    {name}: {region}")

        lines.append(
            f"  TTFT budget: flat ceiling, set by TTFT_BUDGET"
        )
        return "\n".join(lines)
