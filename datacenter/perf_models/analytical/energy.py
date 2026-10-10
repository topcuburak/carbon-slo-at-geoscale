"""Measured per-step energy model, loaded from the calibration artifacts.

Replaces the three-state abstraction (idle / constant 400 W prefill /
log-linear decode) with the form actually fitted to engine-step measurements:

    E_step = e0
           + e0_p    * 1[prefill step]
           + e0_plo  * 1[prefill step and N < L*]
           + e_d     * B          decoding requests in the step
           + e_kv    * K          cached tokens they hold
           + e_p     * N          prefill tokens processed in the step

Two properties of that form drive how the simulator uses it.

**Counting steps needs no latency model.** Every engine step advances each of
the B decoding requests by exactly one token, so the cache grows by exactly B
tokens per step. Between two points where B is constant,

    n_steps = (K1 - K0) / B

which is a count, not an estimate. Integrating an effective power over elapsed
time would give the same answer -- the simulator sleeps one step time per step
-- but only after re-deriving that step time, and an inconsistency there
produces energies that violate the device TDP. The count cannot.

**K grows linearly inside a segment**, so the mean of e_kv*K over the segment
is exactly e_kv*(K0+K1)/2 and the segment closes in one expression rather than
being sampled.

Coefficients are per (model, TP) and are NOT interchangeable across TP. That
was measured: per-GPU TP=2/TP=1 ratios are 0.736 (e0), 0.713 (e_d), 0.366
(e_kv) and 0.628 (e_p) -- dividing by the device count reproduces none of
them. Loading asks for the exact TP and fails rather than rescaling.
"""

import json
import os
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).resolve().parent / "fit_data"

# Per-GPU idle draw with model weights resident, used when a campaign recorded
# no idle blocks of its own. The mean of the two configurations that did:
# 81.6 W/GPU at TP=1 and 75.5 W/GPU at TP=2 (the latter pooled over two
# campaigns, warm windows only). The released simulator assumed 60 W.
#
# Idle does not scale linearly with TP -- splitting the weights across two
# cards lowers the per-card draw by 7% -- so this average carries a few percent
# of error per configuration. That is deliberate: measuring it per (model, TP)
# costs a campaign each and buys less than the term's own spread, which is 4%
# between two idle windows of the SAME configuration depending on how hard the
# card was worked just before.
#
# Do not mistake a small per-card error for a small term. Idle is charged for
# every second a datacentre holds no work, and a carbon-aware policy
# deliberately leaves the dirty datacentres empty -- so idle hours accumulate
# exactly where the policy is doing its job, and can rival the energy of the
# requests themselves.
_FALLBACK_IDLE_W_PER_GPU = 78.5


class StepEnergyModel:
    """Energy of one engine step, in joules, for one (model, TP)."""

    def __init__(self, art: dict, source: str = ""):
        c = art["coefficients"]
        self.model = art["model"]
        self.tp = int(art["tp"])
        self.source = source
        self.e0 = float(c.get("e0", 0.0))
        self.e0_p = float(c.get("e0_p", 0.0))
        self.e0_plo = float(c.get("e0_plo", 0.0))
        self.e_d = float(c.get("e_d", 0.0))
        self.e_kv = float(c.get("e_kv", 0.0))
        self.e_p = float(c.get("e_p", 0.0))
        self.e_pr = float(c.get("e_pr", 0.0))
        # Attention is quadratic in each sequence's own length, so it is
        # fitted against sum_i(n_i^2), NOT against the token total. Zero for
        # any artifact fitted before the term existed, which reproduces the
        # old form exactly.
        self.e_att = float(c.get("e_att", 0.0))

        env = art.get("envelope", {})
        self.plateau_tokens = int(env.get("prefill_plateau_min_tokens", 0))
        self.bs_max = env.get("bs_max")
        self.kv_max = env.get("kv_max")
        self.prefill_tokens_max = env.get("prefill_tokens_per_step_max")

        idle = art.get("idle_levels") or {}
        w = idle.get("_engine_idle_w")
        self.idle_w = (float(w) if w
                       else _FALLBACK_IDLE_W_PER_GPU * self.tp)
        self.idle_measured = bool(w)

        # engine_tdp_w and prefill_plateau_w (= timing.e_p_over_a1_w) were
        # loaded here for a consistency check the artifact's own note describes
        # -- "e_p/a1 is the prefill marginal power and should sit just under
        # engine_tdp_w" -- and the check was never written, so the two values
        # were read and never compared. Removed; both are still in the artifact
        # for anyone who wants to write it.
        self.holdout_mape = art.get("holdout_mape")

        # Extrapolation is reported once per kind rather than per call: a
        # simulation makes millions of these and a per-call warning would bury
        # the run. Silence would be worse -- a policy that pushes past the
        # calibrated envelope should leave a mark in the log.
        self._warned = set()

    # ---- loading -----------------------------------------------------------

    @classmethod
    def load(cls, model_name: str, tp: int,
             data_dir: Optional[str] = None) -> Optional["StepEnergyModel"]:
        """Load the artifact for exactly this (model, TP), or return None.

        None, not a rescaled neighbour: TP scaling of these coefficients was
        tested and refuted, so a missing artifact means the configuration was
        never calibrated and the caller has to decide what to do about it.
        """
        if not model_name:
            return None
        d = Path(data_dir or os.environ.get("ENERGY_FIT_DIR") or _DATA_DIR)
        p = d / f"energy_fit_tp{tp}_{model_name}.json"
        if not p.exists():
            return None
        try:
            return cls(json.loads(p.read_text()), source=str(p))
        except Exception:
            return None

    @staticmethod
    def available(data_dir: Optional[str] = None):
        """(model, tp) pairs that have an artifact."""
        d = Path(data_dir or os.environ.get("ENERGY_FIT_DIR") or _DATA_DIR)
        out = []
        for p in sorted(d.glob("energy_fit_tp*_*.json")):
            try:
                a = json.loads(p.read_text())
                out.append((a["model"], int(a["tp"])))
            except Exception:
                continue
        return out

    # ---- the model ---------------------------------------------------------

    def prefill_step_j(self, n_tokens: int, n_reqs: int = 0,
                       sum_sq_tokens: float = None) -> float:
        """Energy of one prefill forward pass over `n_tokens` tokens.

        Below the plateau threshold the step draws well under TDP -- 274 W
        against 400 for Llama-3.2-3B at 128 tokens -- which is what e0_plo
        carries. Pinning prefill at a constant 400 W, as the released
        simulator does, is only right above that threshold.
        """
        if n_tokens <= 0:
            return 0.0
        self._check("prefill_tokens", n_tokens, self.prefill_tokens_max)
        e = self.e0 + self.e0_p + self.e_p * n_tokens
        if self.plateau_tokens and n_tokens < self.plateau_tokens:
            e += self.e0_plo
        # sum_i(n_i^2). e_p is linear in the token total and so spreads one long
        # sequence's attention cost over every length; this term puts it where
        # it belongs. Callers that know the individual prompt lengths pass the
        # exact sum; the fallback assumes the step's sequences are equal, which
        # is what makes sum_sq = n_tokens^2 / n_reqs.
        if self.e_att:
            if sum_sq_tokens is None:
                sum_sq_tokens = float(n_tokens) ** 2 / max(int(n_reqs), 1)
            e += self.e_att * sum_sq_tokens
        return e + self.e_pr * n_reqs

    def decode_step_j(self, b: int, kv_tokens: float) -> float:
        """Energy of one decode step with `b` decoding requests holding `kv_tokens`."""
        if b <= 0:
            return 0.0
        self._check("bs", b, self.bs_max)
        self._check("kv", kv_tokens, self.kv_max)
        return self.e0 + self.e_d * b + self.e_kv * kv_tokens

    def decode_segment_j(self, b: int, kv0: float, kv1: float):
        """Energy of a run of decode steps at constant `b`, cache kv0 -> kv1.

        Returns (joules, n_steps). The step count comes from the cache growth,
        not from elapsed time: each step adds exactly `b` tokens.
        """
        if b <= 0 or kv1 <= kv0:
            return 0.0, 0
        n_steps = (kv1 - kv0) / float(b)
        kv_mean = 0.5 * (kv0 + kv1)          # exact: K is linear in the segment
        self._check("bs", b, self.bs_max)
        self._check("kv", kv1, self.kv_max)
        return n_steps * (self.e0 + self.e_d * b + self.e_kv * kv_mean), n_steps

    def idle_j(self, seconds: float) -> float:
        return self.idle_w * max(0.0, seconds)

    # ---- diagnostics -------------------------------------------------------

    def power_w(self, energy_j: float, seconds: float) -> float:
        """Effective power for a segment. Only meaningful with a real step time.

        Kept separate from the energy path on purpose: this is the one place a
        latency error turns into a power error, and the TDP ceiling makes such
        an error visible.
        """
        return energy_j / seconds if seconds > 0 else 0.0

    def _check(self, kind: str, value, limit) -> None:
        if limit and value > limit and kind not in self._warned:
            self._warned.add(kind)
            print(f"[energy] {self.model} TP={self.tp}: {kind}={value:.0f} is "
                  f"outside the calibrated envelope (max {limit:.0f}); the "
                  f"model is extrapolating from here on.", flush=True)

    def summary(self) -> str:
        idle = f"{self.idle_w:.1f} W" + ("" if self.idle_measured else " (assumed)")
        return (f"{self.model} TP={self.tp}: e0={self.e0:.3f} J, "
                f"e_d={self.e_d*1e3:.1f} mJ/req, e_kv={self.e_kv*1e6:.1f} uJ/tok, "
                f"e_p={self.e_p*1e3:.2f} mJ/tok, L*={self.plateau_tokens}, "
                f"idle {idle}, holdout MAPE {self.holdout_mape:.2f}%")


if __name__ == "__main__":
    for model, tp in StepEnergyModel.available():
        m = StepEnergyModel.load(model, tp)
        print(m.summary())
