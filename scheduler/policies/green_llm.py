"""Green-LLM (arXiv:2507.09942) as a baseline in the geo simulator.

Green-LLM routes LLM queries across geo-distributed datacentres with a linear
programme over fractional flows: minimise carbon (after cost, which we have no
data for) subject to per-datacentre capacity and a cap on the AVERAGE
end-to-end delay, then minimise delay among near-carbon-optimal solutions. It
is an LP, not a MILP -- v1's "MO-MILP" label is a misnomer that v3 drops.

What is reproduced, and what is held fixed.

  The DECISION RULE is Green-LLM's:
    * batch-blind latency: prefill at batch 1, every output token at the batch-1
      decode step, plus the network hop. No queue, no batch, no KV term, no
      TTFT/TPOT split -- the paper has none.
    * batch-blind energy: every request is charged the batch-1 step energy for
      each of its tokens, i.e. per-token linear coefficients fitted at one batch
      size, which is what Green-LLM's tau_in/tau_out are.
    * the latency constraint, in one of two modes. slo_mode="budgets" (the
      default) applies OUR per-request TTFT and TPOT budgets, evaluated with the
      batch-1 estimates above: every method is then asked to meet the same SLO
      and only the latency model differs, which is the comparison the paper
      makes. slo_mode="mean_e2e" is Green-LLM as published: one aggregate row,
      mean end-to-end delay over the window <= 5 s.
    * lexicographic objective: carbon first, then delay within (1 + eps) of the
      carbon optimum.

  The SYSTEM is ours and identical to every other policy's: the same projected
  view, the same RTT matrix read (`_lookup_rtt_ms`), the same PUE-aware carbon
  signal (`_get_carbon_intensity`), the same per-DC headroom row
  (`dc_headroom`), the same step models, the same batch window and the same
  fallbacks. Only the rule differs, so the comparison is at the algorithm level.

Registered as "greenllm": the testbed port keeps "green_llm", and both can be
imported in one process. Ported from testbed/phase2_endtoend/green_llm_policy.py.
Differences from that
port: energy comes from the calibrated step model at batch 1 instead of a flat
300 W; capacity is the simulator's per-engine headroom; `choose_dc` returns a
DC id and `choose_dc_extended` the (dc_id, is_local) pair, as the server
expects of a GLOBAL policy.
"""
import logging
from typing import Dict, List

import inference_pb2

from scheduler.policies import register_policy
from scheduler.policies.common.base import BasePolicy, PolicyScope
from scheduler.policies.common.capacity import at_cap, dc_headroom, greenest_under_cap
from scheduler.policies.common.estimators import (
    RequestInfo,
    RoundRobinFallback,
    _get_carbon_intensity,
    _lookup_rtt_ms,
    _step_model,
)

log = logging.getLogger("scheduler.green_llm")

try:
    import numpy as np
    from scipy.optimize import linprog
    HAS_SOLVER = True
except ImportError:                                              # pragma: no cover
    HAS_SOLVER = False


@register_policy("greenllm")
class GreenLLM(BasePolicy):
    """Batch-blind carbon LP with a mean end-to-end delay cap (Green-LLM)."""

    def __init__(self, scheduler_group: str = "europe_north",
                 ttft_budget_ms: float = 0, tpot_budget_ms: float = 0,
                 analytical_model=None, tp_size: int = 2, bs_cap: int = 0,
                 carbon_eps: float = 0.02, e2e_cap_ms: float = 5000.0,
                 ref_osl: int = 500, slo_mode: str = "", **kwargs):
        self.max_num_seqs = int(kwargs.pop("max_num_seqs", 0) or 0)
        super().__init__(scheduler_group=scheduler_group, **kwargs)
        self.ttft_budget_ms = ttft_budget_ms
        self.tpot_budget_ms = tpot_budget_ms
        self.analytical_model = analytical_model
        self.tp_size = tp_size
        self.bs_cap = bs_cap
        self.carbon_eps = carbon_eps
        self.e2e_cap_ms = e2e_cap_ms
        self.ref_osl = ref_osl
        import os
        mode = (slo_mode or os.environ.get("GREENLLM_SLO_MODE", "budgets")).strip()
        if mode not in ("budgets", "mean_e2e"):
            raise ValueError(f"slo_mode must be 'budgets' or 'mean_e2e', got {mode!r}")
        self.slo_mode = mode
        self._rr_fallback = RoundRobinFallback()

    @property
    def scope(self) -> PolicyScope:
        return PolicyScope.GLOBAL

    # -- batch-blind per-(request, DC) terms ---------------------------------
    def _ttft_tpot_bs1(self, isl: int, osl: int):
        """Batch-1 prefill time and batch-1 decode step, Green-LLM's view."""
        m = self.analytical_model
        if m is None:
            return 0.0, 0.0
        isl = max(1, int(isl))
        osl = max(1, int(osl))
        return (m.ttft_ms(total_prefill_tokens=isl, bs=1),
                m.tpot_ms(1, total_kv=int(isl + osl / 2)))

    def _latency_ms(self, isl: int, osl: int, rtt_ms: float) -> float:
        """Network hop + batch-1 prefill + osl batch-1 decode steps."""
        ttft, tpot = self._ttft_tpot_bs1(isl, osl)
        return rtt_ms + ttft + tpot * max(1, int(osl))

    def _meets_budgets(self, isl: int, osl: int, rtt_ms: float) -> bool:
        """Our TTFT/TPOT budgets, judged on Green-LLM's batch-blind estimates."""
        ttft, tpot = self._ttft_tpot_bs1(isl, osl)
        ok = True
        if self.ttft_budget_ms > 0:
            ok = ok and (rtt_ms + ttft) <= self.ttft_budget_ms
        if self.tpot_budget_ms > 0:
            ok = ok and tpot <= self.tpot_budget_ms
        return ok

    def _feasible(self, isl: int, osl: int, rtt_ms: float) -> bool:
        if self.slo_mode == "budgets":
            return self._meets_budgets(isl, osl, rtt_ms)
        return self._latency_ms(isl, osl, rtt_ms) <= self.e2e_cap_ms

    def _energy_j(self, isl: int, osl: int) -> float:
        """Every token priced at the batch-1 step energy (no amortisation)."""
        isl = max(1, int(isl))
        osl = max(1, int(osl))
        sm = _step_model(self.analytical_model, self.tp_size)
        if sm is not None:
            try:
                return (sm.prefill_step_j(isl, n_reqs=1)
                        + osl * sm.decode_step_j(1, isl + osl / 2.0))
            except Exception:                                    # noqa: BLE001
                pass
        # Uncalibrated setup: the constant-power form the testbed port used.
        m = self.analytical_model
        tpot = m.tpot_ms(1, total_kv=int(isl + osl / 2)) if m is not None else 20.0
        return 300.0 * (tpot / 1000.0) * osl

    @staticmethod
    def _ci(stats) -> float:
        ci = _get_carbon_intensity(stats)
        return ci if ci > 0 else 1.0

    # -- single-request path (forwarded requests, per-arrival decisions) -----
    def choose_dc_extended(self, dc_view, region_id, **kwargs):
        req_isl = int(kwargs.get("req_isl", 0) or 0)
        req_osl = int(kwargs.get("req_osl", 0) or 0) or self.ref_osl
        rtt_matrix = kwargs.get("rtt_matrix") or {}
        energy = self._energy_j(req_isl, req_osl)
        best_dc, best = None, float("inf")
        for dc_id, stats in dc_view.items():
            if at_cap(stats, self.bs_cap):
                continue
            rtt = _lookup_rtt_ms(rtt_matrix, region_id, dc_id)
            if not self._feasible(req_isl, req_osl, rtt):
                continue
            score = energy * self._ci(stats)
            if score < best:
                best, best_dc = score, dc_id
        if best_dc is None:
            best_dc = greenest_under_cap(dc_view, self.bs_cap)
        if best_dc is None:
            best_dc = self._rr_fallback.pick(dc_view)
        if best_dc is None:
            return None
        return (best_dc, best_dc in self._local_regions)

    def choose_dc(self, dc_view, region_id, **kwargs):
        result = self.choose_dc_extended(dc_view, region_id, **kwargs)
        return result[0] if result else None

    # -- batch path: the Green-LLM LP ----------------------------------------
    def choose_dc_batch(self, requests: List[RequestInfo],
                        dc_view: Dict[str, inference_pb2.DcStats],
                        region_id: str, max_batch_size: int = 0,
                        rtt_matrix: Dict[str, float] = None) -> Dict[str, str]:
        rtt_matrix = rtt_matrix or {}
        n, dc_ids = len(requests), sorted(dc_view.keys())
        m = len(dc_ids)
        if n == 0 or m == 0:
            return {}
        if not HAS_SOLVER:
            return self._fallback(requests, dc_view, region_id, rtt_matrix)

        carbon = np.zeros((n, m))
        delay = np.zeros((n, m))
        feasible = np.ones((n, m), dtype=bool)
        rtt = [_lookup_rtt_ms(rtt_matrix, region_id, d) for d in dc_ids]
        ci = [self._ci(dc_view[d]) for d in dc_ids]
        for i, req in enumerate(requests):
            isl = int(req.in_tok)
            osl = int(getattr(req, "predicted_osl", 0) or 0) or self.ref_osl
            e = self._energy_j(isl, osl)
            base = self._latency_ms(isl, osl, 0.0)
            for j in range(m):
                carbon[i, j] = e * ci[j]
                delay[i, j] = base + rtt[j]
                if self.slo_mode == "budgets":
                    feasible[i, j] = self._meets_budgets(isl, osl, rtt[j])
            if not feasible[i].any():
                # No DC meets the budgets for this request; it must still go
                # somewhere, so it competes on carbon alone.
                feasible[i, :] = True

        # each request placed exactly once
        A_eq = np.zeros((n, n * m))
        for i in range(n):
            A_eq[i, i * m:(i + 1) * m] = 1.0
        b_eq = np.ones(n)

        # per-DC headroom, the same row the ILPs carry
        cap_rows, cap_vals = [], []
        for j, d in enumerate(dc_ids):
            h = dc_headroom(dc_view[d], self.bs_cap)
            if h is None:
                continue
            row = np.zeros(n * m)
            row[j::m] = 1.0
            cap_rows.append(row)
            cap_vals.append(float(h))
        # Green-LLM's latency constraint. "mean_e2e": one row, MEAN end-to-end
        # delay <= cap. "budgets": per-pair bounds below, no aggregate row.
        use_mean = self.slo_mode == "mean_e2e"
        mean_row = delay.flatten()
        mean_val = n * self.e2e_cap_ms
        bounds = [(0, 1) if feasible[i, j] else (0, 0)
                  for i in range(n) for j in range(m)]

        def solve(c, with_cap=True, extra=None):
            with_cap = with_cap and use_mean
            rows = list(cap_rows) + ([mean_row] if with_cap else [])
            vals = list(cap_vals) + ([mean_val] if with_cap else [])
            if extra is not None:
                rows.append(extra[0])
                vals.append(extra[1])
            A_ub = np.vstack(rows) if rows else None
            b_ub = np.array(vals) if vals else None
            return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                           bounds=bounds, method="highs")

        try:
            c_vec = carbon.flatten()
            with_cap = True
            r1 = solve(c_vec, with_cap=True)
            if not r1.success:
                # No placement meets the latency constraint within capacity. The
                # paper would call the instance infeasible; we must still route,
                # so the constraint is dropped and carbon is re-solved on
                # capacity alone.
                with_cap = False
                bounds[:] = [(0, 1)] * (n * m)
                r1 = solve(c_vec, with_cap=False)
            if not r1.success:
                return self._fallback(requests, dc_view, region_id, rtt_matrix)
            c_star = float(c_vec @ r1.x)
            r2 = solve(delay.flatten(), with_cap=with_cap,
                       extra=(c_vec, c_star * (1.0 + self.carbon_eps)))
            x = (r2.x if r2.success else r1.x).reshape(n, m)
        except Exception as e:                                   # noqa: BLE001
            log.warning("Green-LLM solve failed (%s); greedy fallback", e)
            return self._fallback(requests, dc_view, region_id, rtt_matrix)

        # The mean-delay row breaks total unimodularity, so x can be
        # fractional; Green-LLM routes fractions of a flow, and at request
        # granularity the largest share decides.
        return {req.req_id: dc_ids[int(np.argmax(x[i]))]
                for i, req in enumerate(requests)}

    def _fallback(self, requests, dc_view, region_id, rtt_matrix):
        out = {}
        for req in requests:
            osl = int(getattr(req, "predicted_osl", 0) or 0) or self.ref_osl
            r = self.choose_dc_extended(dc_view, region_id, req_isl=req.in_tok,
                                        req_osl=osl, rtt_matrix=rtt_matrix)
            if r is not None:
                out[req.req_id] = r[0]
        return out
