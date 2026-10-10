"""Formula-based latency and power prediction.

Uses fitted curves instead of interpolated lookup tables:

- **TPOT**: Saturating model ``a * (1 - exp(-b * bs)) + c``
- **Prefill rate**: Cubic in log₂-log₂ space
- **Power**: Log-linear ``p0 + p1 * log2(bs)``

Two ways to create a predictor:

1. **Direct coefficients** (for hypothetical configs)::

       predictor = LatencyPredictor(
           tpot_coeffs={"a": 0.185, "b": 0.021, "c": 0.008},
           prefill_coeffs={"a": -0.002, "b": 0.074, "c": 0.621, "d": -13.281},
           power_coeffs={"p0": 617.0, "p1": 16.3},
       )

2. **Auto-fit from benchmark CSV** (same CSV format as LatencyLookup)::

       predictor = LatencyPredictor.from_csv("benchmark.csv")
"""

import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from datacenter.perf_models.base import LatencyBackend


def _fit_saturating(points: List[Tuple[int, float]]) -> Dict[str, float]:
    """Fit TPOT = a * (1 - exp(-b * bs)) + c using scipy."""
    from scipy.optimize import curve_fit
    import numpy as np

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    def model(x, a, b, c):
        return a * (1.0 - np.exp(-b * x)) + c

    popt, _ = curve_fit(model, xs, ys, p0=[ys.max(), 0.02, ys.min()], maxfev=10000)
    return {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])}


def _fit_cubic_loglog(points: List[Tuple[int, float]]) -> Dict[str, float]:
    """Fit log2(rate) = a*log2(bs)^3 + b*log2(bs)^2 + c*log2(bs) + d."""
    import numpy as np

    xs = np.log2(np.array([p[0] for p in points], dtype=float))
    ys = np.log2(np.array([p[1] for p in points], dtype=float))

    coeffs = np.polyfit(xs, ys, 3)
    return {"a": float(coeffs[0]), "b": float(coeffs[1]),
            "c": float(coeffs[2]), "d": float(coeffs[3])}


class LatencyPredictor(LatencyBackend):
    """Formula-based TPOT / TTFT / Power prediction.

    Args:
        tpot_coeffs: {"a", "b", "c"} for saturating model.
        prefill_coeffs: {"a", "b", "c", "d"} for cubic log-log model.
    """

    def __init__(
        self,
        tpot_coeffs: Dict[str, float],
        prefill_coeffs: Dict[str, float],
    ) -> None:
        self._tpot = tpot_coeffs
        self._prefill = prefill_coeffs

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
    ) -> "LatencyPredictor":
        """Auto-fit curves from the same CSV format as LatencyLookup."""
        tpot_by_bs: Dict[int, List[float]] = defaultdict(list)
        prefill_rate_by_bs: Dict[int, List[float]] = defaultdict(list)

        with open(csv_path) as f:
            for row in csv.DictReader(f):
                bs = int(row["batch_size"])
                tpot_by_bs[bs].append(float(row["tpot"]))
                in_tokens = int(row["in_tokens"])
                ttft_val = float(row["ttft"])
                if in_tokens > 0:
                    prefill_rate_by_bs[bs].append(ttft_val / in_tokens)

        tpot_points = sorted(
            (bs, sum(v) / len(v)) for bs, v in tpot_by_bs.items()
        )
        prefill_points = sorted(
            (bs, sum(v) / len(v)) for bs, v in prefill_rate_by_bs.items()
        )

        if not tpot_points:
            raise ValueError(f"No data loaded from {csv_path}")

        tpot_coeffs = _fit_saturating(tpot_points)
        prefill_coeffs = _fit_cubic_loglog(prefill_points)

        return cls(tpot_coeffs, prefill_coeffs)

    def tpot(self, batch_size: int) -> float:
        """TPOT = a * (1 - exp(-b * bs)) + c."""
        a, b, c = self._tpot["a"], self._tpot["b"], self._tpot["c"]
        return max(0.0, a * (1.0 - math.exp(-b * batch_size)) + c)

    def ttft(self, batch_size: int, in_tokens: int) -> float:
        """TTFT = prefill_rate(bs) * in_tokens.

        prefill_rate = 2^(a*log2(bs)^3 + b*log2(bs)^2 + c*log2(bs) + d)
        """
        log_bs = math.log2(max(batch_size, 1))
        a = self._prefill["a"]
        b = self._prefill["b"]
        c = self._prefill["c"]
        d = self._prefill["d"]
        log_rate = a * log_bs**3 + b * log_bs**2 + c * log_bs + d
        rate = 2.0 ** log_rate
        return max(0.0, rate * in_tokens)

    def coefficients(self) -> dict:
        """Return all fitted coefficients (for serialization/logging)."""
        return {
            "tpot": dict(self._tpot),
            "prefill": dict(self._prefill),
            "power": dict(self._power) if self._power else None,
        }

    def summary(self) -> str:
        lines = ["LatencyPredictor (formula-based):"]
        t = self._tpot
        lines.append(f"  TPOT:    {t['a']:.6f} * (1 - exp(-{t['b']:.6f} * bs)) + {t['c']:.6f}")
        p = self._prefill
        lines.append(f"  Prefill: log2(rate) = {p['a']:.6f}*log2(bs)^3 + {p['b']:.6f}*log2(bs)^2 + {p['c']:.6f}*log2(bs) + {p['d']:.6f}")
        lines.append("")
        lines.append(f"  {'BS':>5}  {'TPOT(s)':>10}  {'Prefill(s/tok)':>14}", )
        for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768]:
            rate = self.ttft(bs, 1)  # rate = ttft for 1 token
            lines.append(f"  {bs:>5}  {self.tpot(bs):>10.6f}  {rate:>14.8f}")
        return "\n".join(lines)
