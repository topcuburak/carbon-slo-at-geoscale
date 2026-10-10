"""Latency and power lookup tables built from vLLM benchmark results.

Loads a CSV produced by ``scripts/benchmark_vllm.py`` and provides
interpolated TPOT and TTFT for any batch size.

TTFT model:
    A piecewise power-law fitted from benchmark data as a function of
    total prefill tokens in the batch:

        if total_prefill <= breakpoint:
            ttft = a_lo * total_prefill ^ b_lo
        else:
            ttft = a_hi * total_prefill ^ b_hi

    Falls back to per-BS prefill-rate model if fitting fails.

Latency only. This backend used to load a power CSV as well; see the note in
__init__ for why it does not any more.

Expected latency CSV columns:
    sample_id, batch_size, in_tokens, out_tokens, ttft, tpot, inf_latency
"""

import csv
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from pathlib import Path

from datacenter.perf_models.base import LatencyBackend

log = logging.getLogger("datacenter")


def _power_func(x, a, b):
    """Power-law: y = a * x^b."""
    return a * np.power(x, b)


_TABLE_DIR = Path(__file__).resolve().parent / "table_data"


class LatencyLookup(LatencyBackend):
    """Direct TTFT lookup from BS=1 benchmark data + interpolated TPOT/Power."""

    # Class-level defaults for the binned (BS, total_KV) → step_ms table and
    # the binned (BS, total_prefill_tokens) → prefill_ms table built from the
    # joint 3-dataset drain pool. Set to None to disable.
    DEFAULT_STEP_MS_JSON = str(_TABLE_DIR / "tpot_lookup_llama3_8B_3ds.json")
    DEFAULT_TTFT_JSON = str(_TABLE_DIR / "ttft_lookup_llama3_8B_3ds.json")

    # Per-model overrides — used when a caller passes ``model_name=`` to
    # resolve the right JSON pair. Datacenters always use lookup-based
    # latency for all models (the scheduler also uses lookup but only for
    # MoE models — see datacenter/latency_analytical.py).
    _MODEL_LOOKUP_PATHS = {
        "llama3_8B": (
            str(_TABLE_DIR / "tpot_lookup_llama3_8B_3ds.json"),
            str(_TABLE_DIR / "ttft_lookup_llama3_8B_3ds.json"),
        ),
        "mixtral_8x7b": (
            str(_TABLE_DIR / "tpot_lookup_mixtral_8x7b_3ds.json"),
            str(_TABLE_DIR / "ttft_lookup_mixtral_8x7b_3ds.json"),
        ),
        "deepseek_moe_16b": (
            str(_TABLE_DIR / "tpot_lookup_deepseek_moe_16b_3ds.json"),
            str(_TABLE_DIR / "ttft_lookup_deepseek_moe_16b_3ds.json"),
        ),
    }

    def __init__(
        self,
        csv_path: str,
        step_ms_lookup_json: Optional[str] = None,
        ttft_lookup_json: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        # If model_name is provided and JSON paths weren't explicitly given,
        # resolve to the model-specific lookup files.
        if model_name and model_name in self._MODEL_LOOKUP_PATHS:
            tpot_p, ttft_p = self._MODEL_LOOKUP_PATHS[model_name]
            if step_ms_lookup_json is None:
                step_ms_lookup_json = tpot_p
            if ttft_lookup_json is None:
                ttft_lookup_json = ttft_p
        # raw data grouped by batch size
        tpot_by_bs: Dict[int, List[float]] = defaultdict(list)

        # BS=1 per-request (ISL → TTFT) for direct lookup.
        # Since the trace is drawn from the same dataset samples as the
        # benchmarks, every (dataset, ISL) in the trace has exact matches
        # in the benchmark data. We store the mean TTFT per (dataset, ISL)
        # pair for O(1) lookup — no interpolation needed.
        bs1_by_key: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        bs1_all: Dict[int, List[float]] = defaultdict(list)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                bs = int(row["batch_size"])
                tpot_by_bs[bs].append(float(row["tpot"]))
                if bs == 1:
                    isl = int(row["in_tokens"])
                    ttft = float(row["ttft"])
                    dataset = row.get("dataset", "")
                    bs1_by_key[(dataset, isl)].append(ttft)
                    bs1_all[isl].append(ttft)

        # compute averages → sorted list of (bs, avg_value)
        self._tpot_points: List[Tuple[int, float]] = sorted(
            (bs, sum(v) / len(v)) for bs, v in tpot_by_bs.items()
        )

        if not self._tpot_points:
            raise ValueError(f"No data loaded from {csv_path}")

        # ── BS=1 TTFT lookup dict: (dataset, ISL) → mean TTFT (seconds) ──
        self._ttft_exact: Dict[Tuple[str, int], float] = {
            key: sum(vals) / len(vals)
            for key, vals in bs1_by_key.items()
        }
        # Fallback: ISL → mean TTFT across all datasets (for unknown dataset)
        self._ttft_fallback: Dict[int, float] = {
            isl: sum(vals) / len(vals)
            for isl, vals in bs1_all.items()
        }

        n_keys = len(self._ttft_exact)
        n_datasets = len(set(ds for ds, _ in self._ttft_exact))
        log.info("TTFT lookup: %d (dataset, ISL) pairs across %d datasets",
                 n_keys, n_datasets)

        # ── binned (BS, total_KV) → step_ms lookup (3-dataset pool) ──
        self._step_ms_table: Dict[int, dict] = {}
        self._step_ms_bs_keys: List[int] = []
        step_path = step_ms_lookup_json or self.DEFAULT_STEP_MS_JSON
        if step_path and os.path.isfile(step_path):
            self._load_step_ms_lookup(step_path)
            log.info("step_ms lookup loaded: %d BS cells from %s",
                     len(self._step_ms_table), step_path)
        else:
            log.info("step_ms lookup not loaded (path=%s not found)", step_path)

        # ── binned (BS, total_prefill_tokens) → prefill_ms lookup ──
        self._ttft_batch_table: Dict[int, dict] = {}
        self._ttft_batch_bs_keys: List[int] = []
        ttft_path = ttft_lookup_json or self.DEFAULT_TTFT_JSON
        if ttft_path and os.path.isfile(ttft_path):
            self._load_ttft_batch_lookup(ttft_path)
            log.info("ttft batch lookup loaded: %d BS cells from %s",
                     len(self._ttft_batch_table), ttft_path)
        else:
            log.info("ttft batch lookup not loaded (path=%s not found)", ttft_path)

        # There is deliberately no power table here. Power came from a
        # per-model power.csv interpolated on batch size, and every consumer
        # of it is gone: energy is integrated from the measured
        # StepEnergyModel for any calibrated (model, TP), and an uncalibrated
        # one falls back to the analytical log-linear curve. Keeping the CSV
        # reader meant a run could silently pick up watts in the wrong
        # convention -- the LLaMA post-processor wrote per-GPU values and the
        # Mixtral one wrote per-pair, and nothing in the file said which.

    # ------------------------------------------------------------------
    # Binned lookup loaders
    # ------------------------------------------------------------------
    def _load_step_ms_lookup(self, path: str) -> None:
        with open(path) as f:
            payload = json.load(f)
        table = payload.get("table", {})
        self._step_ms_table = {
            int(bs): {
                "kv_edges": np.asarray(cell["kv_edges"], dtype=float),
                "medians_ms": np.asarray(cell["medians_ms"], dtype=float),
            }
            for bs, cell in table.items()
        }
        self._step_ms_bs_keys = sorted(self._step_ms_table.keys())

    def _load_ttft_batch_lookup(self, path: str) -> None:
        with open(path) as f:
            payload = json.load(f)
        table = payload.get("table", {})
        self._ttft_batch_table = {
            int(bs): {
                "edges": np.asarray(cell["total_prefill_edges"], dtype=float),
                "medians_ms": np.asarray(cell["medians_ms"], dtype=float),
            }
            for bs, cell in table.items()
        }
        self._ttft_batch_bs_keys = sorted(self._ttft_batch_table.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def tpot(self, batch_size: int) -> float:
        """Return interpolated/extrapolated TPOT (seconds) for *batch_size*.

        Legacy BS-only API. Prefer ``step_ms(bs, total_kv)`` for the simulator.
        """
        return self._interpolate(self._tpot_points, batch_size)

    def step_ms(self, bs: int, total_kv: int) -> float:
        """Return per-engine-step decode wall time (milliseconds) via the
        binned ``(BS, total_KV)`` lookup built from the 3-dataset drain pool.

        Falls back to BS-only TPOT (converted to ms) if the binned table
        was not loaded.
        """
        if not self._step_ms_table:
            return self._interpolate(self._tpot_points, max(1, int(bs))) * 1000.0
        bs = max(1, int(bs))
        if bs not in self._step_ms_table:
            bs = min(self._step_ms_bs_keys,
                     key=lambda b: abs(b - bs))
        cell = self._step_ms_table[bs]
        edges = cell["kv_edges"]
        meds = cell["medians_ms"]
        idx = int(np.clip(np.searchsorted(edges, total_kv, side="right") - 1,
                          0, len(meds) - 1))
        return float(meds[idx])

    def step_s(self, bs: int, total_kv: int) -> float:
        """Convenience: ``step_ms`` in seconds."""
        return self.step_ms(bs, total_kv) / 1000.0

    def ttft_batch_s(self, bs: int, total_prefill_tokens: int) -> float:
        """Return batched prefill wall time (seconds) via the binned
        ``(BS, total_prefill_tokens)`` lookup built from drain iter-0 data.

        Falls back to summing per-(dataset, ISL) BS=1 lookups if the binned
        table is not loaded (legacy behavior).
        """
        if not self._ttft_batch_table or total_prefill_tokens <= 0:
            return 0.0
        bs = max(1, int(bs))
        if bs not in self._ttft_batch_table:
            bs = min(self._ttft_batch_bs_keys,
                     key=lambda b: abs(b - bs))
        cell = self._ttft_batch_table[bs]
        edges = cell["edges"]
        meds = cell["medians_ms"]
        idx = int(np.clip(np.searchsorted(edges, total_prefill_tokens, side="right") - 1,
                          0, len(meds) - 1))
        return float(meds[idx]) / 1000.0

    def has_step_ms_lookup(self) -> bool:
        return bool(self._step_ms_table)

    def has_ttft_batch_lookup(self) -> bool:
        return bool(self._ttft_batch_table)

    def ttft(self, batch_size: int, in_tokens: int, dataset: str = "") -> float:
        """Return TTFT (seconds) for a single request.

        batch_size is accepted for LatencyBackend API compatibility
        (the provisioner calls ttft(bs, ref_isl)) but is not used —
        TTFT depends only on ISL. The actual batch-internal queueing
        is handled by inference_engine's chunked prefill loop.
        """
        return self.ttft_by_isl(in_tokens, dataset=dataset)

    def ttft_by_isl(self, isl: int, dataset: str = "") -> float:
        """Return mean TTFT (seconds) from BS=1 benchmark data.

        Exact (dataset, ISL) match → returns mean of all BS=1
        measurements at that point. Falls back to ISL-only mean
        (across all datasets) if dataset is empty or not found.
        """
        # Try exact (dataset, ISL) match first
        if dataset:
            val = self._ttft_exact.get((dataset, isl))
            if val is not None:
                return val

        # Fallback: ISL-only (mean across all datasets)
        val = self._ttft_fallback.get(isl)
        if val is not None:
            return val

        # No match at all — shouldn't happen if trace matches benchmarks.
        # Log a warning and return a rough estimate from the analytical
        # overhead constant.
        log.warning("TTFT lookup miss: dataset=%s ISL=%d — no benchmark data", dataset, isl)
        return 0.037  # ~37ms fallback (LLaMA-8B short-prompt overhead)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _interpolate(points: List[Tuple[int, float]], x: int) -> float:
        """Linear inter/extrapolation on sorted (x, y) points."""
        if len(points) == 1:
            return points[0][1]

        # Clamp / extrapolate
        if x <= points[0][0]:
            # extrapolate below using first two points
            x0, y0 = points[0]
            x1, y1 = points[1]
            if x1 == x0:
                return y0
            slope = (y1 - y0) / (x1 - x0)
            return max(0.0, y0 + slope * (x - x0))

        if x >= points[-1][0]:
            # extrapolate above using last two points
            x0, y0 = points[-2]
            x1, y1 = points[-1]
            if x1 == x0:
                return y1
            slope = (y1 - y0) / (x1 - x0)
            return max(0.0, y1 + slope * (x - x1))

        # interpolate between surrounding points
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            if x0 <= x <= x1:
                if x1 == x0:
                    return y0
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

        return points[-1][1]  # fallback

    def summary(self) -> str:
        lines = ["LatencyLookup table:"]
        lines.append(f"  {'BS':>5}  {'TPOT(s)':>10}")
        for bs_t, tpot_v in self._tpot_points:
            lines.append(f"  {bs_t:>5}  {tpot_v:>10.6f}")

        # Binned TTFT lookup (new 3-dataset pool)
        if self.has_ttft_batch_lookup():
            lines.append(
                f"\nBatched TTFT lookup: {len(self._ttft_batch_table)} BS cells "
                f"(binned (BS, total_prefill_tokens) → prefill_ms)"
            )
        else:
            lines.append(
                f"\nBatched TTFT lookup: not loaded. "
                f"Falling back to per-(dataset, ISL) BS=1 lookup "
                f"({len(self._ttft_exact)} exact keys)."
            )

        # Binned step_ms (TPOT) lookup
        if self.has_step_ms_lookup():
            lines.append(
                f"Binned step_ms lookup: {len(self._step_ms_table)} BS cells "
                f"(binned (BS, total_KV) → step_ms)"
            )
        else:
            lines.append("Binned step_ms lookup: not loaded (falling back to BS-only TPOT).")

        return "\n".join(lines)
