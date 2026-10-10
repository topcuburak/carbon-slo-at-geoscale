#!/usr/bin/env python3
"""Build the TPOT / TTFT lookup tables for a model from its drain CSVs.

This is the lookup family's fitting step, the counterpart of
testbed/phase1_calibration/fit_latency.py on the analytical side. It reads the
per-step CSVs a drain campaign produced and writes the tables the simulator
loads from ../table_data/.

The raw drain CSVs are NOT in the repository -- they are hundreds of megabytes
and live on our testbed. See README.md in this directory for where they are and how to
unpack them; `--drain-root` points this script at them.

Reproduces the format used by datacenter/perf_models/lookup/table_data/tpot_lookup_llama3_8B_3ds.json
and ttft_lookup_llama3_8B_3ds.json so the new files plug into the existing
scheduler / GPU simulator code paths.

For each model we ingest 3 datasets (humaneval, mbpp, govreport) and produce:
  datacenter/perf_models/lookup/table_data/tpot_lookup_<key>_3ds.json
  datacenter/perf_models/lookup/table_data/ttft_lookup_<key>_3ds.json
  datacenter/perf_models/lookup/table_data/ttft_fit_<key>_3ds.json           # linear fit
  datacenter/perf_models/lookup/table_data/ttft_fit_<key>_3ds_powerlaw.json  # 2- and 3-param power law

Usage:
    python -m datacenter.perf_models.lookup.campaign.build_lookups
    DRAIN_ROOT=/path/to/unpacked python .../build_lookups.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


_HERE = Path(__file__).resolve().parent
OUT = _HERE.parent / "table_data"          # the tables the simulator loads
#: Where the drain CSVs were unpacked. Overridable, because they are not in the
#: repository: --drain-root, or DRAIN_ROOT in the environment.
DRAIN_ROOT = Path(os.environ.get("DRAIN_ROOT",
                                 _HERE.parent / "measured" / "archives" / "unpacked" / "results"))

# ── Model registry ──────────────────────────────────────────────────────
# Each entry: (key, full_name, tp, hardware, drain_suffix)
MODELS = [
    ("mixtral_8x7b",
     "mistralai/Mixtral-8x7B-Instruct-v0.1",
     4, "A100-40GB x4",
     "drain_mixtral"),
    ("deepseek_moe_16b",
     "deepseek-ai/deepseek-moe-16b-chat",
     2, "A100-40GB x2",
     "drain_moe"),
]
DATASETS = ["humaneval", "mbpp", "govreport"]
WARMUP_GROUPS_DROPPED = 2  # match LLaMA-8B build (drop first 2 groups per BS×ds)
N_KV_BINS = 30
N_TTFT_BINS = 20


# ── Loaders ─────────────────────────────────────────────────────────────
def load_decode_steps(base_root: Path, suffix: str) -> pd.DataFrame:
    """Concatenate per_step CSVs across the 3 datasets, keeping decode rows only.

    Drops the first WARMUP_GROUPS_DROPPED groups per (BS, dataset) to match
    the LLaMA-8B lookup build (their first decode steps were warmup-skewed).
    Adds 'total_kv_before' and 'n_active' columns expected downstream.
    """
    rows = []
    for ds in DATASETS:
        ddir = base_root / f"{ds}_{suffix}"
        for csv in sorted(ddir.glob("per_step_bs*.csv")):
            d = pd.read_csv(csv, low_memory=False)
            d = d[d.step_type == "decode"]
            d["dataset"] = ds
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)

    # Drop warmup groups: per (BS, dataset), groups 0..WARMUP_GROUPS_DROPPED-1
    df = df[df.group_idx >= WARMUP_GROUPS_DROPPED].copy()

    # Normalize column names to what plot_latency_validation.py expects
    df["n_active"] = df.n_decoding
    return df


def load_drain_summaries(base_root: Path, suffix: str) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        sf = base_root / f"{ds}_{suffix}" / "drain_summary.csv"
        if not sf.exists():
            continue
        d = pd.read_csv(sf)
        d["dataset"] = ds
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df = df[df.group_idx >= WARMUP_GROUPS_DROPPED].copy()
    return df


# ── TPOT lookup builder ─────────────────────────────────────────────────
def build_tpot_lookup(decode_df: pd.DataFrame, model_name: str, tp: int,
                       hardware: str) -> dict:
    """Per-BS quantile-binned median step_ms over total_kv_before."""
    bs_values = sorted(decode_df.n_active.unique())
    table = {}
    total_n = 0
    for bs in bs_values:
        sub = decode_df[decode_df.n_active == bs]
        if sub.empty:
            continue
        # Quantile bin edges for equal-sample-count cells
        kv = sub.total_kv_before.values
        edges = np.quantile(kv, np.linspace(0, 1, N_KV_BINS + 1))
        # Make edges strictly increasing (ties at low BS can collapse bins)
        edges = np.unique(edges)
        if len(edges) < 3:
            continue
        bins = pd.cut(sub.total_kv_before, bins=edges, include_lowest=True)
        agg = sub.groupby(bins, observed=True)["step_ms"].agg(["median", "count"])
        # Some bins may be empty if quantile collapse — drop them
        agg = agg.dropna()
        kv_centers = [(iv.left + iv.right) / 2 for iv in agg.index]
        table[str(int(bs))] = {
            "kv_edges": list(map(float, edges)),
            "medians_ms": list(map(float, agg["median"].values)),
            "counts": list(map(int, agg["count"].values)),
        }
        total_n += int(agg["count"].sum())

    # Self-fit MAPE (the lookup applied to its own data)
    pred = []
    actual = []
    for bs in bs_values:
        sub = decode_df[decode_df.n_active == bs]
        if str(int(bs)) not in table or sub.empty:
            continue
        cell = table[str(int(bs))]
        edges = np.asarray(cell["kv_edges"])
        meds = np.asarray(cell["medians_ms"])
        # For each step, find its bin and use its median
        bin_idx = np.digitize(sub.total_kv_before.values, edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(meds) - 1)
        pred.extend(meds[bin_idx])
        actual.extend(sub.step_ms.values)
    pred = np.asarray(pred); actual = np.asarray(actual)
    mape = float(np.mean(np.abs((actual - pred) / np.maximum(actual, 1e-3)))) * 100
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))

    return {
        "model": model_name,
        "datasets": [f"{d}_drain" for d in DATASETS],
        "tp_size": tp,
        "hardware": hardware,
        "quantity": "step_ms",
        "quantity_unit": "ms",
        "unit_per_step": "one decoded token per active sequence",
        "description": (
            "Binned median step_ms for vLLM engine.step() during decode, "
            "indexed by (BS, total_KV). BS = n_decoding (active sequences). "
            "total_KV = sum of (ISL + tokens_generated_so_far) across active "
            "sequences, measured BEFORE the step. Per-BS KV axis uses quantile "
            "bin edges so each cell has ~equal sample count."
        ),
        "fit_quality": {
            "self_mape_percent": round(mape, 2),
            "self_rmse_ms": round(rmse, 4),
        },
        "n_bs_values": len(table),
        "n_kv_bins": N_KV_BINS,
        "warmup_groups_dropped": WARMUP_GROUPS_DROPPED,
        "n_observations": total_n,
        "table": table,
    }


# ── TTFT lookup builder ─────────────────────────────────────────────────
def build_ttft_lookup(summary_df: pd.DataFrame, model_name: str, tp: int) -> dict:
    """Per-BS quantile-binned median prefill_ms over total_isl."""
    bs_values = sorted(summary_df.bs.unique())
    table = {}
    total_n = 0
    for bs in bs_values:
        sub = summary_df[summary_df.bs == bs]
        if len(sub) < 3:
            continue
        edges = np.quantile(sub.total_isl, np.linspace(0, 1, N_TTFT_BINS + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            continue
        bins = pd.cut(sub.total_isl, bins=edges, include_lowest=True)
        agg = sub.groupby(bins, observed=True)["prefill_ms"].agg(["median", "count"])
        agg = agg.dropna()
        table[str(int(bs))] = {
            "total_prefill_edges": list(map(float, edges)),
            "medians_ms": list(map(float, agg["median"].values)),
            "counts": list(map(int, agg["count"].values)),
        }
        total_n += int(agg["count"].sum())

    # 80/20 holdout MAPE
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(summary_df))
    cut = int(len(idx) * 0.8)
    train = summary_df.iloc[idx[:cut]]
    test = summary_df.iloc[idx[cut:]]
    # Build train-only lookup, apply to test
    train_table = {}
    for bs in sorted(train.bs.unique()):
        sub = train[train.bs == bs]
        if len(sub) < 3: continue
        edges = np.quantile(sub.total_isl, np.linspace(0, 1, N_TTFT_BINS + 1))
        edges = np.unique(edges)
        if len(edges) < 3: continue
        bins = pd.cut(sub.total_isl, bins=edges, include_lowest=True)
        meds = sub.groupby(bins, observed=True)["prefill_ms"].median().dropna().values
        train_table[int(bs)] = (edges, meds)

    pred = []; actual = []
    for _, row in test.iterrows():
        bs = int(row.bs)
        if bs not in train_table: continue
        edges, meds = train_table[bs]
        idx_b = max(0, min(np.digitize(row.total_isl, edges) - 1, len(meds) - 1))
        pred.append(meds[idx_b])
        actual.append(row.prefill_ms)
    pred = np.asarray(pred); actual = np.asarray(actual)
    holdout_mape = float(np.mean(np.abs((actual - pred) / np.maximum(actual, 1e-3)))) * 100 if len(pred) else float("nan")

    return {
        "model": model_name,
        "datasets": [f"{d}_drain" for d in DATASETS],
        "tp_size": tp,
        "quantity": "prefill_ms",
        "description": f"Binned median prefill_ms (dropped first {WARMUP_GROUPS_DROPPED} warmup groups per BS×dataset).",
        "holdout_mape_percent": round(holdout_mape, 2),
        "n_observations": int(total_n),
        "warmup_groups_dropped": WARMUP_GROUPS_DROPPED,
        "n_bs_values": len(table),
        "n_ttft_bins": N_TTFT_BINS,
        "table": table,
    }


# ── TTFT fits ────────────────────────────────────────────────────────────
def build_ttft_linear_fit(summary_df: pd.DataFrame, model_name: str, tp: int) -> dict:
    """3-parameter linear fit: prefill_ms ≈ a0 + a1·total_isl + a2·BS"""
    X = np.column_stack([np.ones(len(summary_df)),
                          summary_df.total_isl.values,
                          summary_df.bs.values])
    y = summary_df.prefill_ms.values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, a1, a2 = map(float, coef)
    # Holdout MAPE
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(summary_df))
    cut = int(0.8 * len(idx))
    train_X, train_y = X[idx[:cut]], y[idx[:cut]]
    test_X, test_y = X[idx[cut:]], y[idx[cut:]]
    coef_tr, *_ = np.linalg.lstsq(train_X, train_y, rcond=None)
    pred_test = test_X @ coef_tr
    mape = float(np.mean(np.abs((test_y - pred_test) / np.maximum(test_y, 1e-3)))) * 100
    return {
        "model": model_name,
        "datasets": [f"{d}_drain" for d in DATASETS],
        "tp_size": tp,
        "formula_3param": "prefill_ms ≈ a0 + a1·total_prefill_tokens + a2·BS",
        "coefficients_3param": {"a0": a0, "a1": a1, "a2": a2},
        "holdout_mape_percent": round(mape, 2),
        "n_observations": int(len(summary_df)),
        "warmup_groups_dropped": WARMUP_GROUPS_DROPPED,
    }


def build_ttft_powerlaw_fit(summary_df: pd.DataFrame, model_name: str, tp: int) -> dict:
    """Power-law fits: 2-param (a·N^b) and 3-param (a·N^b·BS^c)."""
    df = summary_df[(summary_df.total_isl > 0) & (summary_df.prefill_ms > 0)]
    N = df.total_isl.values; bs = df.bs.values; y = df.prefill_ms.values

    # 2-param: log y = log a + b·log N
    A2 = np.column_stack([np.ones(len(N)), np.log(N)])
    sol, *_ = np.linalg.lstsq(A2, np.log(y), rcond=None)
    a_2, b_2 = float(np.exp(sol[0])), float(sol[1])
    pred2 = a_2 * np.power(N, b_2)
    mape2 = float(np.mean(np.abs((y - pred2) / np.maximum(y, 1e-3)))) * 100

    # 3-param: log y = log a + b·log N + c·log BS
    A3 = np.column_stack([np.ones(len(N)), np.log(N), np.log(bs)])
    sol, *_ = np.linalg.lstsq(A3, np.log(y), rcond=None)
    a_3, b_3, c_3 = float(np.exp(sol[0])), float(sol[1]), float(sol[2])
    pred3 = a_3 * np.power(N, b_3) * np.power(bs, c_3)
    mape3 = float(np.mean(np.abs((y - pred3) / np.maximum(y, 1e-3)))) * 100

    return {
        "model": model_name,
        "datasets": [f"{d}_drain" for d in DATASETS],
        "tp_size": tp,
        "formula_powerlaw_2param": "prefill_ms ≈ a · total_prefill_tokens^b",
        "coefficients_powerlaw_2param": {"a": a_2, "b": b_2},
        "mape_powerlaw_2param": round(mape2, 2),
        "formula_powerlaw_3param": "prefill_ms ≈ a · total_prefill_tokens^b · BS^c",
        "coefficients_powerlaw_3param": {"a": a_3, "b": b_3, "c": c_3},
        "mape_powerlaw_3param": round(mape3, 2),
    }


# ── Driver ───────────────────────────────────────────────────────────────
def main():
    for key, full_name, tp, hw, suffix in MODELS:
        base = DRAIN_ROOT
        print(f"\n=== {full_name} (key={key}) ===")
        decode = load_decode_steps(base, suffix)
        summary = load_drain_summaries(base, suffix)
        print(f"  loaded: {len(decode):,} decode steps  /  {len(summary):,} groups")

        if decode.empty or summary.empty:
            print(f"  SKIP: missing data")
            continue

        tpot = build_tpot_lookup(decode, full_name, tp, hw)
        ttft_lk = build_ttft_lookup(summary, full_name, tp)
        ttft_lin = build_ttft_linear_fit(summary, full_name, tp)
        ttft_pow = build_ttft_powerlaw_fit(summary, full_name, tp)

        for name, obj in [
            (f"tpot_lookup_{key}_3ds.json", tpot),
            (f"ttft_lookup_{key}_3ds.json", ttft_lk),
            (f"ttft_fit_{key}_3ds.json", ttft_lin),
            (f"ttft_fit_{key}_3ds_powerlaw.json", ttft_pow),
        ]:
            path = OUT / name
            with open(path, "w") as f:
                json.dump(obj, f, indent=2)
            print(f"  wrote {path}")

        print(f"\n  TPOT lookup self-MAPE: {tpot['fit_quality']['self_mape_percent']}%")
        print(f"  TTFT lookup holdout-MAPE: {ttft_lk['holdout_mape_percent']}%")
        print(f"  TTFT linear fit:    a0={ttft_lin['coefficients_3param']['a0']:.3f}  a1={ttft_lin['coefficients_3param']['a1']:.5f}  a2={ttft_lin['coefficients_3param']['a2']:.4f}  MAPE={ttft_lin['holdout_mape_percent']}%")
        print(f"  TTFT powerlaw 2p:   a={ttft_pow['coefficients_powerlaw_2param']['a']:.4f}  b={ttft_pow['coefficients_powerlaw_2param']['b']:.4f}  MAPE={ttft_pow['mape_powerlaw_2param']}%")
        print(f"  TTFT powerlaw 3p:   a={ttft_pow['coefficients_powerlaw_3param']['a']:.4f}  b={ttft_pow['coefficients_powerlaw_3param']['b']:.4f}  c={ttft_pow['coefficients_powerlaw_3param']['c']:.4f}  MAPE={ttft_pow['mape_powerlaw_3param']}%")


if __name__ == "__main__":
    main()
