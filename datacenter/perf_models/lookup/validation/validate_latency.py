"""Validate TTFT and TPOT predictions against the measured benchmarks.

Scores this package's own latency predictions -- the interpolated tables and
the fitted curves -- against `measured/summaries/`, on a held-out split. It
lived under the output-length predictor because it borrows one directory parser
from there; it has nothing to do with output length.

Fits analytical models (lookup interpolation + formula-based curves) on a
train split and evaluates prediction accuracy on a held-out test split.

Reports per-model, per-batch-size MAE, MAPE, and R² for both TTFT and TPOT.

Usage:
    python -m datacenter.perf_models.lookup.validate_latency --inferences-dir datacenter/perf_models/lookup/measured/summaries

    # Single model
    python -m datacenter.perf_models.lookup.validate_latency --inferences-dir datacenter/perf_models/lookup/measured/summaries \
        --models llama3_8B
"""

import argparse
import csv
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
)
log = logging.getLogger("validate_latency")

# Reuse benchmark_loader's directory parser
# One helper, borrowed: the benchmark directory naming is the same on both
# sides, and duplicating the parser would let the two drift.
from osl_prediction.benchmark_loader import _parse_benchmark_dir


def load_raw_benchmarks(inferences_dir, models=None):
    """Load raw per-sample benchmark data (not deduplicated).

    Returns list of dicts with: model, dataset, batch_size, in_tokens,
    out_tokens, ttft, tpot, inf_latency
    """
    inferences_dir = Path(inferences_dir)
    all_rows = []

    for subdir in sorted(inferences_dir.iterdir()):
        if not subdir.is_dir():
            continue
        model_key, dataset = _parse_benchmark_dir(subdir.name)
        if model_key is None:
            continue
        if models and model_key not in models:
            continue

        csv_path = subdir / "merged_benchmark.csv"
        if not csv_path.exists():
            continue

        with open(csv_path) as f:
            for row in csv.DictReader(f):
                all_rows.append({
                    "model": model_key,
                    "dataset": dataset,
                    "batch_size": int(row["batch_size"]),
                    "in_tokens": int(row["in_tokens"]),
                    "out_tokens": int(row["out_tokens"]),
                    "ttft": float(row["ttft"]),
                    "tpot": float(row["tpot"]),
                    "inf_latency": float(row["inf_latency"]),
                })

    return all_rows


def compute_metrics(actual, predicted):
    """Compute MAE, MAPE, R² between actual and predicted arrays."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    ae = np.abs(actual - predicted)
    mae = ae.mean()
    median_ae = np.median(ae)

    nonzero = actual > 0
    mape = np.mean(ae[nonzero] / actual[nonzero]) * 100 if nonzero.any() else 0

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "mae": float(mae),
        "median_ae": float(median_ae),
        "mape": float(mape),
        "r2": float(r2),
        "n": len(actual),
    }


def fit_tpot_model(rows):
    """Fit TPOT = a * (1 - exp(-b * bs)) + c from training data."""
    from scipy.optimize import curve_fit

    bs_to_tpot = defaultdict(list)
    for r in rows:
        bs_to_tpot[r["batch_size"]].append(r["tpot"])

    xs = []
    ys = []
    for bs, vals in sorted(bs_to_tpot.items()):
        xs.append(bs)
        ys.append(np.mean(vals))

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    def model(x, a, b, c):
        return a * (1.0 - np.exp(-b * x)) + c

    try:
        popt, _ = curve_fit(model, xs, ys, p0=[ys.max(), 0.02, ys.min()], maxfev=10000)
        coeffs = {"a": popt[0], "b": popt[1], "c": popt[2]}
    except Exception as e:
        log.warning("TPOT curve fit failed: %s, using linear interpolation", e)
        coeffs = None

    return coeffs, list(zip(xs.astype(int), ys))


def predict_tpot_curve(bs, coeffs):
    """Predict TPOT from fitted saturating curve."""
    return max(0, coeffs["a"] * (1.0 - np.exp(-coeffs["b"] * bs)) + coeffs["c"])


def interpolate_tpot(bs, points):
    """Linear interpolation from sorted (bs, tpot) points."""
    if bs <= points[0][0]:
        return points[0][1]
    if bs >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= bs <= x1:
            t = (bs - x0) / (x1 - x0) if x1 != x0 else 0
            return y0 + t * (y1 - y0)
    return points[-1][1]


def fit_ttft_model(rows):
    """Fit piecewise power-law TTFT model on total_prefill_tokens."""
    from scipy.optimize import curve_fit

    # Group by batch_size, compute total_prefill per batch
    bs_to_samples = defaultdict(list)
    for r in rows:
        bs_to_samples[r["batch_size"]].append(r)

    tp_list = []
    ttft_list = []

    for bs, samples in sorted(bs_to_samples.items()):
        for i in range(0, len(samples) - bs + 1, bs):
            batch = samples[i:i + bs]
            if len(batch) < bs:
                break
            tp = sum(s["in_tokens"] for s in batch)
            max_ttft = max(s["ttft"] for s in batch)
            tp_list.append(tp)
            ttft_list.append(max_ttft)

    tp_arr = np.array(tp_list, dtype=float)
    ttft_arr = np.array(ttft_list, dtype=float)

    if len(tp_arr) < 10:
        log.warning("Too few batches for TTFT model")
        return None

    # Also fit simple per-BS average for interpolation
    bs_to_ttft_rate = defaultdict(list)
    for r in rows:
        if r["in_tokens"] > 0:
            bs_to_ttft_rate[r["batch_size"]].append(r["ttft"] / r["in_tokens"])

    ttft_rate_points = sorted(
        (bs, np.mean(vals)) for bs, vals in bs_to_ttft_rate.items()
    )

    # Piecewise power-law
    def power_func(x, a, b):
        return a * np.power(x, b)

    best_err = float("inf")
    best_result = None
    candidates = np.percentile(tp_arr, np.arange(15, 85, 5))

    for bp in candidates:
        mask_lo = tp_arr <= bp
        mask_hi = tp_arr > bp
        if mask_lo.sum() < 5 or mask_hi.sum() < 5:
            continue
        try:
            p_lo, _ = curve_fit(power_func, tp_arr[mask_lo], ttft_arr[mask_lo],
                                p0=[1e-5, 1.0], maxfev=5000)
            p_hi, _ = curve_fit(power_func, tp_arr[mask_hi], ttft_arr[mask_hi],
                                p0=[1e-5, 1.0], maxfev=5000)
            pred_lo = power_func(tp_arr[mask_lo], *p_lo)
            pred_hi = power_func(tp_arr[mask_hi], *p_hi)
            err = (np.mean(np.abs(pred_lo - ttft_arr[mask_lo])) +
                   np.mean(np.abs(pred_hi - ttft_arr[mask_hi]))) / 2
            if err < best_err:
                best_err = err
                best_result = {
                    "breakpoint": float(bp),
                    "a_lo": float(p_lo[0]), "b_lo": float(p_lo[1]),
                    "a_hi": float(p_hi[0]), "b_hi": float(p_hi[1]),
                }
        except Exception:
            continue

    return {
        "piecewise": best_result,
        "rate_points": ttft_rate_points,
    }


def predict_ttft_piecewise(total_prefill, model):
    """Predict TTFT from piecewise power-law."""
    if model is None:
        return 0
    p = model
    tp = max(1.0, float(total_prefill))
    if tp <= p["breakpoint"]:
        return p["a_lo"] * (tp ** p["b_lo"])
    else:
        return p["a_hi"] * (tp ** p["b_hi"])


def main():
    parser = argparse.ArgumentParser(description="Validate TTFT/TPOT predictions")
    parser.add_argument("--inferences-dir", type=str, required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log.info("Loading benchmark data from %s ...", args.inferences_dir)
    all_rows = load_raw_benchmarks(args.inferences_dir, models=args.models)
    log.info("Loaded %d raw samples", len(all_rows))

    # Group by model
    by_model = defaultdict(list)
    for r in all_rows:
        by_model[r["model"]].append(r)

    for model_key in sorted(by_model.keys()):
        rows = by_model[model_key]
        log.info("=" * 70)
        log.info("MODEL: %s  (%d samples)", model_key, len(rows))
        log.info("=" * 70)

        # Train/test split by sample index within each batch_size
        rng = np.random.RandomState(args.seed)
        indices = np.arange(len(rows))
        rng.shuffle(indices)
        n_test = int(len(rows) * args.test_frac)
        test_idx = set(indices[:n_test])
        train_rows = [rows[i] for i in range(len(rows)) if i not in test_idx]
        test_rows = [rows[i] for i in range(len(rows)) if i in test_idx]

        log.info("Split: train=%d  test=%d", len(train_rows), len(test_rows))

        # ── TPOT ──
        log.info("\n--- TPOT Prediction ---")
        tpot_coeffs, tpot_points = fit_tpot_model(train_rows)

        if tpot_coeffs:
            log.info("Fitted TPOT curve: a=%.6f  b=%.6f  c=%.6f",
                     tpot_coeffs["a"], tpot_coeffs["b"], tpot_coeffs["c"])

        # Evaluate per batch_size
        bs_to_test = defaultdict(list)
        for r in test_rows:
            bs_to_test[r["batch_size"]].append(r)

        log.info("\nTPOT per batch_size (curve fit):")
        log.info("  %5s  %8s  %8s  %8s  %8s  %6s  %5s",
                 "BS", "act_mean", "pred", "MAE", "MAPE%", "R²", "n")

        all_tpot_actual = []
        all_tpot_pred_curve = []
        all_tpot_pred_interp = []

        for bs in sorted(bs_to_test.keys()):
            samples = bs_to_test[bs]
            actual = [s["tpot"] for s in samples]
            if tpot_coeffs:
                pred_c = [predict_tpot_curve(bs, tpot_coeffs)] * len(actual)
            else:
                pred_c = [interpolate_tpot(bs, tpot_points)] * len(actual)
            pred_i = [interpolate_tpot(bs, tpot_points)] * len(actual)

            m = compute_metrics(actual, pred_c)
            log.info("  %5d  %8.6f  %8.6f  %8.6f  %7.1f  %5.3f  %5d",
                     bs, np.mean(actual), pred_c[0], m["mae"], m["mape"], m["r2"], m["n"])

            all_tpot_actual.extend(actual)
            all_tpot_pred_curve.extend(pred_c)
            all_tpot_pred_interp.extend(pred_i)

        if all_tpot_actual:
            m_curve = compute_metrics(all_tpot_actual, all_tpot_pred_curve)
            m_interp = compute_metrics(all_tpot_actual, all_tpot_pred_interp)
            log.info("\nTPOT overall (curve):  MAE=%.6f  MAPE=%.1f%%  R²=%.4f",
                     m_curve["mae"], m_curve["mape"], m_curve["r2"])
            log.info("TPOT overall (interp): MAE=%.6f  MAPE=%.1f%%  R²=%.4f",
                     m_interp["mae"], m_interp["mape"], m_interp["r2"])

        # ── TTFT ──
        log.info("\n--- TTFT Prediction ---")
        ttft_model = fit_ttft_model(train_rows)

        if ttft_model and ttft_model["piecewise"]:
            p = ttft_model["piecewise"]
            log.info("Fitted piecewise TTFT: bp=%.0f  lo=%.2e*tp^%.4f  hi=%.2e*tp^%.4f",
                     p["breakpoint"], p["a_lo"], p["b_lo"], p["a_hi"], p["b_hi"])

        log.info("\nTTFT per batch_size (piecewise power-law):")
        log.info("  %5s  %8s  %8s  %8s  %8s  %6s  %5s",
                 "BS", "act_mean", "pred_mean", "MAE", "MAPE%", "R²", "n")

        all_ttft_actual = []
        all_ttft_pred = []

        for bs in sorted(bs_to_test.keys()):
            samples = bs_to_test[bs]
            actual = [s["ttft"] for s in samples]

            if ttft_model and ttft_model["piecewise"]:
                # Predict using total_prefill = bs * in_tokens
                predicted = [predict_ttft_piecewise(
                    bs * s["in_tokens"], ttft_model["piecewise"]
                ) for s in samples]
            else:
                # Fallback: use rate interpolation
                rate_points = ttft_model["rate_points"] if ttft_model else []
                if rate_points:
                    rate = interpolate_tpot(bs, rate_points)  # reuse interpolator
                    predicted = [rate * s["in_tokens"] for s in samples]
                else:
                    predicted = [0] * len(actual)

            m = compute_metrics(actual, predicted)
            log.info("  %5d  %8.4f  %8.4f  %8.4f  %7.1f  %5.3f  %5d",
                     bs, np.mean(actual), np.mean(predicted),
                     m["mae"], m["mape"], m["r2"], m["n"])

            all_ttft_actual.extend(actual)
            all_ttft_pred.extend(predicted)

        if all_ttft_actual:
            m_ttft = compute_metrics(all_ttft_actual, all_ttft_pred)
            log.info("\nTTFT overall: MAE=%.4f  MAPE=%.1f%%  R²=%.4f",
                     m_ttft["mae"], m_ttft["mape"], m_ttft["r2"])

        # ── Total Inference Latency ──
        log.info("\n--- Total Latency (TTFT + TPOT * out_tokens) ---")
        all_lat_actual = []
        all_lat_pred = []

        for r in test_rows:
            actual_lat = r["inf_latency"]
            bs = r["batch_size"]

            if tpot_coeffs:
                pred_tpot = predict_tpot_curve(bs, tpot_coeffs)
            else:
                pred_tpot = interpolate_tpot(bs, tpot_points)

            if ttft_model and ttft_model["piecewise"]:
                pred_ttft = predict_ttft_piecewise(
                    bs * r["in_tokens"], ttft_model["piecewise"])
            else:
                pred_ttft = 0

            pred_lat = pred_ttft + pred_tpot * r["out_tokens"]
            all_lat_actual.append(actual_lat)
            all_lat_pred.append(pred_lat)

        if all_lat_actual:
            m_lat = compute_metrics(all_lat_actual, all_lat_pred)
            log.info("Total latency: MAE=%.4f  MAPE=%.1f%%  R²=%.4f  MedAE=%.4f",
                     m_lat["mae"], m_lat["mape"], m_lat["r2"], m_lat["median_ae"])


if __name__ == "__main__":
    main()
