#!/usr/bin/env python3
"""Split a cell's TPOT into the two terms that produce it.

    PYTHONPATH=.:proto python3 scripts/sweep/tpot_decompose.py results/grid_v3/carbon_aware_a0.25/rate_3200
    PYTHONPATH=.:proto python3 scripts/sweep/tpot_decompose.py results/grid_v3      # every cell

Measured TPOT is wall clock per output token, so it is a PRODUCT, not a sum:

    tpot  =  step(bs, kv)  x  1 / (1 - prefill_duty)

The first term is how long one decode step takes at the batch the placement
produced -- it grows with batch even with no prefill anywhere. The second is
that decode does not run at all while a prefill pass holds the engine. A policy
that concentrates raises BOTH, so a single TPOT number cannot say which one is
spending the budget, and the two have different fixes: the first is a placement
question, the second is a scheduling one.

`implied step` is the model-free version of term one: measured TPOT times
(1 - duty). It is worth more than the fitted column beside it, because at these
batches total_KV runs past the fit's `kv_max` envelope, where
[[ttft_estimator_repair]] measured the decode model ~1.9x optimistic. Read
`fit step` as a LOWER BOUND. If `implied` sits far above `fit`, either the
engine is outside the envelope or duty is not the whole of the second term.

prefill_duty is CUMULATIVE per engine, so the site's last sample is its run
average -- percentiles over epochs would describe the counter warming up, not
the fleet. Batch is instantaneous, so it gets percentiles, taken over plateau
epochs only (fleet decode above `--plateau` of its peak) so that warm-up and
drain do not drag the median down.
"""

import argparse
import collections
import csv
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def read_series(path, plateau):
    """per-site: plateau batch samples, engine count, and plateau duty.

    prefill_duty is a WINDOWED rate at the source as of 180df50 -- the share of
    the last DUTY_WINDOW_S the engine spent with decode blocked. So the plateau
    value is the mean of the samples inside it, and nothing has to be
    reconstructed.

    It was cumulative-since-boot before that, and this reader differenced
    across the plateau to recover the window. Doing both is double counting:
    it multiplies a rate by an elapsed time as though it were a total, and
    returns noise that looks like real per-site variation. Runs recorded before
    180df50 therefore report a duty this reader cannot correct, and their
    `cum` column is the number their scheduler actually used.
    """
    rows = collections.defaultdict(list)
    fleet = collections.Counter()
    for r in csv.DictReader(open(path)):
        try:
            e, dc = int(r["epoch"]), r["dc_id"]
            dec, gpus = int(r["decoding"]), int(r["num_gpus"])
            duty = float(r["prefill_duty"])
        except (KeyError, ValueError, TypeError):
            continue
        rows[dc].append((e, dec, gpus, duty))
        fleet[e] += dec
    if not fleet:
        return {}
    peak = max(fleet.values())
    live = {e for e, v in fleet.items() if v >= plateau * peak}
    out = {}
    for dc, seq in rows.items():
        seq.sort()
        batches = [d for e, d, _, _ in seq if e in live]
        duties = [d for e, _, _, d in seq if e in live]
        if not batches:
            continue
        out[dc] = {"batch": batches, "gpus": seq[-1][2],
                   "duty": min(0.95, sum(duties) / len(duties)) if duties else 0.0,
                   "duty_cum": seq[-1][3]}
    return out


def read_tpot(path):
    xs = []
    for r in csv.DictReader(open(path)):
        if (r.get("error") or "").strip():
            continue
        try:
            xs.append(float(r["avg_tpot_ms"]))
        except (KeyError, ValueError):
            continue
    return xs


def load_fit(model, tp):
    p = (Path(__file__).resolve().parents[2] / "datacenter/perf_models/analytical"
         / "fit_data" / f"latency_fit_tp{tp}_{model}.json")
    if not p.exists():
        return None
    return json.load(open(p)).get("tpot")


def fit_step(t, bs, kv_per_seq):
    if not t or bs <= 0:
        return None
    K = bs * kv_per_seq
    return (t.get("c0", 0) + t.get("c1", 0) * bs + t.get("c2", 0) * K
            + t.get("c3", 0) * bs * bs + t.get("c4", 0) * (K / bs)
            + t.get("c5", 0) * bs * K)


def cell_config(run):
    for d in (run, run.parent, run.parent.parent):
        p = d / "config.json"
        if p.exists():
            try:
                return json.load(open(p))
            except Exception:
                pass
    return {}


def report(run, plateau, kv_per_seq, model, tp):
    ts = run / "logs" / "fleet_timeseries.csv"
    res = run / "results.csv"
    if not (ts.is_file() and res.is_file()):
        return
    sites = read_series(ts, plateau)
    tpots = read_tpot(res)
    if not sites or not tpots:
        return
    cfg = cell_config(run)
    tp = int(cfg.get("TP", cfg.get("tp", tp)) or tp)
    model = cfg.get("MODEL", cfg.get("model", model)) or model
    t = load_fit(model, tp)

    tp50 = statistics.median(tpots)
    tp90 = pct(tpots, 0.90)
    print(f"\n=== {run}")
    print(f"    n={len(tpots)}  TPOT p50={tp50:.1f} ms  p90={tp90:.1f} ms   "
          f"model={model} TP={tp}  plateau>={plateau:.0%}  kv/seq={kv_per_seq:.0f}")
    print(f"{'site':<20}{'eng':>5}{'share':>7}{'bs/eng p50':>11}{'p90':>7}"
          f"{'max':>7}{'duty':>7}{'son':>7}{'1/(1-d)':>9}{'implied':>9}"
          f"{'fit@p50':>8}{'E[fit]':>8}{'ratio':>7}")
    order = sorted(sites.items(),
                   key=lambda kv: -statistics.mean(kv[1]["batch"]))
    # TPOT is a fleet-wide percentile, so `implied` only means anything on a
    # site that actually holds requests. `share` says which rows to read.
    load = {dc: statistics.mean(v["batch"]) for dc, v in sites.items()}
    tot = sum(load.values()) or 1.0
    for dc, s in order:
        eng = max(1, s["gpus"] // tp)
        b50, b90, bmx = (pct(s["batch"], 0.50) / eng,
                         pct(s["batch"], 0.90) / eng,
                         max(s["batch"]) / eng)
        d = s["duty"]
        mult = 1.0 / max(1e-6, 1.0 - d)
        implied = tp50 * (1.0 - d)
        f = fit_step(t, b50, kv_per_seq)
        # A request does not experience the median batch: it experiences every
        # batch it sits in, and a bigger batch holds more requests for longer.
        # So the fit is averaged over SEAT-STEPS, not over epochs. step() is
        # convex in bs, so once the batch has a tail the two separate and the
        # median one reads low -- which is what `ratio` was over-reading.
        num = den = 0.0
        for s_i in (x / eng for x in s["batch"]):
            f_i = fit_step(t, s_i, kv_per_seq)
            if f_i:
                num += s_i * f_i
                den += s_i
        fe = num / den if den else None
        fs = f"{f:>8.1f}" if f else f"{'-':>8}"
        fes = f"{fe:>8.1f}" if fe else f"{'-':>8}"
        rs = f"{implied / fe:>7.2f}" if fe else f"{'-':>7}"
        print(f"{dc:<20}{eng:>5}{100*load[dc]/tot:>6.1f}%{b50:>11.1f}"
              f"{b90:>7.1f}{bmx:>7.1f}{d:>7.2f}{s['duty_cum']:>7.2f}"
              f"{mult:>9.2f}{implied:>9.1f}{fs}{fes}{rs}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--plateau", type=float, default=0.5)
    ap.add_argument("--kv-per-seq", type=float, default=876.0,
                    help="mean live KV per sequence; mlperf_llama3_8B is "
                         "ISL 841 + OSL 71/2")
    ap.add_argument("--model", default="llama3_8B")
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args(argv)

    root = Path(a.root)
    runs = []
    for dp, _, fs in os.walk(root):
        if "fleet_timeseries.csv" in fs and os.path.basename(dp) == "logs":
            runs.append(Path(dp).parent)
    if not runs:
        sys.exit(f"no fleet_timeseries.csv under {root}")
    for run in sorted(runs):
        report(run, a.plateau, a.kv_per_seq, a.model, a.tp)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
