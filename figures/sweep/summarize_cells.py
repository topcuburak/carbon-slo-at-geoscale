"""One row per (model, policy, rate) cell: carbon, SLO attainment, latency tails.

Reads a sweep root laid out as <root>/<policy>/rate_<r>/{results.csv,energy.json}
and appends to a CSV that plot_ipdps_frontier.py draws from. Run it where the
data lives (the external disk for the grid_v8 sets, the testbed for new sweeps) and
copy the CSV back.

Definitions match scripts/sweep/carbon_slo_tables.py, which the paper's numbers
come from:
  user TTFT = scheduling_ms + forwarding_ms + queue_ms + ttft_ms
              (forwarding_ms is the full round trip a forward pays)
  SLO met   = status OK and user TTFT <= TTFT budget and avg_tpot <= TPOT budget
              (a failed request counts as a miss, never drops out)
Budgets are fixed per model here, never read from the run's config.json, so an
arm that disables the scheduler's own SLO filter is still measured against the
same budgets.

    python3 figures/sweep/summarize_cells.py --model llama3_8B --tp 1 \
        --root <grid root> --tpot 100 --out data/figures/ipdps_cells.csv
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def _summarize_fast(cell, ttft_budget, tpot_budget):
    """pandas C parser; None if the file needs the tolerant path."""
    try:
        import pandas as pd
        d = pd.read_csv(os.path.join(cell, "results.csv"),
                        usecols=["status", "scheduling_ms", "forwarding_ms",
                                 "queue_ms", "ttft_ms", "avg_tpot_ms"])
    except Exception:                                            # noqa: BLE001
        return None
    n = len(d)
    ok = d[d.status == "OK"].fillna(0)
    u = (ok.scheduling_ms + ok.forwarding_ms + ok.queue_ms + ok.ttft_ms).to_numpy()
    t = ok.avg_tpot_ms.to_numpy()
    met = int(((u <= ttft_budget) & (t <= tpot_budget)).sum())
    return n, len(ok), met, list(u), list(t), list(ok.scheduling_ms.to_numpy())


def summarize(cell, ttft_budget, tpot_budget):
    fast = _summarize_fast(cell, ttft_budget, tpot_budget)
    if fast is not None:
        n, ok, met, uttft, tpot, sched = fast
        return _pack(cell, n, ok, met, uttft, tpot, sched)
    uttft, tpot, sched = [], [], []
    n = ok = met = 0
    with open(os.path.join(cell, "results.csv"), newline="") as f:
        for r in csv.DictReader(f):
            n += 1
            if r["status"] != "OK":
                continue
            ok += 1
            u = (float(r["scheduling_ms"] or 0) + float(r["forwarding_ms"] or 0)
                 + float(r["queue_ms"] or 0) + float(r["ttft_ms"] or 0))
            t = float(r["avg_tpot_ms"] or 0)
            uttft.append(u)
            tpot.append(t)
            sched.append(float(r["scheduling_ms"] or 0))
            met += (u <= ttft_budget and t <= tpot_budget)
    return _pack(cell, n, ok, met, uttft, tpot, sched)


def _pack(cell, n, ok, met, uttft, tpot, sched):
    en = json.load(open(os.path.join(cell, "energy.json")))
    q = lambda a, p: float(np.percentile(a, p)) if a else float("nan")
    return dict(
        n=n, ok_pct=100.0 * ok / max(1, n), slo_pct=100.0 * met / max(1, n),
        carbon_g=float(en["total_carbon_g"]),
        ttft_p50=q(uttft, 50), ttft_p90=q(uttft, 90), ttft_p99=q(uttft, 99),
        tpot_p50=q(tpot, 50), tpot_p90=q(tpot, 90), tpot_p99=q(tpot, 99),
        sched_p50=q(sched, 50), sched_p90=q(sched, 90), sched_p99=q(sched, 99),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ttft", type=float, default=3000.0)
    ap.add_argument("--tpot", type=float, required=True)
    ap.add_argument("--osl", default="oracle", help="label for the OSL source")
    ap.add_argument("--tag", default="", help="free label, e.g. poll1_sync")
    ap.add_argument("--policies", default="", help="space-separated subset")
    ap.add_argument("--rates", default="", help="space-separated subset")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want_p = set(a.policies.split()) if a.policies else None
    want_r = set(int(x) for x in a.rates.split()) if a.rates else None
    rows = []
    for cell in sorted(glob.glob(os.path.join(a.root, "*", "rate_*"))):
        pol = os.path.basename(os.path.dirname(cell))
        rate = int(cell.rsplit("_", 1)[1])
        if (want_p and pol not in want_p) or (want_r and rate not in want_r):
            continue
        if not os.path.exists(os.path.join(cell, "energy.json")):
            continue
        s = summarize(cell, a.ttft, a.tpot)
        rows.append(dict(model=a.model, tp=a.tp, osl=a.osl, tag=a.tag, policy=pol,
                         rate=rate, **s))
        print(f"{a.model} tp{a.tp} {pol:28s} {rate:5d}  carbon {s['carbon_g']:8.1f} g"
              f"  SLO {s['slo_pct']:5.1f}%  ok {s['ok_pct']:6.2f}%")
    if not rows:
        return
    new = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    # Append under the file's own header, so older files without a column
    # (e.g. tag) stay aligned instead of shifting every later field.
    fields = list(rows[0].keys())
    if not new:
        with open(a.out, newline="") as f:
            fields = next(csv.reader(f))
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
