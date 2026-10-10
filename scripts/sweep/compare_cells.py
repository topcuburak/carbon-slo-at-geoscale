#!/usr/bin/env python3
"""One row per cell, for comparing the same arm across runs.

    python3 scripts/sweep/compare_cells.py \
        results/grid_v3/carbon_aware_a0.0/rate_1200 \
        results/tpotbatch_probe/carbon_aware_a0.0/rate_1200

`carbon_slo_tables.py` reads a whole sweep and groups by rate, which is the
wrong shape when the question is what one policy at one rate did under three
different settings. Each argument is a `<root>/<policy>/rate_<r>` directory and
becomes one row, in the order given, so probes with different policy lists can
be put side by side.

The `vs RR` column needs a same-rate round_robin: it is looked for under every
root passed, and left blank if none of them ran it. RR carbon is within 0.1%
across runs, so borrowing it from a sibling root is safe -- and the column says
which root it came from.

Budgets from TTFT_BUDGET / TPOT_BUDGET, defaulting to 3000 / 100.
"""

import csv
import json
import os
import sys
from pathlib import Path


def carbon(p):
    with open(p) as f:
        d = json.load(f)
    idle = host = 0.0
    for dc in d.get("per_dc", {}).values():
        k = dc.get("pue", 1.0) * dc.get("carbon_intensity", 0.0) / 3.6e6
        idle += dc.get("idle_energy_j", 0.0) * k
        host += dc.get("host_energy_j", 0.0) * k
    return d.get("total_carbon_g", 0.0), d.get("total_work_carbon_g", 0.0), idle + host


def slo(p, tb, pb):
    n = t = q = b = ok = 0
    for r in csv.DictReader(open(p)):
        if (r.get("error") or "").strip():
            continue
        try:
            u = (float(r["scheduling_ms"]) + float(r["forwarding_ms"])
                 + float(r["queue_ms"]) + float(r["ttft_ms"]))
            v = float(r["avg_tpot_ms"])
        except (KeyError, ValueError):
            continue
        n += 1
        T, Q = u > tb, v > pb
        t += T
        q += Q
        b += T and Q
        ok += not (T or Q)
    if not n:
        return None
    return n, 100.0 * ok / n, 100.0 * t / n, 100.0 * q / n, 100.0 * b / n


def label_of(d):
    p = Path(d)
    return f"{p.parent.parent.name}/{p.parent.name}"


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    tb = float(os.environ.get("TTFT_BUDGET", "3000"))
    pb = float(os.environ.get("TPOT_BUDGET", "100"))

    # A same-rate round_robin from any root given, for the vs-RR column.
    rr, rr_from = None, None
    rate = Path(argv[0]).name
    for d in argv:
        cand = Path(d).parent.parent / "round_robin" / rate
        e = cand / "energy.json"
        if e.is_file():
            rr, rr_from = carbon(e)[0], cand.parent.name
            break

    w = max(len(label_of(d)) for d in argv) + 2
    print(f"{'cell':<{w}}{'n':>9}{'total g':>9}{'work g':>9}{'floor%':>7}"
          f"{'vs RR':>7}{'SLO ok%':>9}{'TTFT bad':>9}{'TPOT bad':>9}{'both':>7}")
    for d in argv:
        e, r = Path(d) / "energy.json", Path(d) / "results.csv"
        lab = label_of(d)
        if not (e.is_file() and r.is_file()):
            print(f"{lab:<{w}}   (missing)")
            continue
        tot, work, floor = carbon(e)
        s = slo(r, tb, pb)
        if s is None:
            print(f"{lab:<{w}}   (no rows)")
            continue
        n, ok, t, q, b = s
        vs = f"{100.0 * (rr - tot) / rr:>6.0f}%" if rr else f"{'-':>7}"
        print(f"{lab:<{w}}{n:>9}{tot:>9.1f}{work:>9.1f}"
              f"{100.0 * floor / tot:>6.0f}%{vs}{ok:>8.1f}%{t:>8.1f}%"
              f"{q:>8.1f}%{b:>6.1f}%")
    if rr:
        print(f"\nvs RR against round_robin/{rate} from {rr_from}: {rr:.1f} g")
    else:
        print(f"\nno round_robin/{rate} under any root given; vs RR omitted")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
