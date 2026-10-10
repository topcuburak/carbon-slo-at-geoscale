#!/usr/bin/env python3
"""What real per-site PUE does to the carbon saving, split in two.

    PYTHONPATH=.:proto TTFT_BUDGET=3000 TPOT_BUDGET=200 \\
        python3 scripts/sweep/pue_decomposition.py <uniform_root> <real_root> [pue_table]

The two roots are the same sweep on the same topology, one with every site at
PUE 1.1 and one with `pue_table` (default common/pue_by_dc_real.json). Three
savings against round_robin, per policy and rate:

  A  uniform     the uniform run as reported: decisions and accounting at 1.1.
  B  blind       the uniform run's energy re-priced at the real PUE. The
                 scheduler did not know PUE, the bill does. This is what the
                 published results would read under honest accounting.
  C  aware       the real run: the scheduler ranks sites on CI x PUE and the
                 bill uses the same values.

B - A is the accounting effect: PUE differs by site and does not cancel.
C - B is the decision effect: what making the scheduler PUE-aware buys.
Energy per site is in each cell's energy.json, so B needs no extra run.

round_robin is carbon-blind, so its B and C totals should agree to within run
noise; the rrB/rrC column checks that.
"""

import csv
import json
import os
import sys
from pathlib import Path


def slo_ok(path, ttft_budget, tpot_budget):
    n = ok = 0
    for r in csv.DictReader(open(path)):
        if (r.get("error") or "").strip():
            continue
        try:
            uttft = (float(r["scheduling_ms"]) + float(r["forwarding_ms"])
                     + float(r["queue_ms"]) + float(r["ttft_ms"]))
            tpot = float(r["avg_tpot_ms"])
        except (KeyError, ValueError):
            continue
        n += 1
        ok += uttft <= ttft_budget and tpot <= tpot_budget
    return 100.0 * ok / n if n else float("nan")


def cells(root):
    out = {}
    for rdir in Path(root).glob("*/rate_*"):
        tail = rdir.name[len("rate_"):]
        en, res = rdir / "energy.json", rdir / "results.csv"
        if tail.isdigit() and en.exists() and res.exists():
            out[(rdir.parent.name, int(tail))] = (json.load(open(en)), res)
    return out


def repriced(energy, pue):
    g = 0.0
    for dc, d in energy.get("per_dc", {}).items():
        if "energy_j" not in d:
            continue
        g += d["energy_j"] * pue.get(dc, d.get("pue", 1.1)) * d.get("carbon_intensity", 0.0) / 3.6e6
    return g


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    uniform, real = cells(argv[0]), cells(argv[1])
    table = argv[2] if len(argv) > 2 else "common/pue_by_dc_real.json"
    pue = {k: float(v) for k, v in json.load(open(table)).items() if not k.startswith("_")}
    ttft = float(os.environ.get("TTFT_BUDGET", "3000"))
    tpot = float(os.environ.get("TPOT_BUDGET", "100"))

    for rate in sorted({r for _, r in uniform}):
        rr_u, rr_r = uniform.get(("round_robin", rate)), real.get(("round_robin", rate))
        if not (rr_u and rr_r):
            continue
        rr_a = rr_u[0]["total_carbon_g"]
        rr_b = repriced(rr_u[0], pue)
        rr_c = rr_r[0]["total_carbon_g"]
        print(f"== {rate} req/s   RR g: A {rr_a:.0f}  B {rr_b:.0f}  C {rr_c:.0f}  (rrB/rrC {rr_b / rr_c:.3f})")
        print(f"{'policy':<26}{'A uniform':>10}{'B blind':>9}{'C aware':>9}{'C-B':>7}{'B-A':>7}{'SLO unif':>10}{'SLO real':>10}")
        rows = []
        for (pol, r), (en_u, res_u) in uniform.items():
            if r != rate or (pol, r) not in real:
                continue
            en_r, res_r = real[(pol, r)]
            a = 100.0 * (1 - en_u["total_carbon_g"] / rr_a)
            b = 100.0 * (1 - repriced(en_u, pue) / rr_b)
            c = 100.0 * (1 - en_r["total_carbon_g"] / rr_c)
            rows.append((pol, a, b, c, slo_ok(res_u, ttft, tpot), slo_ok(res_r, ttft, tpot)))
        for pol, a, b, c, su, sr in sorted(rows, key=lambda x: -x[3]):
            print(f"{pol:<26}{a:>9.1f}%{b:>8.1f}%{c:>8.1f}%{c - b:>+7.1f}{b - a:>+7.1f}{su:>9.1f}%{sr:>9.1f}%")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
