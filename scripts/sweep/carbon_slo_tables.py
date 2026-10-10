#!/usr/bin/env python3
"""One table per rate: where the carbon went, and which budget was missed.

    PYTHONPATH=.:proto python3 scripts/sweep/carbon_slo_tables.py results/grid_v3

Two breakdowns that the other summaries fold away, and both change what a row
means.

CARBON, split three ways rather than two. `work` is what serving cost; `idle`
is what the sites drew with nothing in flight; `host` is the nodes' non-GPU
draw. Only `work` is what a placement decision moves. The split matters most
where it is largest: at 1200 req/s the floor is over 40% of the total, so two
policies can differ by a factor of ten on the part they control and by a
factor of two on the number. Which column belongs in a comparison depends on
whether the idle sites would have been powered anyway -- in a real fleet they
would, which argues for `total`, but then most of the number is not the
policy's doing. Report both and say which one the claim rests on.

Idle and host carbon are computed PER SITE from that site's own carbon
intensity, not by splitting the total by energy share: the empty sites are
frequently the dirty ones, so an energy-weighted split would put the floor at
the wrong price.

SLO, split by which budget was missed. A single "SLO met" number cannot say
whether a policy is failing on time-to-first-token or on inter-token latency,
and those have different causes -- TTFT is queueing plus prefill, TPOT is the
decode batch. Measured 2026-09-07 on grid_v3: above 1200 req/s the binding
constraint is TPOT, not TTFT, which the combined number hides.

uTTFT is the full round trip the client sees -- scheduling + forwarding +
queue + ttft -- because that is where the budget is spent, not the datacentre's
own ttft alone.
"""

import csv
import json
import os
import sys
from pathlib import Path


def carbon_g(energy_j, pue, ci):
    return energy_j * pue * ci / 3.6e6


def read_energy(path):
    """total / work / idle / host carbon in grams, idle and host per site."""
    with open(path) as f:
        d = json.load(f)
    idle = host = 0.0
    for dc in d.get("per_dc", {}).values():
        pue = dc.get("pue", 1.0)
        ci = dc.get("carbon_intensity", 0.0)
        idle += carbon_g(dc.get("idle_energy_j", 0.0), pue, ci)
        host += carbon_g(dc.get("host_energy_j", 0.0), pue, ci)
    return {
        "total": d.get("total_carbon_g", 0.0),
        "work": d.get("total_work_carbon_g", 0.0),
        "idle": idle,
        "host": host,
    }


def read_slo(path, ttft_budget, tpot_budget):
    n = ttft_bad = tpot_bad = both = neither = 0
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
        t_bad = uttft > ttft_budget
        p_bad = tpot > tpot_budget
        ttft_bad += t_bad
        tpot_bad += p_bad
        both += t_bad and p_bad
        neither += not (t_bad or p_bad)
    if not n:
        return None
    return {"n": n, "ttft": 100.0 * ttft_bad / n, "tpot": 100.0 * tpot_bad / n,
            "both": 100.0 * both / n, "ok": 100.0 * neither / n}


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    root = Path(argv[0])
    ttft_budget = float(os.environ.get("TTFT_BUDGET", "3000"))
    tpot_budget = float(os.environ.get("TPOT_BUDGET", "100"))

    cells = {}
    skipped = []
    for pdir in sorted(root.iterdir()):
        if not pdir.is_dir():
            continue
        for rdir in sorted(pdir.iterdir()):
            if not (rdir.is_dir() and rdir.name.startswith("rate_")):
                continue
            tail = rdir.name[len("rate_"):]
            if not tail.isdigit():
                # Quarantined by rename; see summarise_e2e for why that is the
                # convention and why it is announced rather than dropped.
                skipped.append(f"{pdir.name}/{rdir.name}")
                continue
            res, en = rdir / "results.csv", rdir / "energy.json"
            if not (res.exists() and en.exists()):
                continue
            slo = read_slo(res, ttft_budget, tpot_budget)
            if slo is None:
                continue
            cells.setdefault(int(tail), {})[pdir.name] = (read_energy(en), slo)

    if skipped:
        print(f"SKIPPED {len(skipped)} quarantined cell(s): "
              f"{', '.join(sorted(skipped))}\n")

    for rate in sorted(cells):
        block = cells[rate]
        base = block.get("round_robin")
        print(f"══ {rate} req/s     TTFT<{ttft_budget:.0f}ms  "
              f"TPOT<{tpot_budget:.0f}ms")
        print(f"{'policy':<26}{'total g':>9}{'work g':>9}{'idle g':>8}"
              f"{'host g':>8}{'floor%':>7}{'vs RR':>7}   "
              f"{'SLO ok%':>8}{'TTFT bad%':>10}{'TPOT bad%':>10}{'both%':>7}")
        rows = sorted(block.items(), key=lambda kv: kv[1][0]["total"])
        for pol, (c, s) in rows:
            floor = 100.0 * (c["idle"] + c["host"]) / c["total"] if c["total"] else 0
            vs = (100.0 * (base[0]["total"] - c["total"]) / base[0]["total"]
                  if base else float("nan"))
            print(f"{pol:<26}{c['total']:>9.1f}{c['work']:>9.1f}"
                  f"{c['idle']:>8.1f}{c['host']:>8.1f}{floor:>6.0f}%{vs:>6.0f}%"
                  f"   {s['ok']:>7.1f}%{s['ttft']:>9.1f}%{s['tpot']:>9.1f}%"
                  f"{s['both']:>6.1f}%")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
