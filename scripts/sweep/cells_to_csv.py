#!/usr/bin/env python3
"""One CSV row per cell, for plotting off the machine that ran the sweep.

    python3 scripts/sweep/cells_to_csv.py results/grid_v8_cap40 results/... > cells.csv

Every `<root>/<policy>/rate_<r>` directory under each root becomes a row. The
point is to avoid moving `results.csv` at all: a sweep is tens of gigabytes of
per-request rows, and every frontier, cap-axis and TP figure needs only the
per-cell aggregate. Percentiles are included for the same reason -- a latency
CDF can be drawn from these columns without the raw file.

The run's own settings come from `<root>/config.json` when it is there, so a
plot can facet by tp/bs_cap/max_batch_size without the caller tracking which
root was which.

Budgets from TTFT_BUDGET / TPOT_BUDGET, defaulting to 3000 / 100, matching
compare_cells.py.
"""
import csv
import json
import os
import sys
from pathlib import Path

TB = float(os.environ.get("TTFT_BUDGET", 3000))
PB = float(os.environ.get("TPOT_BUDGET", 100))

COLS = ["root", "model", "policy", "rate", "tp", "max_batch_size", "bs_cap",
        "n", "total_g", "work_g", "idle_g", "host_g", "sim_duration_s",
        "slo_ok_pct", "ttft_bad_pct", "tpot_bad_pct", "both_bad_pct",
        "uttft_p50", "uttft_p90", "uttft_p99", "tpot_p50", "tpot_p90", "tpot_p99"]


def pct(a, q):
    return a[min(len(a) - 1, int(q * len(a)))] if a else ""


def energy(p):
    d = json.load(open(p))
    idle = host = 0.0
    for dc in d.get("per_dc", {}).values():
        k = dc.get("pue", 1.0) * dc.get("carbon_intensity", 0.0) / 3.6e6
        idle += dc.get("idle_energy_j", 0.0) * k
        host += dc.get("host_energy_j", 0.0) * k
    return (d.get("total_carbon_g", 0.0), d.get("total_work_carbon_g", 0.0),
            idle, host, d.get("sim_duration_s", ""))


def main(roots):
    w = csv.writer(sys.stdout)
    w.writerow(COLS)
    for root in roots:
        cfg = {}
        c = Path(root) / "config.json"
        if c.exists():
            try:
                cfg = json.load(open(c))
            except ValueError:
                pass
        for cell in sorted(Path(root).glob("*/rate_*")):
            res, en = cell / "results.csv", cell / "energy.json"
            if not res.exists() or not en.exists():
                continue
            # Both violations per request, so `both` is not a product of margins.
            n = tbad = pbad = bbad = 0
            u, v = [], []
            for r in csv.DictReader(open(res)):
                if (r.get("error") or "").strip():
                    continue
                try:
                    x = (float(r["scheduling_ms"]) + float(r["forwarding_ms"])
                         + float(r["queue_ms"]) + float(r["ttft_ms"]))
                    y = float(r["avg_tpot_ms"])
                except (KeyError, ValueError):
                    continue
                n += 1
                u.append(x)
                v.append(y)
                T, Q = x > TB, y > PB
                tbad += T
                pbad += Q
                bbad += T and Q
            if not n:
                continue
            u.sort()
            v.sort()
            tot, work, idle, host, dur = energy(en)
            w.writerow([
                Path(root).name, cfg.get("model_name", ""), cell.parent.name,
                int(cell.name.split("_")[1]),
                cfg.get("tp_size", ""), cfg.get("max_batch_size", ""),
                cfg.get("bs_cap", ""),
                n, f"{tot:.1f}", f"{work:.1f}", f"{idle:.1f}", f"{host:.1f}", dur,
                f"{100.0 * (n - tbad - pbad + bbad) / n:.2f}",
                f"{100.0 * tbad / n:.2f}", f"{100.0 * pbad / n:.2f}",
                f"{100.0 * bbad / n:.2f}",
                f"{pct(u, .5):.0f}", f"{pct(u, .9):.0f}", f"{pct(u, .99):.0f}",
                f"{pct(v, .5):.2f}", f"{pct(v, .9):.2f}", f"{pct(v, .99):.2f}"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) if len(sys.argv) > 1 else (print(__doc__) or 1))
