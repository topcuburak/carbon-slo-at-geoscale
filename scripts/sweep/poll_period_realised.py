#!/usr/bin/env python3
"""What poll period the provisioner ACTUALLY achieved, per run.

    PYTHONPATH=.:proto python3 scripts/sweep/poll_period_realised.py results/poll05_probe

The configured period is a request, not a measurement. `POLL_DEADLINE_S` is 5 s,
so at a 0.5 s setting the poller can fall ten periods behind without the
"deadline passed" warning ever firing -- the deadline is per-site-reply, not per
cycle. So a probe that varies the period must first show the period varied.

One epoch is one fan-out. Its timestamp is the FIRST row written for it, so the
gap between consecutive epoch starts is the realised period. p50 says whether
the setting took; p90 and max say whether it held under load, which is exactly
when a stale view costs a placement.

`sites` is rows/epochs: if it drops below the fleet size the poller is not
merely late, it is dropping sites and the view is partial as well as stale.
"""

import csv
import os
import statistics
import sys
from pathlib import Path


def realised(path):
    first = {}
    rows = 0
    for r in csv.DictReader(open(path)):
        try:
            e, t = int(r["epoch"]), float(r["t_ms"])
        except (KeyError, ValueError, TypeError):
            continue
        rows += 1
        if e not in first or t < first[e]:
            first[e] = t
    if len(first) < 2:
        return None
    ts = [first[e] for e in sorted(first)]
    gaps = [(b - a) / 1000.0 for a, b in zip(ts, ts[1:])]
    gaps.sort()
    q = lambda p: gaps[min(len(gaps) - 1, int(p * len(gaps)))]
    return {"epochs": len(first), "span": (ts[-1] - ts[0]) / 1000.0,
            "p50": statistics.median(gaps), "p90": q(0.90), "max": gaps[-1],
            "sites": rows / len(first)}


def configured(run_dir):
    """(mode, interval) from the sweep's config.json.

    The file sits at the ROOT of a sweep, not in each cell, so walk up -- the
    same thing score_estimator does. In async mode the interval that matters is
    the WAVE interval: POLL_PERIOD_S is not what the provisioner sleeps on
    there, and a DC's own refresh period is wave_count times the wave interval.
    So `sites` reads one wave rather than the fleet, which is how an async run
    identifies itself in this table.
    """
    import json
    d = None
    for up in (run_dir, *run_dir.parents):
        f = up / "config.json"
        if f.is_file():
            try:
                d = json.load(open(f))
            except Exception:
                d = None
            break
    if not d:
        return None, None
    mode = str(d.get("poll_mode", "sync"))
    if mode == "async":
        n = float(d.get("poll_wave_count", 4) or 4)
        iv = d.get("poll_wave_interval_s")
        return f"async/{n:.0f}", (float(iv) if iv is not None else None)
    iv = d.get("poll_period_s", d.get("POLL_PERIOD_S"))
    return mode, (float(iv) if iv is not None else None)


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    root = Path(argv[0])
    runs = []
    for dp, _, fs in os.walk(root):
        if "fleet_timeseries.csv" in fs and os.path.basename(dp) == "logs":
            runs.append(Path(dp).parent)
    if not runs:
        sys.exit(f"no fleet_timeseries.csv under {root}")

    print(f"{'run':<40}{'mode':>9}{'set':>6}{'epochs':>8}{'span s':>8}"
          f"{'p50 s':>8}{'p90 s':>8}{'max s':>8}{'sites':>7}")
    for run in sorted(runs):
        st = realised(run / "logs" / "fleet_timeseries.csv")
        if st is None:
            continue
        mode, cfg = configured(run)
        name = str(run.relative_to(root)) if run != root else run.name
        print(f"{name:<40}{(mode or '-'):>9}{(f'{cfg:.2f}' if cfg else '-'):>6}"
              f"{st['epochs']:>8}{st['span']:>8.0f}{st['p50']:>8.2f}"
              f"{st['p90']:>8.2f}{st['max']:>8.2f}{st['sites']:>7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
