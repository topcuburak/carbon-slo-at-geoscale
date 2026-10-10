#!/usr/bin/env python3
"""Where a request's time to first token goes, and what breaks the budget.

    PYTHONPATH=.:proto python3 scripts/sweep/uttft_split.py <cell> [<cell> ...]

uTTFT is what the client waits for its first token:

    scheduling + forwarding + queue + ttft

and the SLO compares that sum against TTFT_BUDGET. Which of the four is large
decides what to do about it, and they have nothing in common: `scheduling` is
the policy deciding, `forwarding` is a cross-region hop, `queue` is the wait
for a seat at the datacentre, and `ttft` is the prefill itself -- the request's
own pass plus whatever was fused ahead of it.

The percentiles are printed per component, and then, over the requests that
MISSED the budget only, the share for which each component was the largest
term. A median split says what a typical request pays; the violators' split
says what to fix, and the two differ whenever the tail has its own cause.
"""

import csv
import os
import sys
from pathlib import Path

COLS = ("scheduling_ms", "forwarding_ms", "queue_ms", "ttft_ms")


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        if (r.get("error") or "").strip():
            continue
        try:
            v = [float(r[c]) for c in COLS]
        except (KeyError, ValueError):
            continue
        rows.append(v)
    return rows


def q(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    budget = float(os.environ.get("TTFT_BUDGET", "3000"))
    for d in argv:
        p = Path(d) / "results.csv"
        if not p.is_file():
            print(f"{d}: missing"); continue
        rows = load(p)
        if not rows:
            print(f"{d}: no usable rows"); continue
        tot = [sum(v) for v in rows]
        bad = [v for v, t in zip(rows, tot) if t > budget]
        print(f"\n=== {d}")
        print(f"    n={len(rows)}  uTTFT p50={q(tot,.5):.0f}  p90={q(tot,.9):.0f}"
              f"  p99={q(tot,.99):.0f} ms   over {budget:.0f}: "
              f"{100.0*len(bad)/len(rows):.1f}%")
        print(f"{'component':<16}{'p50':>9}{'p90':>9}{'p99':>9}"
              f"{'p50 share':>11}{'largest%':>10}"
              f"{'violator mean':>15}{'of their ms':>13}")
        bad_tot = sum(sum(v) for v in bad) if bad else 0.0
        for i, c in enumerate(COLS):
            xs = [v[i] for v in rows]
            share = 100.0 * q(xs, .5) / q(tot, .5) if q(tot, .5) else 0.0
            if bad:
                # Two different questions about the violators, and they can
                # disagree. `largest%` counts the requests each term dominated,
                # so a term that is huge on a few and absent on the rest scores
                # low. `violator mean` and its share are where their milliseconds
                # actually went, which is what a fix has to remove.
                win = sum(1 for v in bad if v.index(max(v)) == i)
                mean_i = sum(v[i] for v in bad) / len(bad)
                w = (f"{100.0*win/len(bad):>9.1f}%"
                     f"{mean_i:>15.0f}"
                     f"{100.0*sum(v[i] for v in bad)/bad_tot:>12.0f}%")
            else:
                w = f"{'-':>10}{'-':>15}{'-':>13}"
            print(f"{c:<16}{q(xs,.5):>9.0f}{q(xs,.9):>9.0f}{q(xs,.99):>9.0f}"
                  f"{share:>10.0f}%{w}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
