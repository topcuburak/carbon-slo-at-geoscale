#!/usr/bin/env python3
"""Per cell: the batch an engine carries and how full its KV cache runs.

    python3 scripts/sweep/batch_kv_table.py <root> [<root> ...]

Every `<root>/<policy>/rate_<r>` becomes a row. Samples are (site, epoch) pairs
from `logs/fleet_timeseries.csv`, kept to the middle 60% of epochs so warm-up
and drain do not dilute the plateau.

Batch columns are per engine: the site's count over its engines. So `max` is
the busiest SITE's average engine, not one engine's peak -- the series carries
no per-engine counts. `active` is every seated request (prefilling plus
decoding), `pending` the queue behind the seat cap, `kv%` the site's cache
occupancy, and `kv_blk` the largest `kv_blocked` the window saw.
"""
import csv
import statistics as st
import sys
from pathlib import Path


def pct(a, q):
    a = sorted(a)
    return a[min(len(a) - 1, int(q * len(a)))]


def cell(path):
    rows = list(csv.DictReader(open(path)))
    ep = sorted({int(r["epoch"]) for r in rows})
    lo, hi = ep[int(.2 * len(ep))], ep[int(.8 * len(ep))]
    act, dec, pend, kv, blk = [], [], [], [], 0
    for r in rows:
        g = int(r["num_gpus"] or 0)
        if g < 1 or not lo <= int(r["epoch"]) <= hi:
            continue
        act.append(int(r["active"]) / g)
        dec.append(int(r["decoding"]) / g)
        pend.append(int(r["pending"]) / g)
        cap = int(r.get("kv_capacity") or 0)
        if cap:
            kv.append(100.0 * int(r["kv_tokens"]) / cap)
        blk = max(blk, int(r.get("kv_blocked") or 0))
    return act, dec, pend, kv, blk


def order(ts):
    rate = ts.parts[-3][len("rate_"):]
    return ts.parts[-4], int(rate) if rate.isdigit() else float("inf"), rate


print(f"{'policy':<24}{'rate':>6} |{'active p10':>11}{'p50':>7}{'p90':>7}"
      f"{'ort':>7}{'max':>7} |{'decode p50':>11}{'max':>7} |"
      f"{'pending p90':>12}{'max':>7} |{'kv% p10':>8}{'p50':>6}{'p90':>6}"
      f"{'max':>6} |{'kv_blk':>7}")
for root in sys.argv[1:]:
    for ts in sorted(Path(root).glob("*/rate_*/logs/fleet_timeseries.csv"),
                     key=order):
        pol, rate = ts.parts[-4], ts.parts[-3][len("rate_"):]
        act, dec, pend, kv, blk = cell(ts)
        if not act:
            print(f"{pol:<24}{rate:>6} | empty")
            continue
        kvs = (f"{pct(kv, .1):>8.1f}{pct(kv, .5):>6.1f}{pct(kv, .9):>6.1f}"
               f"{max(kv):>6.1f}" if kv else f"{'-':>8}{'-':>6}{'-':>6}{'-':>6}")
        print(f"{pol:<24}{rate:>6} |{pct(act, .1):>11.1f}{pct(act, .5):>7.1f}"
              f"{pct(act, .9):>7.1f}{st.mean(act):>7.1f}{max(act):>7.1f} |"
              f"{pct(dec, .5):>11.1f}{max(dec):>7.1f} |"
              f"{pct(pend, .9):>12.1f}{max(pend):>7.1f} |{kvs} |{blk:>7}")
