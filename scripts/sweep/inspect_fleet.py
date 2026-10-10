#!/usr/bin/env python3
"""What the fleet was doing, epoch by epoch, from `logs/fleet_timeseries.csv`.

    python3 scripts/sweep/inspect_fleet.py results/<run>/<policy>/rate_<r>
    python3 scripts/sweep/inspect_fleet.py results/<run>          # every run under it

Energy is totalled once at the end of a run, over warm-up and drain as well as
the part being measured, and those phases are a different size for every
policy. This is the series that says where their boundaries are -- and, twice
now, that a run was not measuring what it appeared to.

The three things it answers:

  SHAPE    does the fleet fill, hold, and drain? A healthy run rises to a
           plateau and returns to zero. `active` and `decoding` should move
           together, separated only by the prefilling set.

  FREEZE   `active` pinned at a constant while `decoding` sits at zero is a
           counter that stopped being decremented. Seen at 2400 req/s: 4871
           requests frozen in `prefilling` for 220 s across one scheduler
           group, while the run served every request and reported no errors.
           It matters beyond the log line, because `active` is what the
           provisioner ships to every scheduler -- phantom load that
           least_loaded avoids, the SLA filter excludes, and the ILPs price a
           capacity constraint against.

  CAUSE    the KV gate is FCFS and BREAKS rather than skipping, so one prompt
           that does not fit stalls everything behind it. `kv_blocked` climbing
           says the gate is the reason. `kv_tokens` high while nothing decodes
           says the cache accounting leaked instead, which is a different bug.
"""

import collections
import csv
import os
import sys


def read(path):
    ep = collections.defaultdict(lambda: collections.Counter())
    for r in csv.DictReader(open(path)):
        e = int(r["epoch"])
        for k in ("active", "pending", "prefilling", "decoding"):
            ep[e][k] += int(r[k])
        for k, col in (("kv", "kv_tokens"), ("cap", "kv_capacity"),
                       ("blocked", "kv_blocked")):
            if col in r and r[col]:
                ep[e][k] += int(r[col])
    return ep


def report(run_dir):
    path = os.path.join(run_dir, "logs", "fleet_timeseries.csv")
    if not os.path.isfile(path):
        return
    ep = read(path)
    if not ep:
        return
    keys = sorted(ep)
    print(f"\n=== {run_dir}   {len(keys)} epoch")
    has_kv = any(ep[e]["cap"] for e in keys)
    hdr = (f"{'epoch':>6}{'active':>9}{'prefill':>9}{'decoding':>10}{'pending':>9}"
           + (f"{'kv %':>8}{'blocked':>10}" if has_kv else ""))
    print(hdr)
    step = max(1, len(keys) // 25)
    for e in keys[::step]:
        v = ep[e]
        row = (f"{e:>6}{v['active']:>9}{v['prefilling']:>9}"
               f"{v['decoding']:>10}{v['pending']:>9}")
        if has_kv:
            pct = 100.0 * v["kv"] / v["cap"] if v["cap"] else 0.0
            row += f"{pct:>7.1f}%{v['blocked']:>10}"
        print(row)

    last = ep[keys[-1]]
    peak = max(ep[e]["active"] for e in keys)
    print(f"  peak active {peak}, last epoch active {last['active']}")

    # A tail of identical `active` with nothing decoding is the freeze.
    tail, frozen = [ep[e]["active"] for e in keys[-15:]], 0
    for e in reversed(keys):
        if ep[e]["active"] == last["active"] and ep[e]["decoding"] == 0:
            frozen += 1
        else:
            break
    if last["active"] > 0 and last["decoding"] == 0:
        print(f"  !! FROZEN: {last['active']} requests stuck with nothing "
              f"decoding, for the last {frozen} epochs.")
        if has_kv:
            b, kv, cap = last["blocked"], last["kv"], last["cap"]
            pct = 100.0 * kv / cap if cap else 0.0
            print(f"     kv_blocked {b}, cache {pct:.1f}% full "
                  f"({kv} of {cap} tokens)")
            print("     gate blocking" if b else
                  ("     cache accounting leaked -- the gate never fired"
                   if kv else "     neither: look upstream of the engine"))
    elif last["active"] == 0:
        print("  drained clean")
    if len(set(tail)) == 1 and tail[0] > 0:
        print("  !! the last 15 epochs are identical -- nothing moved")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    if os.path.isfile(os.path.join(root, "logs", "fleet_timeseries.csv")):
        report(root)
        return
    found = False
    for dp, _, fs in os.walk(root):
        if "fleet_timeseries.csv" in fs and os.path.basename(dp) == "logs":
            report(os.path.dirname(dp))
            found = True
    if not found:
        sys.exit(f"no fleet_timeseries.csv under {root}")


if __name__ == "__main__":
    main()
