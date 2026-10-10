#!/usr/bin/env python3
"""Why a TTFT violation happened: nobody was feasible, or the filter was wrong.

    python3 scripts/sweep/fallthrough_split.py <decision_log.jsonl> [limit]

`sla_ok` is what the scheduler believed about the site it CHOSE. False means no
site passed the filter and the policy fell through to the carbon rung -- the
violation was accepted, not mispredicted. True means the filter certified this
placement, so a breach is the estimator's error.

The record carries the outcome beside the belief (`m_queue_ms`, `m_ttft_ms`), so
this needs no join. Pass a limit to sample: the split converges in a few tens of
thousands of decisions and a full log is gigabytes.
"""
import json, sys
from statistics import median

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BUDGET = float(__import__("os").environ.get("TTFT_BUDGET", 3000))

g = {True: [], False: [], None: []}
for i, line in enumerate(open(path)):
    if limit and i >= limit:
        break
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if "m_ttft_ms" not in r:
        continue
    u = (r.get("fwd_ms") or 0) + (r.get("m_queue_ms") or 0) + (r.get("m_ttft_ms") or 0)
    g[r.get("sla_ok")].append((u, r.get("est_ttft_ms"), r.get("snap_age_ms")))

n = sum(len(v) for v in g.values())
print(f"{n} decisions from {path}")
print(f"{'sla_ok':>8}{'share':>9}{'bad%':>8}{'uTTFT p50':>11}{'p90':>8}"
      f"{'est p50':>10}{'est/act':>9}{'age p50':>9}")
for k in (True, False, None):
    v = g[k]
    if not v:
        continue
    u = sorted(x[0] for x in v)
    bad = sum(1 for x in u if x > BUDGET) / len(u)
    est = [x[1] for x in v if x[1] is not None]
    age = [x[2] for x in v if x[2] is not None]
    p = lambda a, q: a[min(len(a) - 1, int(q * len(a)))]
    print(f"{str(k):>8}{len(v)/n:>8.1%}{bad:>8.1%}{p(u,.5):>11.0f}{p(u,.9):>8.0f}"
          f"{(median(est) if est else 0):>10.0f}"
          f"{(median(est)/p(u,.5) if est and p(u,.5) else 0):>9.2f}"
          f"{(median(age) if age else 0):>9.0f}")

viol_total = sum(1 for v in g.values() for x in v if x[0] > BUDGET)
if viol_total:
    ft = sum(1 for x in g[False] if x[0] > BUDGET)
    print(f"\nof {viol_total} violations: {ft/viol_total:.1%} were fall-throughs "
          f"(no site passed), {1-ft/viol_total:.1%} were certified and breached")
