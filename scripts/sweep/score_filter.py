#!/usr/bin/env python3
"""Was the SLA filter right about the sites it turned down?

    PYTHONPATH=.:proto python3 scripts/sweep/score_filter.py results/e2e_<date>

Needs a run recorded with DECISION_LOG_CANDIDATES=1, which puts the whole
candidate set in each record: every site the scheduler could have used, its
estimate, and the verdict. Without it a decision log shows only the site that
was CHOSEN, which by construction is one the filter accepted -- so it can say
how often acceptance was wrong and nothing at all about rejection.

Rejection has no direct counterfactual: nothing was sent there, so nothing was
measured there for THIS request. What there is instead is other requests, at
that same site, at nearly the same moment. So each verdict is matched against
the outcomes actually observed at that site within a window, and the question
becomes: when the filter said a site would breach, did the requests that went
there anyway breach?

The matching is approximate on purpose and its bias is worth stating. Requests
that reached a site the filter was rejecting arrived from schedulers whose view
of it differed, so they are not a random sample of what this request would have
met. Read the table as evidence about the filter's discrimination, not as a
counterfactual.
"""

import json
import os
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict

WINDOW_MS = 500.0


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    cfg_path, d = None, os.path.abspath(root)
    for _ in range(5):
        if os.path.isfile(os.path.join(d, "config.json")):
            cfg_path = os.path.join(d, "config.json")
            break
        d = os.path.dirname(d)
    if cfg_path is None:
        sys.exit(f"no config.json at or above {root}")
    budget = float(json.load(open(cfg_path)).get("ttft_budget_ms") or 0)

    # per policy: the timeline of what actually happened at each site, and
    # every verdict the filter handed out
    outcomes = defaultdict(lambda: defaultdict(list))   # pol -> dc -> [(t, breach)]
    verdicts = defaultdict(list)                        # pol -> [(t, dc, ok, chosen)]
    seen_cands = 0
    for dp, _, fs in os.walk(root):
        for fn in fs:
            if not (fn.startswith("decisions_") and fn.endswith(".jsonl")):
                continue
            pol = os.path.relpath(dp, root).split(os.sep)[0]
            for line in open(os.path.join(dp, fn)):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = r.get("t_ms")
                if t is None or "m_ttft_ms" not in r:
                    continue
                actual = (float(r["m_queue_ms"]) + float(r["m_ttft_ms"])
                          + float(r.get("fwd_ms", 0.0)))
                outcomes[pol][r["dc_id"]].append((t, actual > budget))
                for c in r.get("cands") or []:
                    seen_cands += 1
                    verdicts[pol].append((t, c[0], bool(c[3]), c[0] == r["dc_id"]))

    if not seen_cands:
        sys.exit("no candidate sets in these logs -- rerun with "
                 "DECISION_LOG_CANDIDATES=1")

    for pol in outcomes:
        for dc in outcomes[pol]:
            outcomes[pol][dc].sort()

    print(f"# {root}   TTFT budget {budget:.0f} ms   window +/-{WINDOW_MS:.0f} ms")
    print()
    hdr = (f"{'policy':<30}{'verdicts':>10}{'said OK':>9}{'->breach':>10}"
           f"{'said NO':>9}{'->breach':>10}{'lift':>7}{'chosen':>8}{'->breach':>10}")
    print(hdr)
    print("-" * len(hdr))
    for pol in sorted(verdicts):
        by = {True: [0, 0], False: [0, 0]}          # ok -> [matched, breached]
        chosen = [0, 0]
        for t, dc, ok, was_chosen in verdicts[pol]:
            tl = outcomes[pol].get(dc)
            if not tl:
                continue
            ts = [x[0] for x in tl]
            lo = bisect_left(ts, t - WINDOW_MS)
            hi = bisect_right(ts, t + WINDOW_MS)
            if hi <= lo:
                continue
            br = sum(1 for _, b in tl[lo:hi] if b) / (hi - lo)
            by[ok][0] += 1
            by[ok][1] += br
            if was_chosen:
                chosen[0] += 1
                chosen[1] += br
        f = lambda a: (a[1] / a[0] * 100) if a[0] else float("nan")   # noqa: E731
        ok_r, no_r = f(by[True]), f(by[False])
        lift = (no_r / ok_r) if ok_r else float("nan")
        print(f"{pol:<30}{len(verdicts[pol]):>10}{by[True][0]:>9}{ok_r:>9.1f}%"
              f"{by[False][0]:>9}{no_r:>9.1f}%{lift:>6.2f}x"
              f"{chosen[0]:>8}{f(chosen):>9.1f}%")
    print()
    print("said OK/NO   the filter's verdict on a site; ->breach is how often")
    print("             requests that actually ran there in the same window")
    print("             missed the budget")
    print("lift         breach rate where it said NO, over where it said OK.")
    print("             1.0 means the filter separated nothing: the sites it")
    print("             turned down were no worse than the ones it kept.")
    print("chosen       the subset it accepted AND picked -- what the run paid")


if __name__ == "__main__":
    main()
