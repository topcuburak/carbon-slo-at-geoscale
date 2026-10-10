#!/usr/bin/env python3
"""What the SLA filter certified, against what actually happened.

    PYTHONPATH=.:proto python3 scripts/sweep/score_estimator.py results/e2e_<date>

Every policy with an SLA constraint admits a site only if the estimator says
the request fits its TTFT budget there. If the estimator is optimistic, the
filter certifies sites that will breach, and every carbon number bought under
that filter was bought at a discount nobody priced. This measures the discount.

The estimator's own output is not logged -- only the snapshot it read and the
outcome that followed -- so it is RECOMPUTED here from the logged snapshot,
through the same `estimate_ttft_ms` the scheduler calls. That is the point:
the number recomputed today is the number a repaired estimator would produce,
so a run recorded before a repair can be rescored after one.

Compared against `m_queue_ms + m_ttft_ms`, not `m_ttft_ms` alone. The estimate
covers admission wait and the step boundary, and those are what the datacentre
reports as queue; scoring against TTFT alone credits the estimator with a
backlog it did predict.
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "proto"))

from datacenter.perf_models.analytical.latency import AnalyticalLatencyModel
from scheduler.policies.common.latency_estimates import estimate_ttft_ms

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/e2e_20260903"

# The decision log stores the datacentre's own field names; the estimator reads
# the proto's. One table, so a rename breaks loudly here instead of silently
# scoring zeros.
FIELDS = {
    "num_gpus": "num_gpus",
    "decoding": "decoding_requests",
    "active": "active_requests",
    "pending": "pending_requests",
    "prefilling": "prefilling_requests",
    "prefill_isl": "total_prefill_isl",
    "pending_isl": "total_pending_isl",
    "active_isl": "total_active_isl",
    "generated": "total_generated_tokens",
    "remaining_osl": "total_remaining_osl",
}
# The KV gate's terms, priced by the site; absent (0) in logs before e8ac4ae.
FLOAT_FIELDS = {"kv_queue_ms": "kv_queue_ms",
                "kv_ms_per_token": "kv_ms_per_token",
                "kv_room_tokens": "kv_room_tokens"}


class Snap:
    """The snapshot the scheduler saw, rebuilt as the estimator expects it."""

    def __init__(self, row):
        for src, dst in FIELDS.items():
            setattr(self, dst, int(row.get(src, 0) or 0))
        for src, dst in FLOAT_FIELDS.items():
            setattr(self, dst, float(row.get(src, 0.0) or 0.0))


def q(v, p):
    return sorted(v)[int(p * (len(v) - 1))] if v else float("nan")


def main():
    # Walk up for config.json, so this works when pointed at one policy's
    # directory and not only at a whole sweep root.
    cfg_path, d = None, os.path.abspath(ROOT)
    for _ in range(5):
        if os.path.isfile(os.path.join(d, "config.json")):
            cfg_path = os.path.join(d, "config.json")
            break
        d = os.path.dirname(d)
    if cfg_path is None:
        sys.exit(f"no config.json at or above {ROOT}")
    cfg = json.load(open(cfg_path))
    budget = float(cfg.get("ttft_budget_ms") or 0)
    max_bs = int(cfg.get("max_batch_size") or 0)
    model = AnalyticalLatencyModel(cfg.get("model_name", "llama3_8B"),
                                   tp=int(cfg.get("tp_size", 1)))

    per_policy = defaultdict(list)
    n_logs = 0
    for dirpath, _, files in os.walk(ROOT):
        for fn in files:
            if not (fn.startswith("decisions_") and fn.endswith(".jsonl")):
                continue
            n_logs += 1
            # <root>/<policy>/rate_<r>/logs/decisions_*.jsonl
            policy = os.path.relpath(dirpath, ROOT).split(os.sep)[0]
            for line in open(os.path.join(dirpath, fn)):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "m_ttft_ms" not in r:
                    continue
                snap, isl = Snap(r), int(r.get("isl", 0))
                fwd = float(r.get("fwd_ms", 0.0))
                # Two comparisons, because they answer different questions and
                # mixing them is wrong in a way that looks like a finding.
                #
                # The FILTER decided on the estimate including the forward leg,
                # and the budget is on what the user waits, so certification is
                # scored on est_full against act_full -- the datacentre's own
                # report plus the same RTT the scheduler charged.
                #
                # The RATIO diagnoses the DATACENTRE model, so it uses neither:
                # a forwarded request's measured queue+ttft has no RTT in it,
                # and comparing it against an estimate that does made
                # round_robin look 4.8x pessimistic when what it actually shows
                # is that 75% of its requests crossed a region. The local
                # policies scoring ~1.00 while the global ones scored 3-5x was
                # the tell: the difference between them is forwarding, not
                # accuracy.
                est_dc = estimate_ttft_ms(snap, isl, model, fwd_ms=0.0,
                                          max_num_seqs=max_bs)
                act_dc = float(r["m_queue_ms"]) + float(r["m_ttft_ms"])
                per_policy[policy].append((est_dc + fwd, act_dc + fwd,
                                           est_dc, act_dc,
                                           r.get("snap_age_ms")))

    if not per_policy:
        sys.exit(f"no decision logs under {ROOT} -- was DECISION_LOG set?")

    print(f"# {ROOT}   budget {budget:.0f} ms   max_num_seqs {max_bs}   "
          f"{n_logs} logs")
    print()
    hdr = (f"{'policy':<30}{'n':>8}{'certified':>11}{'actual':>9}{'gap':>8}"
           f"{'est p50':>9}{'act p50':>9}{'est p90':>9}{'act p90':>9}"
           f"{'r p50':>7}{'r p90':>7}{'r p99':>7}")
    print(hdr)
    print("-" * len(hdr))
    for pol in sorted(per_policy):
        rows = per_policy[pol]
        cert = sum(1 for ef, _, _, _, _ in rows if ef <= budget) / len(rows)
        real = sum(1 for _, af, _, _, _ in rows if af <= budget) / len(rows)
        est = [ed for _, _, ed, _, _ in rows]
        act = [ad for _, _, _, ad, _ in rows]
        ratios = [ed / ad for _, _, ed, ad, _ in rows if ad > 0]
        print(f"{pol:<30}{len(rows):>8}{cert*100:>10.1f}%{real*100:>8.1f}%"
              f"{(cert-real)*100:>7.1f}{q(est,.5):>9.0f}{q(act,.5):>9.0f}"
              f"{q(est,.9):>9.0f}{q(act,.9):>9.0f}"
              f"{q(ratios,.5):>7.2f}{q(ratios,.9):>7.2f}{q(ratios,.99):>7.2f}")
    print()
    print("certified  share the estimator said would fit the budget")
    print("actual     share that did. gap is the over-certification, in points")
    print("certified/actual carry the forward leg on BOTH sides: the budget")
    print("           is on what the user waits, and the filter charged the RTT.")
    print("est/act    the DATACENTRE part only, RTT removed from both sides --")
    print("           this is the model the repair work targets.")
    print("r p50/90/99  percentiles of the PER-REQUEST ratio, which is a")
    print("           different statistic and usually a wider one. A median")
    print("           below 1 with a large r p90 means the estimator is")
    print("           optimistic in the middle and pessimistic on a tail --")
    print("           it both admits sites it should not and rejects sites it")
    print("           should not. Large r values come mostly from near-idle")
    print("           sites, where a small denominator inflates the ratio.")
    # Does the error track how stale the view was? This is the test the log
    # was instrumented for. A ratio that climbs with age says the estimate is
    # priced against a backlog that had already drained; one that is flat says
    # staleness is not the mechanism and the formula is.
    aged = {p: [r for r in rows if r[4] is not None] for p, rows in per_policy.items()}
    aged = {p: r for p, r in aged.items() if len(r) >= 200}
    if aged:
        period = float(cfg.get("poll_period_s") or 2) * 1000.0
        edges = [0, .125, .25, .375, .5, .625, .75, .875, 1.0]
        print()
        print(f"Estimate/actual by snapshot age, in eighths of the {period:.0f} ms "
              f"poll period:")
        hdr2 = f"{'policy':<30}" + "".join(
            f"{f'{int(100*edges[i])}-{int(100*edges[i+1])}%':>10}"
            for i in range(len(edges) - 1))
        print(hdr2)
        print("-" * len(hdr2))
        for pol in sorted(aged):
            cells = []
            for i in range(len(edges) - 1):
                lo, hi = edges[i] * period, edges[i + 1] * period
                sub = [ed / ad for _, _, ed, ad, ag in aged[pol]
                       if ad > 0 and lo <= ag < hi]
                cells.append(f"{q(sub, .5):>10.2f}" if len(sub) >= 25
                             else f"{'-':>10}")
            print(f"{pol:<30}" + "".join(cells))
        print()
        print("A row that RISES left to right means the estimate is charging for")
        print("work that drained before the request arrived -- the older the")
        print("view, the more of it was already gone. A FLAT row rules staleness")
        print("out for that policy and points at the formula instead.")
    print()
    print("Policies with no SLA constraint (round_robin, least_loaded) are")
    print("scored too, as a control: their estimator error is the same, but")
    print("nothing was decided on it.")


if __name__ == "__main__":
    main()
