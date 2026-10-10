#!/usr/bin/env python3
"""Score the scheduler's TTFT estimator against what actually happened.

SIMULATOR_TODO item 16 is what this is for now. `estimate_ttft_ms` prices a
fleet-wide mean, which over-charges whenever an engine is empty -- measured, the
median error moves from -0.08% to +30.08% on that split -- and the open question
is whether that ever crosses the SLA gate on a real sweep. This answers it from
a DECISION_LOG without a second run.

Reads a DECISION_LOG jsonl. Each line is one locally dispatched request: the
target datacentre's snapshot exactly as the policy saw it, followed by that
request's measured outcome there. Both halves in one record, so a candidate
estimator is scored offline with no re-run and no join.

    DECISION_LOG=/tmp/dec.jsonl bash scripts/sweep/sweep_all_policies.sh
    PYTHONPATH=proto:. python scheduler/score_ttft_estimator.py /tmp/dec.jsonl

Ground truth is queue_ms + ttft_ms. ttft_ms alone starts AFTER the admission
wait, so scoring against it hides the largest term in the estimate.

Read the ratio columns, not MAPE. MAPE is asymmetric -- an estimator that
predicts zero scores 100% however wrong it is, while one that overshoots by 3x
scores 200% -- so it rewards under-prediction, which is the failure this
estimator had. Median and mean ratio say where the centre is. And there is a
floor on any of these: run --spread to see how differently two requests can
turn out when the policy saw the same snapshot for both.
"""
import argparse, json, statistics as s
from collections import defaultdict
from types import SimpleNamespace

from datacenter.perf_models.analytical.latency import AnalyticalLatencyModel
from scheduler.policies.common.latency_estimates import estimate_ttft_ms


def as_stats(r):
    return SimpleNamespace(
        num_gpus=r["num_gpus"], active_requests=r["active"],
        pending_requests=r["pending"], prefilling_requests=r["prefilling"],
        decoding_requests=r["decoding"], total_pending_isl=r["pending_isl"],
        # Without this the fused-prefill term is scored with f_isl = 0, which
        # is not the estimator that ships.
        total_prefill_isl=r.get("prefill_isl", 0),
        prefill_duty=max(0.0, r.get("duty", 0.0) or 0.0),
        total_active_isl=r["active_isl"], total_generated_tokens=r["generated"],
        total_remaining_osl=r["remaining_osl"])


def measured(r):
    return r["m_queue_ms"] + r["m_ttft_ms"]


def prefill_only(r, m):
    """The form shipped before 2026-08-30, for comparison."""
    e = max(r["num_gpus"], 1)
    p, pisl = r["pending"], r["pending_isl"]
    drain = (m.ttft_ms(total_prefill_tokens=int(pisl / e), bs=max(1, int(p / e)))
             if p > 0 and pisl > 0 else 0.0)
    return drain + m.ttft_ms(r["isl"], bs=1)


def report(rows, label, m, quiet_forms):
    q = lambda a, p: sorted(a)[min(len(a) - 1, int(p * len(a)))]
    rows = [r for r in rows if measured(r) > 0]
    if not rows:
        return
    print(f"\n{label}  (n={len(rows)})")
    print(f"  {'form':<20}{'p50':>7}{'p90':>7}{'mean':>7}{'MAPE%':>8}")
    for name, f in quiet_forms:
        rat = [f(r) / measured(r) for r in rows]
        ape = [abs(f(r) - measured(r)) / measured(r) * 100 for r in rows]
        print(f"  {name:<20}{q(rat,.5):>7.2f}{q(rat,.9):>7.2f}"
              f"{s.mean(rat):>7.2f}{s.mean(ape):>8.1f}")


def spread(rows):
    """How much outcomes differ among placements the policy cannot distinguish."""
    g = defaultdict(list)
    for r in rows:
        g[(r["dc_id"], r["active"], r["pending"], r["decoding"],
           r["remaining_osl"])].append(measured(r))
    multi = [v for v in g.values() if len(v) >= 4]
    if not multi:
        print("\nNot enough repeated snapshots to measure the spread.")
        return
    sp = sorted((max(v) - min(v)) / max(s.median(v), 1e-9) for v in multi)
    big = max(multi, key=len)
    print(f"\nSnapshot spread over {len(g)} snapshots "
          f"({len(multi)} seen by >= 4 requests):")
    print(f"  (max-min)/median   p50 {sp[len(sp)//2]:.2f}   "
          f"p90 {sp[int(.9*len(sp))]:.2f}")
    print(f"  largest group      n={len(big)}  "
          f"{min(big):.0f} .. {s.median(big):.0f} .. {max(big):.0f} ms")
    print("  No estimator reading this snapshot can score inside that spread.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--model", default="llama3_8B")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-num-seqs", type=int, default=32,
                    help="engine max_num_seqs: gates the admission term and "
                         "splits the regimes below")
    ap.add_argument("--ungated", action="store_true",
                    help="score with the gate off, as if max_num_seqs were "
                         "unknown to the scheduler")
    ap.add_argument("--spread", action="store_true")
    a = ap.parse_args()

    m = AnalyticalLatencyModel(a.model, tp=a.tp)
    rows = [json.loads(l) for l in open(a.log)]
    forms = [("prefill-only (old)", lambda r: prefill_only(r, m)),
             ("shipped", lambda r: estimate_ttft_ms(
                 as_stats(r), r["isl"], m,
                 max_num_seqs=0 if a.ungated else a.max_num_seqs))]

    report(rows, "ALL decisions", m, forms)
    cap = a.max_num_seqs
    report([r for r in rows if r["decoding"] + r["pending"] >= cap * r["num_gpus"]],
           "SATURATED (at the admission cap)", m, forms)
    report([r for r in rows if r["decoding"] + r["pending"] < cap * r["num_gpus"]],
           "BELOW the cap", m, forms)
    if a.spread:
        spread(rows)


if __name__ == "__main__":
    main()
