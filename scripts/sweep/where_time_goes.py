#!/usr/bin/env python3
"""Split a request's wall time into the parts that are measured and the part
that is not.

The per-request columns do not add up to wall_ms, and the difference is the
interesting one:

  pool_wait_ms   the loadgen's own queueing, before the RPC is issued.
                 Large means the run went closed-loop and the offered rate is
                 not the configured rate -- the numbers below are then about a
                 slower workload, not about the fleet.

  admit_ms       wall_ms - latency_ms - pool_wait_ms. The request is on the
                 wire but the servicer has not started it: waiting for a gRPC
                 worker, plus transport. Nothing else instruments this.
                 scheduling_ms cannot see it -- t_sched_start is stamped
                 INSIDE the handler, after a worker picked the request up.

  dc_queue_ms    latency_ms - service_ms - forwarding_ms. Everything in a
                 request's life that no clock inside the DATACENTRE covers.
                 NOT purely datacentre-side, despite the name: `latency_ms` is
                 the SCHEDULER's measure, so this also carries the scheduler's
                 own loop lag on the way out and back. Read it against `admit`
                 -- when both are large the scheduler is the congested one.
                 Measured 2026-09-06 on the async datacentre, which has no
                 worker pool left to queue in, it still read 5.0 s while admit
                 read 6.0 s.
                 THE DATACENTRES ARE STILL THREAD-PER-REQUEST: the pool is
                 GRPC_MAX_WORKERS_DC (8000 by default), so a site can hold
                 only that many requests in the system at once and everything
                 beyond waits in gRPC, before any clock the DC keeps. service_ms
                 starts when the handler does, and queue_ms (the engine
                 semaphore wait) is INSIDE service_ms, so neither can see it.

                 It stays near zero while load is spread -- round_robin at
                 4800 req/s measured p50 2.4 ms -- and explodes the moment a
                 policy concentrates: least_loaded 15.1 s at the same rate,
                 carbon_aware alpha=0.0 240.5 s at 4000. In that last run it
                 was 94% of the request's life, with service_ms reading a
                 comparatively innocent 13.5 s. Making the datacentre async cut
                 that 240.5 s to 3.8 s and raised the policy's delivered rate
                 from 863 to 3107 req/s; what remains is the scheduler's share. Any concentrating policy
                 (low-alpha carbon_aware, the ILPs) meets this before it meets
                 the engines, so read this column BEFORE concluding anything
                 about a policy from service_ms or from the error count.

  latency_ms     what the scheduler thinks the request cost, end to end.

forwarding_ms is excluded from admit on purpose: it is 2*rtt_ms, a modelled
charge, and is constant regardless of load. It IS subtracted in dc_queue_ms,
where it would otherwise be counted as datacentre-side waiting.

Usage:
  python3 scripts/sweep/where_time_goes.py RESULTS_DIR [RESULTS_DIR ...]
  python3 scripts/sweep/where_time_goes.py results/rr_capacity/round_robin/rate_*/
"""
import csv
import os
import sys
from collections import Counter


def q(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * (len(s) - 1)))]


def row_stats(path):
    ok, err = [], Counter()
    arrivals = []
    with open(path) as f:
        for r in csv.DictReader(f):
            a = r.get("arrival_s")
            if a:
                arrivals.append(float(a))
            e = (r.get("error") or "").strip()
            if e:
                err[e.split(":")[0][:40]] += 1
            else:
                ok.append(r)
    return ok, err, arrivals


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    for d in argv:
        path = d if d.endswith(".csv") else os.path.join(d, "results.csv")
        if not os.path.exists(path):
            print(f"-- missing: {path}")
            continue
        ok, err, arrivals = row_stats(path)
        n_err = sum(err.values())
        label = os.path.basename(os.path.dirname(path)) or path
        print(f"\n== {label}   ok={len(ok)}  err={n_err} "
              f"({100.0 * n_err / max(1, len(ok) + n_err):.1f}%)")
        for e, c in err.most_common(3):
            print(f"     err {c:>7}  {e}")
        if not ok:
            continue

        def col(name):
            return [float(r[name]) for r in ok if r.get(name) not in (None, "")]

        pool = col("pool_wait_ms")
        wall = col("wall_ms")
        lat = col("latency_ms")
        svc = col("service_ms")
        # admit: on the wire, not yet in the handler.
        admit = [max(0.0, float(r["wall_ms"]) - float(r["latency_ms"]))
                 for r in ok if r.get("wall_ms") and r.get("latency_ms")]
        # dc_queue: at the datacentre, handler not yet started. See the module
        # docstring -- this is the third queue in this system to have had no
        # instrument, after the scheduler's admit and the stream mux's outbox.
        dc_q = [max(0.0, float(r["latency_ms"]) - float(r["service_ms"])
                    - float(r.get("forwarding_ms") or 0.0))
                for r in ok
                if r.get("latency_ms") and r.get("service_ms")]

        print(f"   {'':<14}{'p50':>11}{'p90':>11}{'p99':>12}{'max':>12}")
        for name, v in (("pool_wait", pool), ("admit", admit),
                        ("dc_queue", dc_q), ("service", svc),
                        ("latency", lat), ("wall", wall)):
            if not v:
                continue
            print(f"   {name:<14}{q(v, .5):>11.1f}{q(v, .9):>11.1f}"
                  f"{q(v, .99):>12.1f}{max(v):>12.1f}")

        # Offered vs delivered.
        #
        # Offered counts EVERY row, failed ones included, over the whole
        # arrival window. Counting only the successful rows over their own
        # span -- which this did at first -- reads the generator as slowing
        # down when what actually happened is that its later requests failed:
        # at 30% errors it reported 4198 req/s offered where the generator had
        # issued 4760, and that gap was enough to blame the wrong component.
        recv = col("recv_s")
        if arrivals and recv:
            span_off = max(arrivals) - min(arrivals)
            span_del = max(recv) - min(recv)
            if span_off > 0:
                print(f"   offered {len(arrivals) / span_off:8.1f} req/s over "
                      f"{span_off:.0f}s   delivered {len(ok) / span_del:8.1f} "
                      f"req/s over {span_del:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
