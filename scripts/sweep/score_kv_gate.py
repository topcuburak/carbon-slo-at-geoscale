#!/usr/bin/env python3
"""The KV gate forecast against the queue the engine actually charged.

    PYTHONPATH=.:proto python3 scripts/sweep/score_kv_gate.py <root>/<policy>/rate_<r>

score_estimator.py scores the whole TTFT estimate. This separates the part the
site's KV forecast (DcStats.kv_queue_ms, kv_room_tokens, kv_ms_per_token) adds
from the compute terms that were there before it, by rescoring every logged
decision twice -- as logged, and with the three KV fields zeroed.

Rows:
  all          every decision
  gate on      the forecast said the KV gate would hold this prompt
  gate off, q  the forecast said no, but the measured queue ran over 1 s

Columns: n and share; measured queue_ms p50/p90 (arrival to prefill
admission, which is what the gate term forecasts together with the compute
queue); the forecast gate wait p50/p90; and est/act p50 for the TTFT estimate
without and with the KV term, act = m_queue_ms + m_ttft_ms.
"""

import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "proto"))

from datacenter.perf_models.analytical.latency import AnalyticalLatencyModel
from scheduler.policies.common.latency_estimates import estimate_ttft_ms

INT_FIELDS = {"num_gpus": "num_gpus", "decoding": "decoding_requests",
              "active": "active_requests", "pending": "pending_requests",
              "prefilling": "prefilling_requests",
              "prefill_isl": "total_prefill_isl",
              "pending_isl": "total_pending_isl",
              "active_isl": "total_active_isl",
              "generated": "total_generated_tokens",
              "remaining_osl": "total_remaining_osl"}
KV_FIELDS = ("kv_queue_ms", "kv_room_tokens", "kv_ms_per_token")


class Snap:
    def __init__(self, row, kv=True):
        for src, dst in INT_FIELDS.items():
            setattr(self, dst, int(row.get(src, 0) or 0))
        for f in KV_FIELDS:
            setattr(self, f, float(row.get(f, 0.0) or 0.0) if kv else 0.0)


def q(v, p):
    return sorted(v)[int(p * (len(v) - 1))] if v else float("nan")


def main(root):
    cfg, d = None, os.path.abspath(root)
    for _ in range(5):
        if os.path.isfile(os.path.join(d, "config.json")):
            cfg = json.load(open(os.path.join(d, "config.json")))
            break
        d = os.path.dirname(d)
    if cfg is None:
        sys.exit(f"no config.json at or above {root}")
    model = AnalyticalLatencyModel(cfg["model_name"], tp=int(cfg["tp_size"]))
    max_bs = int(cfg.get("max_batch_size") or 0)

    rows = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not (fn.startswith("decisions_") and fn.endswith(".jsonl")):
                continue
            for line in open(os.path.join(dirpath, fn)):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "m_ttft_ms" not in r or "kv_queue_ms" not in r:
                    continue
                isl = int(r.get("isl", 0))
                gate = (float(r["kv_queue_ms"])
                        + max(0.0, isl - float(r["kv_room_tokens"]))
                        * float(r["kv_ms_per_token"]))
                act = float(r["m_queue_ms"]) + float(r["m_ttft_ms"])
                old = estimate_ttft_ms(Snap(r, kv=False), isl, model,
                                       max_num_seqs=max_bs)
                new = estimate_ttft_ms(Snap(r), isl, model, max_num_seqs=max_bs)
                rows.append((gate, float(r["m_queue_ms"]), act, old, new))
    if not rows:
        sys.exit(f"no decisions with KV fields under {root}")

    n = len(rows)
    groups = (("all", rows),
              ("gate on", [x for x in rows if x[0] > 0]),
              ("gate off, q>1s", [x for x in rows if x[0] <= 0 and x[1] > 1000]))
    print(f"# {root}   {n} decisions")
    hdr = (f"{'':<16}{'n':>7}{'share':>7}{'q p50':>8}{'q p90':>8}"
           f"{'gate p50':>9}{'gate p90':>9}{'old r50':>8}{'new r50':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, g in groups:
        if not g:
            print(f"{name:<16}{0:>7}")
            continue
        print(f"{name:<16}{len(g):>7}{100 * len(g) / n:>6.1f}%"
              f"{q([x[1] for x in g], .5):>8.0f}{q([x[1] for x in g], .9):>8.0f}"
              f"{q([x[0] for x in g], .5):>9.0f}{q([x[0] for x in g], .9):>9.0f}"
              f"{q([x[3] / x[2] for x in g if x[2] > 0], .5):>8.2f}"
              f"{q([x[4] / x[2] for x in g if x[2] > 0], .5):>8.2f}")

    # Calibration by forecast size. The site's number is a mean over engines
    # an arrival is round-robined across, so a single request draws ONE
    # engine's wait: medians can disagree while the means agree. Mean against
    # mean says whether the forecast is right on average; the two shares say
    # how bimodal what it averages is.
    print()
    hdr = (f"{'gate ms':<16}{'n':>7}{'gate mean':>10}{'q mean':>9}"
           f"{'q<100ms':>9}{'q>2s':>7}")
    print(hdr)
    print("-" * len(hdr))
    for lo, hi in ((0, 1), (1, 1000), (1000, 3000), (3000, 10000),
                   (10000, float("inf"))):
        g = [x for x in rows if lo <= x[0] < hi] if lo else \
            [x for x in rows if x[0] <= 0]
        if not g:
            continue
        name = "0" if not lo else f"{lo}-{hi:.0f}" if hi < float("inf") else f">{lo}"
        print(f"{name:<16}{len(g):>7}{sum(x[0] for x in g) / len(g):>10.0f}"
              f"{sum(x[1] for x in g) / len(g):>9.0f}"
              f"{100 * sum(1 for x in g if x[1] < 100) / len(g):>8.1f}%"
              f"{100 * sum(1 for x in g if x[1] > 2000) / len(g):>6.1f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
