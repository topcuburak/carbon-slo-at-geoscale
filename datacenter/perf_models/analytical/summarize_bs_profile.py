#!/usr/bin/env python3
"""Summarise where the simulator actually spends its GPU-seconds, by batch size.

The energy model is fitted inside a measured envelope (BS <= 512, KV <= 97280
tokens on a 40 GB A100 at TP=1) and the linear form extrapolates past it with
nothing behind it. Extending the campaign into the untested high-BS corner is
only worth the GPU time if the simulator goes there. This reads the per-DC
profiles written when BS_PROFILE_DIR is set and says whether it does.

Usage:
    BS_PROFILE_DIR=results/bs_prof bash scripts/sweep/sweep_all_policies.sh
    python datacenter/perf_models/analytical/summarize_bs_profile.py results/bs_prof
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# The fitted envelope's upper batch size. The artifacts are named per model,
# energy_fit_tp1_<model>.json; there has never been a bare energy_fit_tp1.json.
ENVELOPE_BS_MAX = 512


def main(d):
    files = sorted(Path(d).glob("bs_profile_*.json"))
    if not files:
        sys.exit(f"no bs_profile_*.json under {d}")
    agg = defaultdict(float)
    for f in files:
        p = json.load(open(f))
        for bs, secs in p["seconds_at_bs"].items():
            agg[int(bs)] += secs
    total = sum(agg.values()) or 1.0
    print(f"{len(files)} datacenters, {total:.0f} GPU-seconds\n")
    print(f"{'BS bucket':>14} {'GPU-s':>12} {'share':>8}  cumulative")
    buckets = [(0, 0), (1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 64),
               (65, 128), (129, 256), (257, 512), (513, 10 ** 9)]
    acc = 0.0
    for lo, hi in buckets:
        s = sum(v for bs, v in agg.items() if lo <= bs <= hi)
        if s <= 0:
            continue
        acc += s
        label = "idle (bs=0)" if hi == 0 else (f">{ENVELOPE_BS_MAX}" if lo > ENVELOPE_BS_MAX
                                              else f"{lo}-{hi}")
        print(f"{label:>14} {s:12.1f} {100*s/total:7.1f}% {100*acc/total:10.1f}%")

    over = sum(v for bs, v in agg.items() if bs > ENVELOPE_BS_MAX)
    busy = sum(v for bs, v in agg.items() if bs > 0) or 1.0
    top = max((bs for bs in agg if agg[bs] > 0), default=0)
    print(f"\nmax batch size reached: {top}")
    print(f"time outside the fitted envelope (BS > {ENVELOPE_BS_MAX}): "
          f"{100*over/busy:.2f}% of busy GPU-seconds")
    if over / busy < 0.01:
        print("=> the fitted envelope covers the operating region; extending the\n"
              "   power campaign to higher BS would measure a corner the\n"
              "   simulator never visits.")
    else:
        print("=> the simulator does run outside the envelope; that corner needs\n"
              "   measuring, or the model needs clamping and a stated caveat.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/bs_prof")
