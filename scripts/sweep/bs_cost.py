"""Per site: the batch it sustains, and what that batch costs in prefill delay.

    python3 bs_cost.py <cell> [<cell> ...]

`duty` is the share of engine wall time inside a prefill pass with decode
blocked -- the delay every decoding request pays. Dividing it by the site's own
arrival rate per engine gives the prefill delay ONE request imposes, measured
rather than fitted, and that is what a batch of B costs to assemble and hold.
"""
import csv, collections, json, sys, statistics as st
from pathlib import Path

for D in sys.argv[1:]:
    D = Path(D)
    ts = D / "logs" / "fleet_timeseries.csv"
    rs = D / "results.csv"
    if not (ts.is_file() and rs.is_file()):
        print(f"{D}: missing"); continue

    rows = list(csv.DictReader(open(ts)))
    fl = collections.Counter()
    for r in rows: fl[int(r["epoch"])] += int(r["decoding"])
    peak = max(fl.values()); live = {e for e, v in fl.items() if v >= .5 * peak}
    bs = collections.defaultdict(list); eng = {}
    duties = collections.defaultdict(list)
    for r in rows:
        e = int(r["epoch"])
        if e not in live: continue
        g = max(int(r["num_gpus"]), 1)
        eng[r["dc_id"]] = g
        bs[r["dc_id"]].append(int(r["decoding"]) / g)
        duties[r["dc_id"]].append(float(r["prefill_duty"]))
    # prefill_duty is windowed at the source (180df50), so the plateau value is
    # the mean of its samples. Differencing it, as this did while the counter
    # was cumulative, now multiplies a rate by an elapsed time.
    duty = {dc: min(0.95, sum(v) / len(v)) for dc, v in duties.items() if v}

    n = collections.Counter(); fus = collections.defaultdict(list); t0 = t1 = None
    for r in csv.DictReader(open(rs)):
        if (r.get("error") or "").strip(): continue
        dc = r.get("dc_id")
        if not dc: continue
        n[dc] += 1
        try:
            s = float(r["send_s"]); t0 = s if t0 is None else min(t0, s)
            t1 = s if t1 is None else max(t1, s)
        except (KeyError, ValueError): pass
        try:
            b = int(r["prefill_bs"])
            if b > 0: fus[dc].append(b)
        except (KeyError, ValueError): pass
    span = (t1 - t0) if (t0 is not None and t1 > t0) else 1.0

    q = lambda a, p: sorted(a)[min(len(a) - 1, int(len(a) * p))]
    print(f"\n=== {D}   span {span:.0f}s")
    print(f"{'site':<18}{'bs p50':>8}{'bs p80':>8}{'bs p90':>8}{'duty':>7}"
          f"{'req/s/eng':>11}{'prefill ms/istek':>18}{'fuzyon':>8}")
    for dc in sorted(bs, key=lambda d: -st.mean(bs[d]))[:5]:
        v = bs[dc]; g = eng[dc]
        lam = n[dc] / span / g
        cost = (duty[dc] / lam * 1000.0) if lam > 0 else float("nan")
        f = f"{st.mean(fus[dc]):.1f}" if fus.get(dc) else "-"
        print(f"{dc:<18}{q(v,.5):>8.1f}{q(v,.8):>8.1f}{q(v,.9):>8.1f}"
              f"{duty[dc]:>7.2f}{lam:>11.1f}{cost:>18.1f}{f:>8}")
