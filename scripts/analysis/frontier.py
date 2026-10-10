#!/usr/bin/env python3
"""Read a rate x policy sweep as a frontier, and say which points are on it.

    python3 scripts/analysis/frontier.py results/frontier_full frontier_full.log

Two things it exists to stop.

A run whose DELIVERED rate fell short of the requested one is marked VOID and
kept out of the front. The generator's ceiling is load-dependent, so a slow
policy is handed less work and its carbon flatters it -- the bias runs straight
through the comparison, and it has voided a full sweep before. See the note in
loadgen/client.py.

And a policy is only on the frontier if it MEETS THE SLO. Carbon at 36% SLA is
not a frontier point, it is a collapse; quoting it as one is the mistake the
table below is arranged to prevent.
"""
import csv, glob, os, re, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/frontier_full"
LOG  = sys.argv[2] if len(sys.argv) > 2 else ROOT.split("/")[-1] + ".log"
TTFT_BUDGET, TPOT_BUDGET = 3000.0, 100.0

def q(a, p):
    a = sorted(a); return a[min(len(a) - 1, int(p * len(a)))] if a else float("nan")

# ---- the sweep log: delivered rate, carbon, work carbon, sim ----------------
energy, cur = {}, None
if os.path.exists(LOG):
    for line in open(LOG):
        m = re.search(r'── Policy=(\S+)\s+α=(\S+)\s+Rate=(\d+)', line)
        if m:
            pol, alpha, rate = m.group(1), m.group(2), int(m.group(3))
            cur = (f"{pol}_a{alpha}" if pol.startswith("carbon_aware") else pol, rate)
            energy[cur] = {}
            continue
        if not cur: continue
        m = re.search(r'delivered (\d+)/s of (\d+)/s', line)
        if m: energy[cur]["deliv"] = int(m.group(1)); energy[cur]["want"] = int(m.group(2))
        m = re.search(r'Carbon: ([\d.]+)g.*sim=([\d.]+)s', line)
        if m: energy[cur]["carbon"] = float(m.group(1)); energy[cur]["sim"] = float(m.group(2))
        m = re.search(r'WORK-ONLY carbon: ([\d.]+)g', line)
        if m: energy[cur]["work"] = float(m.group(1))
        if "GENERATOR BOUND" in line: energy[cur]["bound"] = True

# ---- results.csv: what the requests actually got ----------------------------
rows = []
for d in sorted(glob.glob(f"{ROOT}/*/rate_*")):
    f = os.path.join(d, "results.csv")
    if not os.path.exists(f): continue
    pol, rate = d.split("/")[-2], int(d.split("_")[-1])
    every = list(csv.DictReader(open(f)))
    r = [x for x in every if not (x.get("error") or "").strip()]
    if not r: continue
    err_pct = 100.0 * (len(every) - len(r)) / len(every)
    tt = [float(x["queue_ms"]) + float(x["ttft_ms"]) + float(x.get("forwarding_ms") or 0) for x in r]
    tp = [float(x["avg_tpot_ms"]) for x in r]
    ok = sum(1 for a, b in zip(tt, tp) if a <= TTFT_BUDGET and b <= TPOT_BUDGET)
    e = energy.get((pol, rate), {})
    deliv, want = e.get("deliv", 0), e.get("want", rate)
    rows.append(dict(pol=pol, rate=rate, n=len(r), sla=100.0 * ok / len(r),
                     ttft99=q(tt, .99), tpot50=q(tp, .5),
                     carbon=e.get("carbon", float("nan")),
                     work=e.get("work", float("nan")),
                     sim=e.get("sim", float("nan")),
                     dcs=len({x["dc_id"] for x in r}), deliv=deliv, err=err_pct,
                     # Two separate disqualifications. A short delivered rate
                     # means the GENERATOR fell behind; a high error rate means
                     # the FLEET lost requests. Both void a comparison, and
                     # calling one by the other's name sent a diagnosis down
                     # the wrong path once already.
                     void=bool(e.get("bound")) or (want and deliv < 0.95 * want)
                          or err_pct > 1.0))

print(f"{'policy':<26}{'rate':>5}{'deliv':>7}{'SLA%':>7}{'ttft p99':>10}"
      f"{'tpot p50':>9}{'DC':>4}{'work_g':>9}{'total_g':>9}{'sim':>7}{'err%':>6}  flag")
for r in sorted(rows, key=lambda x: (x["rate"], x["carbon"])):
    d = f"{r['deliv']}" if r['deliv'] else "n/a"
    print(f"{r['pol']:<26}{r['rate']:>5}{d:>7}{r['sla']:>7.1f}"
          f"{r['ttft99']:>10.0f}{r['tpot50']:>9.1f}{r['dcs']:>4}"
          f"{r['work']:>9.1f}{r['carbon']:>9.1f}{r['sim']:>7.0f}{r['err']:>6.1f}"
          f"  {'VOID' if r['void'] else ''}")

# ---- the front: among comparable runs, who is not dominated? ---------------
for thr in (99.0, 95.0):
    print(f"\n=== SLO-uygun frontier (SLA >= {thr:.0f}%, VOID haric) ===")
    any_ = False
    for rate in sorted({r["rate"] for r in rows}):
        pool = [r for r in rows if r["rate"] == rate and not r["void"] and r["sla"] >= thr]
        if not pool: continue
        any_ = True
        best = min(pool, key=lambda x: x["carbon"])
        others = ", ".join(f"{r['pol']} {r['carbon']:.0f}g/{r['sla']:.0f}%"
                           for r in sorted(pool, key=lambda x: x["carbon"])[1:])
        print(f"  rate {rate:>4}: EN AZ KARBON {best['pol']} "
              f"{best['carbon']:.1f}g @ {best['sla']:.1f}%   | {others or '(tek nokta)'}")
    if not any_:
        print("  hicbir kol bu esigi karsilamiyor")
