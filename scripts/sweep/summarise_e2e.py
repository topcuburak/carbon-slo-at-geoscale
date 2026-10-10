#!/usr/bin/env python3
"""One table over an end-to-end run: what each policy cost and what it served.

    PYTHONPATH=.:proto python3 scripts/sweep/summarise_e2e.py results/e2e_20260902

Reads every <policy>/rate_<rate>/ directory under the root and reports the four
things a policy is judged on, side by side:

  carbon     from energy.json, which is written after FinalizeEnergy. Reported
             two ways -- total, which charges the fleet for being powered, and
             work-only, which charges just what serving cost. Which one belongs
             in a comparison depends on whether the sites would have been up
             anyway, and that is a question about the deployment.
  tail       user-visible TTFT: scheduling + the FULL round trip + queue +
             TTFT. It used to charge half, on the reasoning that the request
             pays one leg before the first token and the other on the way
             back. That is the right decomposition and the wrong conclusion:
             TTFT is measured at the CLIENT, as MLPerf measures it, so the
             client waits for both legs -- the request out and the first token
             back. Charging half understated attainment by 0.2-1.0 points,
             measured, so no conclusion turned on it; but the TTFT budget is
             derived as MLPerf Server's 2000 ms plus this topology's p99
             wide-area overhead, and a budget that includes the full round trip
             has to be scored against a tail that does too.
  SLO        share of requests inside the configured budgets. TPOT is measured
             per request, so it carries prefill blocking decode -- the
             scheduler's own estimate does not, and the two are not the same
             number.
  spread     how many datacentres were used at all, and how much of the load
             the busiest one took. A policy that parks everything on the
             greenest site scores well on carbon by saturating one engine
             pool, and this is the column that shows it.
"""

import csv
import json
import os
import sys
from collections import Counter

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/e2e_20260902"
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "proto"))


def q(vals, p):
    if not vals:
        return 0.0
    v = sorted(vals)
    return v[int(p * (len(v) - 1))]


def load_run(d):
    f = os.path.join(d, "results.csv")
    if not os.path.isfile(f):
        return None
    rows = list(csv.DictReader(open(f)))
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        return None
    e = {}
    ef = os.path.join(d, "energy.json")
    if os.path.isfile(ef):
        try:
            e = json.load(open(ef))
        except json.JSONDecodeError:
            e = {}
    g = lambda k: [float(r[k]) for r in ok]                        # noqa: E731
    user = [float(r["scheduling_ms"]) + float(r["forwarding_ms"])
            + float(r["queue_ms"]) + float(r["ttft_ms"]) for r in ok]
    dcs = Counter(r["dc_id"] for r in ok)
    fwd = sum(1 for r in ok if float(r["forwarding_ms"]) > 0)
    return {
        "n": len(rows), "err": len(rows) - len(ok),
        "carbon_g": e.get("total_carbon_g", 0.0),
        "work_carbon_g": e.get("total_work_carbon_g", 0.0),
        "energy_kwh": e.get("total_energy_kwh", 0.0),
        "floor": e.get("floor_fraction", 0.0),
        "ttft_p50": q(g("ttft_ms"), .5), "ttft_p99": q(g("ttft_ms"), .99),
        "tpot_p50": q(g("avg_tpot_ms"), .5), "tpot_p99": q(g("avg_tpot_ms"), .99),
        "user_p50": q(user, .5), "user_p90": q(user, .9), "user_p99": q(user, .99),
        "tpot_p90": q(g("avg_tpot_ms"), .9),
        "sched_p90": q(g("scheduling_ms"), .9),
        "dcs_used": len(dcs),
        "top_share": max(dcs.values()) / len(ok),
        "fwd_share": fwd / len(ok),
        # The emission factor of the work that was actually served: total
        # carbon over total energy, from energy.json. NOT a property of the
        # policy -- carbon intensity belongs to a grid zone -- but of where
        # the policy put the work. Energy-weighted on purpose: a long request
        # burns far more than a short one, and a request-weighted mean of the
        # chosen sites' intensities flatters a policy that sends only its
        # small requests somewhere green. It carries PUE, so divide by PUE to
        # compare against a zone's published intensity.
        "gco2_per_kwh": (e.get("total_carbon_g", 0.0)
                         / e["total_energy_kwh"]) if e.get("total_energy_kwh") else 0.0,
    }


def main():
    cfg = {}
    cf = os.path.join(ROOT, "config.json")
    if os.path.isfile(cf):
        cfg = json.load(open(cf))
    ttft_b = float(cfg.get("ttft_budget_ms") or 0)
    tpot_b = float(cfg.get("tpot_budget_ms") or 0)

    runs = {}
    for pol in sorted(os.listdir(ROOT)):
        pd = os.path.join(ROOT, pol)
        if not os.path.isdir(pd) or pol in ("topology", "logs"):
            continue
        for rd in sorted(os.listdir(pd)):
            if not rd.startswith("rate_"):
                continue
            r = load_run(os.path.join(pd, rd))
            if r:
                runs[f"{pol}@{rd[5:]}"] = r
    if not runs:
        sys.exit(f"no completed runs under {ROOT}")

    # SLO attainment needs the raw rows again; cheap enough to redo.
    for key in runs:
        pol, rate = key.split("@")
        f = os.path.join(ROOT, pol, f"rate_{rate}", "results.csv")
        ok = [r for r in csv.DictReader(open(f)) if not r.get("error")]
        user = [float(r["scheduling_ms"]) + float(r["forwarding_ms"])
                + float(r["queue_ms"]) + float(r["ttft_ms"]) for r in ok]
        runs[key]["ttft_met"] = (sum(1 for v in user if v <= ttft_b) / len(ok)
                                 if ttft_b else float("nan"))
        runs[key]["tpot_met"] = (sum(1 for r in ok
                                     if float(r["avg_tpot_ms"]) <= tpot_b) / len(ok)
                                 if tpot_b else float("nan"))

    print(f"# {ROOT}")
    if cfg:
        print(f"#   {cfg.get('duration')}s  TTFT<{ttft_b:.0f}ms  TPOT<{tpot_b:.0f}ms  "
              f"bs_cap={cfg.get('bs_cap')}  max_num_seqs={cfg.get('max_batch_size')}  "
              f"poll={cfg.get('poll_mode')}/{cfg.get('poll_period_s', '?')}s")
    print()
    # One block per arrival rate, ascending, and within a block the policies
    # ordered by carbon. A single table sorted by carbon across rates puts a
    # cheap policy at a low rate beside an expensive one at a high rate, which
    # invites exactly the comparison that is not valid: carbon and attainment
    # both move with load, so only same-rate rows can be read against each
    # other. The saving column is against round_robin AT THAT RATE.
    hdr = (f"{'policy':<30}{'n':>7}{'err':>6}{'carbon g':>10}{'vs RR':>7}{'work g':>9}"
           f"{'gCO2/kWh':>10}{'uTTFT p50':>11}{'p90':>9}{'p99':>9}{'TPOT p50':>10}{'p90':>8}{'p99':>8}"
           f"{'TTFT ok':>9}{'TPOT ok':>9}{'DCs':>5}{'top%':>7}{'fwd%':>7}")
    by_rate = {}
    skipped = []
    for key, r in runs.items():
        pol, rate = key.rsplit("@", 1)
        try:
            rate_f = float(rate)
        except ValueError:
            # A directory whose name is not a rate. This is how a bad cell
            # gets quarantined -- renamed rather than deleted, so the evidence
            # survives and the analysis cannot silently read it. Say which,
            # because a cell missing from a table is otherwise invisible.
            skipped.append(f"{pol}@{rate}")
            continue
        by_rate.setdefault(rate_f, {})[pol] = r
    if skipped:
        print(f"# SKIPPED {len(skipped)} quarantined cell(s): "
              f"{', '.join(sorted(skipped))}\n")
    for rate in sorted(by_rate):
        block = by_rate[rate]
        base = block.get("round_robin")
        print(f"── {rate:.0f} req/s" + (f"   (round_robin {base['carbon_g']:.1f} g, "
              f"TTFT {base['ttft_met']*100:.0f}% / TPOT {base['tpot_met']*100:.0f}%)"
              if base else "   (no round_robin baseline at this rate)"))
        print(hdr)
        print("-" * len(hdr))
        for pol in sorted(block, key=lambda p: block[p]["carbon_g"]):
            r = block[pol]
            vs = (f"{100*(1-r['carbon_g']/base['carbon_g']):>6.0f}%"
                  if base and base["carbon_g"] else f"{'-':>7}")
            print(f"{pol:<30}{r['n']:>7}{r['err']:>6}{r['carbon_g']:>10.1f}{vs}"
                  f"{r['work_carbon_g']:>9.1f}{r['gco2_per_kwh']:>10.1f}"
                  f"{r['user_p50']:>11.0f}{r['user_p90']:>9.0f}{r['user_p99']:>9.0f}"
                  f"{r['tpot_p50']:>10.1f}{r['tpot_p90']:>8.1f}{r['tpot_p99']:>8.1f}"
                  f"{r['ttft_met']*100:>8.0f}%{r['tpot_met']*100:>8.0f}%"
                  f"{r['dcs_used']:>5}{r['top_share']*100:>6.0f}%{r['fwd_share']*100:>6.0f}%")
        print()
    print("carbon g   whole fleet, including the idle floor      work g  serving only")
    print("gCO2/kWh   emission factor of the served work: total carbon / total")
    print("           energy, energy-weighted, PUE included. A placement result,")
    print("           not a property of the policy.")
    print("uTTFT      sched + fwd + queue + ttft, ms -- the FULL round trip,")
    print("           because TTFT is measured at the client")
    print("TPOT ok    share within budget, MEASURED -- includes prefill blocking")
    print("           decode, which the scheduler's own estimate does not carry")
    print("top%       share of requests on the single busiest datacentre")


if __name__ == "__main__":
    main()
