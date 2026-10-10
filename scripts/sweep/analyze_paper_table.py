#!/usr/bin/env python3
"""Paper-ready analysis table for a parallel sweep.

Loads each policy*rate's results.csv and energy.json from a sweep directory
and produces a wide CSV (one row per (policy, rate)) plus a human-readable
text report with carbon savings, SLO compliance, latency percentiles, and
scheduler overhead.

Usage:
    python scripts/sweep/analyze_paper_table.py results/sweep_parallel_YYYYMMDD_HHMM \\
        [--ttft-budget 2000] [--tpot-budget 50]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def user_ttft(df: pd.DataFrame) -> pd.Series:
    """User-perceived TTFT: scheduling + half RTT + queue + first-token latency."""
    return df.scheduling_ms + df.forwarding_ms / 2 + df.queue_ms + df.ttft_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--ttft-budget", type=float, default=None,
                    help="Override TTFT budget (defaults to manifest.ttft_budget_ms)")
    ap.add_argument("--tpot-budget", type=float, default=None,
                    help="Override TPOT budget (defaults to manifest.tpot_budget_ms)")
    args = ap.parse_args()

    sweep_dir = args.sweep_dir
    if not sweep_dir.is_dir():
        print(f"ERROR: not a directory: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    manifest = {}
    if (sweep_dir / "manifest.json").exists():
        manifest = json.load(open(sweep_dir / "manifest.json"))

    ttft_budget = args.ttft_budget or manifest.get("ttft_budget_ms", 2000)
    tpot_budget = args.tpot_budget or manifest.get("tpot_budget_ms", 50)
    print(f"Using TTFT budget = {ttft_budget} ms, TPOT budget = {tpot_budget} ms\n")

    # Discover policy/rate directories
    policy_dirs = []
    for p in sorted(sweep_dir.iterdir()):
        if not p.is_dir():
            continue
        if any(d.is_dir() and d.name.startswith("rate_") for d in p.iterdir()):
            policy_dirs.append(p)

    rows = []
    def _rate_of(d):
        """The rate a directory names, or None if it does not name one.

        A bad cell is QUARANTINED by renaming rather than deleting, so the
        evidence survives and no analysis reads it by accident -- grid_v3 has
        `rate_1200_VOID_provisioner_failed`. Splitting on "_" and taking field
        1 turned that back into 1200, which produced a SECOND
        carbon_aware_a0.5 row at that rate and a duplicate-label crash three
        steps later. Requiring the whole suffix to be a number is what makes
        the rename mean what it says.
        """
        if not (d.is_dir() and d.name.startswith("rate_")):
            return None
        tail = d.name[len("rate_"):]
        return int(tail) if tail.isdigit() else None

    quarantined = []
    for pdir in policy_dirs:
        policy = pdir.name
        for rdir in sorted(pdir.iterdir(),
                           key=lambda d: _rate_of(d) or 0):
            rate = _rate_of(rdir)
            if rate is None:
                if rdir.is_dir() and rdir.name.startswith("rate_"):
                    quarantined.append(f"{policy}/{rdir.name}")
                continue
            results_csv = rdir / "results.csv.gz"
            if not results_csv.exists():
                results_csv = rdir / "results.csv"
            energy_json = rdir / "energy.json"
            if not results_csv.exists():
                print(f"  skip {policy}/rate_{rate} (no results.csv)")
                continue
            df = pd.read_csv(results_csv)
            n_total = len(df)
            n_err = df.error.notna().sum() if "error" in df.columns else 0
            ok = df[df.error.isna()] if "error" in df.columns else df

            # SLO compliance: must satisfy BOTH TTFT and TPOT budgets
            ttft_user = user_ttft(ok)
            tpot = ok.avg_tpot_ms

            # The same numbers over the middle of the run. A run starts with
            # empty queues and ends with them draining, and both ends pull
            # latency down, so percentiles over every row describe the run
            # rather than the load. There is no warm-up to discard here -- this
            # is a simulator, with no cold cache and no clock ramp -- but there
            # IS a transient to steady state, and until now the sweep assumed it
            # away where the within-node replay measures it (_completion_slope,
            # same 30-90% convention). Reported alongside rather than instead:
            # one run, both views, and the gap between them is the evidence for
            # whether the assumption held. req_id counts up from 1 in arrival
            # order, which is what makes the slice possible.
            _n = ok.req_id.astype(str).str.extract(r"(\d+)$")[0].astype(float)
            _ord = ok.assign(_seq=_n).sort_values("_seq")
            _lo, _hi = int(0.3 * len(_ord)), int(0.9 * len(_ord))
            steady = _ord.iloc[_lo:_hi]
            ttft_st = user_ttft(steady)
            tpot_st = steady.avg_tpot_ms
            slo_st = ((ttft_st <= ttft_budget) & (tpot_st <= tpot_budget))
            slo_st_pct = (100.0 * slo_st.sum() / len(steady)) if len(steady) else 0.0
            ttft_ok = ttft_user <= ttft_budget
            tpot_ok = tpot <= tpot_budget
            slo_met = ttft_ok & tpot_ok
            slo_pct = 100.0 * slo_met.sum() / max(n_total, 1)
            violation_pct = 100.0 - slo_pct

            # Energy / carbon
            total_carbon_g = total_energy_j = 0.0
            energy_err_count = 0
            if energy_json.exists():
                try:
                    energy = json.load(open(energy_json))
                    total_carbon_g = energy.get("total_carbon_g", 0.0)
                    total_energy_j = energy.get("total_energy_j", 0.0)
                    energy_err_count = sum(
                        1 for v in energy.get("per_dc", {}).values() if "error" in v
                    )
                except (json.JSONDecodeError, ValueError):
                    pass

            row = {
                "policy": policy,
                "rate": rate,
                "n_total": n_total,
                "n_err": int(n_err),
                "slo_met_pct": round(slo_pct, 2),
                "violation_pct": round(violation_pct, 2),
                # TTFT percentiles
                "ttft_p50": round(ttft_user.median(), 1),
                "ttft_p90": round(ttft_user.quantile(0.9), 1),
                "ttft_p99": round(ttft_user.quantile(0.99), 1),
                "ttft_max": round(ttft_user.max(), 1),
                # TPOT percentiles
                "tpot_p50": round(tpot.median(), 2),
                "tpot_p90": round(tpot.quantile(0.9), 2),
                "tpot_p99": round(tpot.quantile(0.99), 2),
                "tpot_max": round(tpot.max(), 2),
                # Same, over the steady 30-90% window (see above)
                "n_steady": int(len(steady)),
                "ttft_p90_steady": round(ttft_st.quantile(0.9), 1) if len(steady) else None,
                "ttft_p99_steady": round(ttft_st.quantile(0.99), 1) if len(steady) else None,
                "tpot_p90_steady": round(tpot_st.quantile(0.9), 2) if len(steady) else None,
                "slo_met_pct_steady": round(slo_st_pct, 2),
                # Scheduler / queue diagnostics
                "sched_p50": round(ok.scheduling_ms.median(), 2),
                "sched_p90": round(ok.scheduling_ms.quantile(0.9), 2),
                "queue_p99": round(ok.queue_ms.quantile(0.99), 1),
                # Energy / carbon
                "total_carbon_g": round(total_carbon_g, 1),
                "total_energy_j": round(total_energy_j, 1),
                "energy_dc_errors": energy_err_count,
            }
            rows.append(row)

    if not rows:
        print("ERROR: no policy×rate combinations found", file=sys.stderr)
        sys.exit(1)

    if quarantined:
        print(f"  SKIPPED {len(quarantined)} quarantined cell(s): "
              f"{', '.join(sorted(quarantined))}")
    df_all = pd.DataFrame(rows)

    # Carbon savings vs round_robin per rate
    if "round_robin" in df_all.policy.unique():
        rr_carbon = df_all[df_all.policy == "round_robin"].set_index("rate").total_carbon_g.to_dict()
        df_all["carbon_savings_pct_vs_rr"] = df_all.apply(
            lambda r: round((rr_carbon.get(r.rate, 0) - r.total_carbon_g) / max(rr_carbon.get(r.rate, 1), 1) * 100, 1)
            if rr_carbon.get(r.rate, 0) > 0 else 0,
            axis=1,
        )

    out_csv = sweep_dir / "paper_table.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  ({len(df_all)} rows)\n")

    # Sanity: warn if any policy/rate has energy collection errors
    bad = df_all[df_all.energy_dc_errors > 0]
    if len(bad):
        print(f"WARNING: {len(bad)} policy*rate combinations have energy_dc_errors > 0:")
        print(bad[["policy", "rate", "energy_dc_errors", "total_carbon_g"]].to_string(index=False))
        print("(These were affected by the cross-slot pkill bug. Rerun if needed.)\n")

    # Pretty per-rate table for the paper, restricted to focus policies in fixed order
    FOCUS = [
        "round_robin",
        "least_loaded",
        "carbon_aware_a0.25",
        "carbon_aware_a0.5",
        "carbon_aware_a0.9",
        "ilp_carbon_sla",
        "iterative_ilp_carbon_sla",
    ]

    out_lines = []
    for rate in sorted(df_all.rate.unique()):
        sub = df_all[df_all.rate == rate].set_index("policy")
        sub = sub.reindex([p for p in FOCUS if p in sub.index])
        out_lines.append(f"\n{'=' * 130}")
        out_lines.append(f"  Rate {rate} req/s, TTFT budget {ttft_budget} ms, TPOT budget {tpot_budget} ms")
        out_lines.append(f"{'=' * 130}")
        cols = [
            "slo_met_pct", "violation_pct",
            "ttft_p50", "ttft_p90", "ttft_p99", "ttft_max",
            "tpot_p50", "tpot_p90", "tpot_p99", "tpot_max",
            "sched_p90",
            "total_carbon_g",
        ]
        if "carbon_savings_pct_vs_rr" in sub.columns:
            cols.append("carbon_savings_pct_vs_rr")
        out_lines.append(sub[cols].to_string())

    txt_path = sweep_dir / "paper_table.txt"
    txt_path.write_text("\n".join(out_lines))
    print(f"Wrote {txt_path}")
    print("\n".join(out_lines))

    # Iter-vs-Std head-to-head focus block
    print("\n" + "=" * 80)
    print("  Iter ILP vs Std ILP, tail-latency advantage")
    print("=" * 80)
    if {"iterative_ilp_carbon_sla", "ilp_carbon_sla"} <= set(df_all.policy.unique()):
        i = df_all[df_all.policy == "iterative_ilp_carbon_sla"].set_index("rate")
        s = df_all[df_all.policy == "ilp_carbon_sla"].set_index("rate")
        for rate in sorted(set(i.index) & set(s.index)):
            print(f"\n  rate = {rate}:")
            print(f"    SLO%       Iter={i.loc[rate].slo_met_pct:5.2f}  Std={s.loc[rate].slo_met_pct:5.2f}  diff={i.loc[rate].slo_met_pct - s.loc[rate].slo_met_pct:+5.2f} pp")
            print(f"    TTFT p99   Iter={i.loc[rate].ttft_p99:7.1f}  Std={s.loc[rate].ttft_p99:7.1f}  diff={i.loc[rate].ttft_p99 - s.loc[rate].ttft_p99:+7.1f} ms")
            print(f"    TTFT max   Iter={i.loc[rate].ttft_max:7.1f}  Std={s.loc[rate].ttft_max:7.1f}  diff={i.loc[rate].ttft_max - s.loc[rate].ttft_max:+7.1f} ms")
            print(f"    TPOT p99   Iter={i.loc[rate].tpot_p99:6.2f}  Std={s.loc[rate].tpot_p99:6.2f}  diff={i.loc[rate].tpot_p99 - s.loc[rate].tpot_p99:+6.2f} ms")
            print(f"    TPOT max   Iter={i.loc[rate].tpot_max:6.2f}  Std={s.loc[rate].tpot_max:6.2f}  diff={i.loc[rate].tpot_max - s.loc[rate].tpot_max:+6.2f} ms")
            print(f"    Carbon (g) Iter={i.loc[rate].total_carbon_g:7.1f}  Std={s.loc[rate].total_carbon_g:7.1f}  diff={i.loc[rate].total_carbon_g - s.loc[rate].total_carbon_g:+7.1f} g")


if __name__ == "__main__":
    main()
