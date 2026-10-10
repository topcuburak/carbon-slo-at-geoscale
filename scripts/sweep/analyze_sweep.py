#!/usr/bin/env python3
"""Analyze a sweep directory: build summary CSV + Pareto plot.

Usage:
    python scripts/sweep/analyze_sweep.py results/sweep_YYYYMMDD_HHMM

Produces, inside the sweep dir:
    summary.csv          one row per (policy, rate)
    pareto.png           carbon vs SLA satisfaction
    tables/              per-rate PNG tables

──────────────────────────────────────────────────────────────────────
TTFT METRIC DEFINITION  (read this before reporting any number)
──────────────────────────────────────────────────────────────────────
User-perceived TTFT (the headline metric, written as `ttft_*`) is
defined as:

    ttft_user = scheduling_ms  +  forwarding_ms / 2
              + queue_ms  +  ttft_ms

where the four components correspond to:

    scheduling_ms     → time spent inside the scheduler's policy
                        decision (batch window wait + ILP solve, or
                        ~0 for greedy/round-robin).
    forwarding_ms / 2 → one-way network RTT from the user-facing
                        scheduler to the chosen datacenter. Only the
                        outbound leg is on the TTFT critical path.
    queue_ms          → time the request waits at the destination DC
                        for a GPU semaphore slot before joining a batch
                        (datacenter/inference_engine.py:230).
    ttft_ms           → prefill compute time inside the batch on the
                        GPU (the "DC compute TTFT").

The DC's raw `ttft_ms` field is preserved in the summary as
`ttft_compute_*` for diagnostics ONLY. Never report it as TTFT in a
paper without labelling it as "DC compute TTFT" — it understates the
user-visible cost by exactly `queue_ms + forwarding_ms/2`, which can
be tens of seconds for policies that overload a single low-CI DC.

The SLA gate (TTFT_BUDGET_MS, TPOT_BUDGET_MS) is checked against
`ttft_user`, not `ttft_ms`.
──────────────────────────────────────────────────────────────────────
"""

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

# ISL-aware TTFT SLA. TTFT_BUDGET_MS is the additive overhead slack in ms
# (queue + forwarding + batch contention the scheduler is allowed to add on
# top of the unavoidable prefill compute). The per-request budget is then
#   budget(ISL) = TTFT_BUDGET_MS + TTFT_ALPHA * analytical_prefill_ms(ISL)
# Must stay in sync with scheduler/policies/{ilp_carbon, greedy/*}._check_*
TTFT_BUDGET_MS = 1500
TTFT_ALPHA = 1.5
TPOT_BUDGET_MS = 60

# Linear TTFT model (LLaMA-3.1-8B, A100 TP=2) used to derive the ISL-aware
# budget post-hoc. It used to say this must match a ttft_fit_*_3ds.json that
# nothing opened and that has since been deleted; the live fit is
# perf_models/analytical/fit_data/latency_fit_tp2_llama3_8B.json
# (and the fitted artifact datacenter/perf_models/analytical/fit_data/latency_fit_tp2_llama3_8B.json).
_TTFT_A0 = 11.498
_TTFT_A1 = 0.047790
_TTFT_A2 = -0.9639  # BS coefficient; BS=1 for single-request budget

def _ttft_budget_for_isl(isl: int) -> float:
    analytical_prefill_ms = _TTFT_A0 + _TTFT_A1 * max(0, isl) + _TTFT_A2 * 1
    return TTFT_BUDGET_MS + TTFT_ALPHA * analytical_prefill_ms


def pct(values, p):
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def analyze_run(run_dir):
    results_csv = os.path.join(run_dir, "results.csv")
    energy_json = os.path.join(run_dir, "energy.json")
    if not (os.path.isfile(results_csv) and os.path.isfile(energy_json)):
        return None

    wall, ttft, tpot, queue, fwd, sched = [], [], [], [], [], []
    n_total = 0
    n_ok = 0
    n_sla_ok = 0
    n_err = 0

    # User-perceived TTFT is what matters for the SLA. The DC's `ttft_ms`
    # only measures prefill compute *after* the request joins a batch. We
    # add the DC-side GPU queue wait and the one-way network forwarding
    # to recover what the user actually sees.
    ttft_compute = []  # raw DC ttft (kept for diagnostics)

    with open(results_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            if row.get("status") != "OK":
                n_err += 1
                continue
            n_ok += 1
            try:
                w = float(row["wall_ms"])
                t_compute = float(row["ttft_ms"])
                tp = float(row["avg_tpot_ms"])
                q = float(row["queue_ms"])
                fw = float(row["forwarding_ms"])
                sc = float(row["scheduling_ms"])
                isl = int(row.get("in_tok", 0))
            except (ValueError, KeyError):
                continue
            # User-perceived TTFT = scheduling + one-way RTT + DC GPU queue + prefill compute
            t_user = sc + (fw / 2.0) + q + t_compute
            wall.append(w)
            ttft.append(t_user)
            ttft_compute.append(t_compute)
            tpot.append(tp)
            queue.append(q)
            fwd.append(fw)
            sched.append(sc)
            # ISL-aware TTFT SLA: budget(ISL) = slack + α · analytical_prefill_ms(ISL)
            ttft_budget_this_req = _ttft_budget_for_isl(isl)
            if t_user < ttft_budget_this_req and tp < TPOT_BUDGET_MS:
                n_sla_ok += 1

    with open(energy_json) as f:
        energy = json.load(f)

    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_err": n_err,
        # SLA rate is over ALL requests (errors count as SLA violations).
        "sla_rate": (n_sla_ok / n_total) if n_total else 0.0,
        # Kept for curiosity: SLA rate conditional on success.
        "sla_rate_ok_only": (n_sla_ok / n_ok) if n_ok else 0.0,
        "ok_rate": (n_ok / n_total) if n_total else 0.0,
        "err_rate": (n_err / n_total) if n_total else 0.0,
        "wall_p50": pct(wall, 50), "wall_p90": pct(wall, 90), "wall_p99": pct(wall, 99),
        # ttft_p* now reflects user-perceived TTFT (one-way RTT + queue + compute)
        "ttft_p50": pct(ttft, 50), "ttft_p90": pct(ttft, 90), "ttft_p99": pct(ttft, 99),
        # Diagnostic: raw DC compute-only TTFT before queueing
        "ttft_compute_p50": pct(ttft_compute, 50),
        "ttft_compute_p90": pct(ttft_compute, 90),
        "ttft_compute_p99": pct(ttft_compute, 99),
        "tpot_p50": pct(tpot, 50), "tpot_p90": pct(tpot, 90), "tpot_p99": pct(tpot, 99),
        "queue_p90": pct(queue, 90),
        "sched_p90": pct(sched, 90),
        "fwd_p90": pct(fwd, 90),
        "total_energy_j": energy.get("total_energy_j", 0.0),
        "total_energy_kwh": energy.get("total_energy_kwh", 0.0),
        "total_carbon_g": energy.get("total_carbon_g", 0.0),
        "sim_duration_s": energy.get("sim_duration_s", 0.0),
    }


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_sweep.py <sweep_dir>", file=sys.stderr)
        sys.exit(1)
    sweep_dir = sys.argv[1].rstrip("/")

    rows = []
    for policy in sorted(os.listdir(sweep_dir)):
        pdir = os.path.join(sweep_dir, policy)
        if not os.path.isdir(pdir):
            continue
        for rate_dir in sorted(os.listdir(pdir)):
            if not rate_dir.startswith("rate_"):
                continue
            rate = int(rate_dir.split("_")[1])
            run_dir = os.path.join(pdir, rate_dir)
            stats = analyze_run(run_dir)
            if stats is None:
                continue
            stats["policy"] = policy
            stats["rate"] = rate
            rows.append(stats)

    if not rows:
        print("No runs found.", file=sys.stderr)
        sys.exit(1)

    # Write summary.csv
    cols = [
        "policy", "rate", "n_total", "n_ok", "n_err",
        "ok_rate", "err_rate", "sla_rate", "sla_rate_ok_only",
        "wall_p50", "wall_p90", "wall_p99",
        "ttft_p50", "ttft_p90", "ttft_p99",
        "ttft_compute_p50", "ttft_compute_p90", "ttft_compute_p99",
        "tpot_p50", "tpot_p90", "tpot_p99",
        "queue_p90", "sched_p90", "fwd_p90",
        "total_energy_j", "total_energy_kwh", "total_carbon_g", "sim_duration_s",
    ]
    summary_path = os.path.join(sweep_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"Wrote {summary_path} ({len(rows)} rows)")

    # Compute carbon savings vs round_robin baseline per rate
    baseline = {r["rate"]: r["total_carbon_g"] for r in rows if r["policy"] == "round_robin"}
    print()
    print(f"{'policy':<30} {'rate':>4} {'carbon_g':>10} {'vs_RR':>8} "
          f"{'SLA%':>7} {'err%':>6} "
          f"{'ttft_p90':>10} {'tpot_p90':>10} {'wall_p90':>10}")
    print("-" * 110)
    for r in sorted(rows, key=lambda x: (x["rate"], x["policy"])):
        base = baseline.get(r["rate"])
        sav = f"{100*(base - r['total_carbon_g'])/base:+6.1f}%" if base else "  n/a  "
        print(f"{r['policy']:<30} {r['rate']:>4} "
              f"{r['total_carbon_g']:>10.1f} {sav:>8} "
              f"{100*r['sla_rate']:>6.1f}% "
              f"{100*r['err_rate']:>5.1f}% "
              f"{r['ttft_p90']:>10.1f} {r['tpot_p90']:>10.2f} {r['wall_p90']:>10.1f}")

    # Pareto plot: SLA rate vs carbon, one subplot per rate
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping plot)", file=sys.stderr)
        return

    rates = sorted({r["rate"] for r in rows})
    policies = sorted({r["policy"] for r in rows})
    cmap = plt.get_cmap("tab10")
    color = {p: cmap(i % 10) for i, p in enumerate(policies)}

    # Layout the subplots in a near-square grid, favoring 3 columns when
    # we have many rates. For 8 rates this produces a 3x3 grid with the
    # last cell empty (3-3-2 visual layout).
    n_rates = len(rates)
    if n_rates <= 3:
        ncols = n_rates
    else:
        ncols = 3
    nrows = (n_rates + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4.2 * nrows),
        sharey=False,
    )
    # Normalize axes to a flat list regardless of shape
    if n_rates == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(axes.flat) if hasattr(axes, "flat") else list(axes)

    # Bumped sizes (+3 from defaults): font 13/16/16, marker s=140
    label_fs = 13
    title_fs = 16
    tick_fs = 13
    for ax, rate in zip(axes_flat, rates):
        for r in rows:
            if r["rate"] != rate:
                continue
            ax.scatter(r["total_carbon_g"], 100 * r["sla_rate"],
                       s=140, color=color[r["policy"]], label=r["policy"],
                       edgecolor="black", linewidth=0.6)
        ax.set_xlabel("Total carbon (g CO₂)", fontsize=label_fs)
        ax.set_ylabel("SLA satisfaction (%)", fontsize=label_fs)
        ax.set_title(f"rate = {rate} req/s", fontsize=title_fs, weight="bold")
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(alpha=0.3)

    # Hide any unused axes (e.g. the 9th cell in a 3x3 grid with 8 rates)
    for ax in axes_flat[n_rates:]:
        ax.set_visible(False)

    # De-dup legend — collect from all visible axes since some may have
    # more/fewer points depending on data.
    handles, labels = [], []
    for ax in axes_flat[:n_rates]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               ncol=min(5, len(seen)), bbox_to_anchor=(0.5, -0.01),
               fontsize=13)
    # Leave room at the bottom for the legend
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plot_path = os.path.join(sweep_dir, "pareto.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nWrote {plot_path}")

    # ── Per-rate summary tables (one PNG per rate) ──
    tables_dir = os.path.join(sweep_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # Stable policy ordering: baselines → carbon_aware (by α) → SLA-aware
    def policy_sort_key(p):
        order = {
            "round_robin": 0,
            "least_loaded": 1,
            "carbon_aware_a0.95": 2,
            "carbon_aware_a0.9": 3,
            "carbon_aware_a0.75": 4,
            "carbon_aware_a0.5": 5,
            "ilp_carbon_sla": 7,
            "iterative_ilp_carbon_sla": 8,
        }
        return order.get(p, 99)

    # Each column is (group, sub_label, data_key).
    # group=None means a single un-grouped column (renders as one wide
    # header spanning both header rows). Columns sharing a group name are
    # rendered with that group label spanning above their sub_labels.
    table_cols = [
        (None, "Policy", "policy"),
        (None, "Carbon (g)", "total_carbon_g"),
        (None, "Energy (kWh)", "total_energy_kwh"),
        (None, "SLA%", "sla_rate"),
        (None, "err%", "err_rate"),
        ("user TTFT (ms)", "p50", "ttft_p50"),
        ("user TTFT (ms)", "p90", "ttft_p90"),
        ("DC TTFT (ms)", "p50", "ttft_compute_p50"),
        ("DC TTFT (ms)", "p90", "ttft_compute_p90"),
        ("TPOT (ms)", "p50", "tpot_p50"),
        ("TPOT (ms)", "p90", "tpot_p90"),
        ("Wall (ms)", "p50", "wall_p50"),
        ("Wall (ms)", "p90", "wall_p90"),
    ]

    def fmt(key, r):
        v = r.get(key)
        if key == "policy":
            # Wrap long policy names onto two lines so they fit inside the
            # cell. Split on the FIRST underscore — that puts the family
            # name (carbon_aware, ilp_carbon, …) on top and the variant
            # tag (a0.5, sla, …) on the bottom.
            name = r["policy"]
            if "_" in name and len(name) > 12:
                head, tail = name.split("_", 1)
                return f"{head}_\n{tail}"
            return name
        if key == "total_carbon_g":
            return f"{v:.1f}"
        if key == "total_energy_kwh":
            return f"{v:.3f}"
        if key in ("sla_rate", "err_rate"):
            return f"{100*v:.1f}%"
        # latency cols (ms)
        return f"{v:.0f}" if v == v else "—"  # NaN check

    # Pre-compute group spans (start_col, end_col_inclusive) for headers
    n_cols = len(table_cols)
    group_spans = []
    j = 0
    while j < n_cols:
        g = table_cols[j][0]
        if g is None:
            j += 1
            continue
        k = j
        while k < n_cols and table_cols[k][0] == g:
            k += 1
        group_spans.append((j, k - 1, g))
        j = k

    for rate in rates:
        rate_rows = sorted(
            [r for r in rows if r["rate"] == rate],
            key=lambda r: policy_sort_key(r["policy"]),
        )

        # We render only ONE header row in the table itself, containing
        # the per-column sub-labels (p50/p90 for grouped cols, the full
        # name for ungrouped cols). The group labels (user TTFT, etc.)
        # are drawn as text annotations above the table afterwards.
        col_labels = []
        for g, sub, _ in table_cols:
            col_labels.append(sub)

        cell_text = []
        for r in rate_rows:
            cell_text.append([fmt(c[2], r) for c in table_cols])

        # Highlight Pareto-optimal cells: best carbon, best SLA
        best_carbon_idx = min(range(len(rate_rows)), key=lambda i: rate_rows[i]["total_carbon_g"])
        best_sla_idx = max(range(len(rate_rows)), key=lambda i: rate_rows[i]["sla_rate"])

        n_rows = len(rate_rows)
        # Heights: data rows (taller now for 2-line policy names) + header
        # strip + title + legend below
        fig_h = max(3.5, 0.55 * (n_rows + 1) + 3.0)
        fig_w = 0.95 * n_cols + 1.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        ax.set_title(
            f"Rate = {rate} req/s   "
            f"({rate_rows[0]['n_total']} reqs, {rate_rows[0]['sim_duration_s']:.0f}s sim)",
            fontsize=12, pad=14, weight="bold",
        )

        # Anchor the table at the top of the axes (just below the title) so
        # there is no white gap and the legend below has predictable space.
        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="upper center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        # Taller rows so wrapped two-line policy names have room
        table.scale(1.0, 1.9)

        # Style the single header row (row 0)
        for jcol in range(n_cols):
            c = table[(0, jcol)]
            c.set_facecolor("#34495e")
            c.set_text_props(color="white", weight="bold")

        # ── Draw group-label strip above the table ──
        # Need actual cell coordinates; matplotlib computes them after draw
        fig.canvas.draw()
        # Height of one header cell in axes coords (matches the strip)
        header_cell = table[(0, 0)]
        bbox = header_cell.get_window_extent(fig.canvas.get_renderer())
        # Convert to axes-fraction coords for our group strip
        bbox_ax = bbox.transformed(ax.transAxes.inverted())
        cell_h = bbox_ax.height
        # Position group strip directly above row 0 of the table
        group_y_top = bbox_ax.y1 + cell_h
        group_y_bot = bbox_ax.y1

        for start, end, label in group_spans:
            left_cell  = table[(0, start)]
            right_cell = table[(0, end)]
            left_bbox  = left_cell.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
            right_bbox = right_cell.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
            x0 = left_bbox.x0
            x1 = right_bbox.x1
            # Background rectangle
            ax.add_patch(plt.Rectangle(
                (x0, group_y_bot), x1 - x0, group_y_top - group_y_bot,
                facecolor="#34495e", edgecolor="white", linewidth=1.0,
                transform=ax.transAxes, clip_on=False, zorder=4,
            ))
            # Label text
            ax.text((x0 + x1) / 2, (group_y_top + group_y_bot) / 2, label,
                    ha="center", va="center", color="white", fontsize=9,
                    weight="bold", transform=ax.transAxes, clip_on=False, zorder=5)

        # Look up column positions by data key
        col_idx = {c[2]: j for j, c in enumerate(table_cols)}

        # Row striping + Pareto highlighting (data rows start at row 1)
        for i in range(n_rows):
            base_color = "#ecf0f1" if i % 2 == 0 else "#ffffff"
            for jcol in range(n_cols):
                table[(i + 1, jcol)].set_facecolor(base_color)
            # Best carbon → green tint on the carbon column
            if i == best_carbon_idx:
                table[(i + 1, col_idx["total_carbon_g"])].set_facecolor("#a8e6a3")
            # Best SLA → blue tint on the SLA column
            if i == best_sla_idx:
                table[(i + 1, col_idx["sla_rate"])].set_facecolor("#a3d5e6")
            # Bold policy names; two-line format means we can use 8pt
            table[(i + 1, 0)].set_text_props(weight="bold", fontsize=8)

        # Itemized column legend below the table, left-aligned with the
        # table's left edge (looked up from the actual table bbox)
        legend_lines = [
            "Column definitions:",
            "  • Policy        scheduling policy under test",
            "  • Carbon (g)    total CO\u2082 emitted = \u03A3 (energy_j \u00D7 PUE \u00D7 CI) / 3.6e6 over all DCs (best in green)",
            "  • Energy (kWh)  total electrical energy across all DCs over the 120 s sim window",
            "  • SLA%          fraction of submitted requests where user TTFT < 2500 ms AND avg TPOT < 90 ms (best in blue)",
            "  • err%          fraction of requests that failed to complete (status \u2260 OK); errors count as SLA violations",
            "  • user TTFT     end-to-end time-to-first-token = scheduling + forwarding/2 + DC GPU queue + prefill compute",
            "  • DC TTFT       prefill compute only, after the request has joined a batch (diagnostic, do not report alone)",
            "  • TPOT          average time-per-output-token across the decode phase",
            "  • Wall          end-to-end wall-clock latency = scheduling + forwarding + DC queue + prefill + decode",
            "  • p50 / p90     50th and 90th percentile across all OK requests in the run",
        ]
        legend_text = "\n".join(legend_lines)
        # Anchor the legend just below the table's bottom-left corner.
        # We need a fresh draw because the table's bbox depends on its
        # final layout (cell heights, row count, etc.).
        fig.canvas.draw()
        last_row_idx = n_rows  # row 0 = header, last data row = n_rows
        first_cell = table[(0, 0)]
        last_cell  = table[(last_row_idx, 0)]
        first_fig = first_cell.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        last_fig  = last_cell.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        legend_x = first_fig.x0
        legend_y = last_fig.y0 - 0.005  # tiny gap so text doesn't touch border
        fig.text(legend_x, legend_y, legend_text, ha="left", va="top",
                 fontsize=7, family="monospace", color="#333333")

        out_path = os.path.join(tables_dir, f"table_rate_{rate:03d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
