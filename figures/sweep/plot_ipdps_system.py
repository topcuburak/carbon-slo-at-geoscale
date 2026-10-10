"""System-side figure: scheduling overhead, polling period, and topology.

(a) Per-request scheduling time at high load, p50 to p99 with a tick at p90, one
    row per policy and one colour per model (log scale: greedy is sub-ms, the
    iterative ILP reaches seconds).
(b) Facility MILP SLO violation against the polling period at moderate load,
    synchronous (solid) and asynchronous (dashed) polling, one colour per model.
(c) Llama-3.1-8B at high load on three topologies over the v9 sites: the base
    (4 schedulers, 16 DCs), 8 schedulers on the same 16 DCs, and 4 schedulers on
    8 larger DCs. Marker = policy, colour = topology.

Greedy appears only in (a), as its cap-less runs; (b) and (c) ran it with a batch
capacity, unlike the main comparison.

Inputs: data/figures/ipdps_cells_f4.csv, poll_cells_new.csv (sync),
poll_cells_async.csv, topo_cells.csv (all from summarize_cells.py).

    python3 figures/sweep/plot_ipdps_system.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, MultipleLocator
import pandas as pd

from plot_ipdps_frontier import INK, GRID  # also applies the shared rcParams

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/figures"
MILP = "facility_ilp_carbon_sla"
# (model, label, high rate, moderate rate, colour). Panels (a) and (b) share their own
# palette, distinct from the earlier figures: Okabe-Ito bluish green and reddish purple,
# a colour-vision-deficiency-safe pair.
MODELS = [("llama3_8B", "Llama-3.1-8B", 4000, 2400, "#009E73"),
          ("mixtral_8x7b", r"Mixtral-8$\times$7B", 2000, 1200, "#9B1B1B")]
# Two-line row labels keep the label column of the narrow overhead panel short.
OVERHEAD_POLICIES = [("carbon_aware_a0.0", "G,\n" + r"$\alpha$=0"), ("ilp_carbon_sla", "Std\nILP"),
                     ("iterative_ilp_carbon_sla", "Iter\nILP"), (MILP, "Fac.\nMILP"),
                     ("greenllm", "Green-\nLLM")]
POLLS = [0.5, 1, 2, 5, 10]
# Moderate load for both models: the regime where each has polling runs at its Fig. 4 rate.
POLL_RATE = {"llama3_8B": 2400, "mixtral_8x7b": 1200}
TOPOS = [("base", "4 sch. / 16 DCs (base)", "#9a9893"),
         ("sched8", "8 sch. / 16 DCs", "#7b5ea7"),
         ("dc8", "4 sch. / 8 DCs (larger)", "#c9a227")]
TOPO_POLICIES = [("ilp_carbon_sla", "Std ILP", "D"), ("iterative_ilp_carbon_sla", "Iter ILP", "p"),
                 (MILP, "Fac. MILP", "*")]


def style(ax):
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def boxed(leg):
    leg.get_frame().set_edgecolor(INK)
    leg.get_frame().set_linewidth(0.5)


def with_saving(df, keys):
    rr = df[df.policy == "round_robin"].set_index(keys + ["rate"]).carbon_g.rename("rr_g")
    df = df.join(rr, on=keys + ["rate"])
    df["saving"] = 100.0 * (1.0 - df.carbon_g / df.rr_g)
    df["violation"] = 100.0 - df.slo_pct
    return df


def overhead_panel(ax, f4):
    n = len(OVERHEAD_POLICIES)
    for k, (m, lab, high, _, color) in enumerate(MODELS):
        off = 0.18 if k == 0 else -0.18             # Llama above, Mixtral below, per row
        for row, (pol, _) in enumerate(OVERHEAD_POLICIES):
            r = f4[(f4.model == m) & (f4.rate == high) & (f4.policy == pol)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = n - 1 - row + off
            p50, p90, p99 = (max(v, 0.05) for v in (r.sched_p50, r.sched_p90, r.sched_p99))
            ax.plot([p50, p99], [y, y], color=color, lw=1.4, solid_capstyle="butt", zorder=2)
            ax.plot([p50, p90], [y, y], ls="", marker="|", ms=5, mew=0.9, color=color, zorder=3)
            ax.plot(p99, y, "o", ms=4.5, mfc=color, mec=INK, mew=0.4, zorder=4)
    ax.set_xscale("log")
    # One label per decade, even on a narrow panel, with the minor ticks in between.
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=12))
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in OVERHEAD_POLICIES][::-1])
    ax.tick_params(axis="y", length=0, labelsize=6.5)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Scheduling time (ms)")


def poll_panel(ax, sync, asyn):
    for m, lab, _, _, color in MODELS:
        rate = POLL_RATE[m]
        for df, ls, prefix in ((sync, "-", "sync_"), (asyn, "--", "async_")):
            xs, ys = [], []
            for p in POLLS:
                tag = f"{prefix}{p:g}"
                r = df[(df.model == m) & (df.tag == tag) & (df.rate == rate) & (df.policy == MILP)]
                if not r.empty:
                    xs.append(POLLS.index(p))
                    ys.append(r.iloc[0].violation)
            ax.plot(xs, ys, ls=ls, lw=1.3, color=color, marker="*", ms=7.5,
                    mfc=color if ls == "-" else "white", mec=color if ls == "--" else INK,
                    mew=0.8 if ls == "--" else 0.4, zorder=3)
    ax.set_xticks(range(len(POLLS)))
    ax.set_xticklabels([f"{p:g}" for p in POLLS])
    ax.set_xlabel(r"Polling period $\Delta_p$ (s)")
    ax.set_ylabel("Fac. MILP SLO violation (%)")
    model_h = [Patch(facecolor=c, edgecolor="none", label=lab) for _, lab, _, _, c in MODELS]
    mode_h = [Line2D([], [], color=INK, ls="-", lw=1.4, label="Sync"),
              Line2D([], [], color=INK, ls="--", lw=1.4, label="Async")]
    leg = ax.legend(handles=model_h + mode_h, loc="upper left", frameon=True, fancybox=False,
                    fontsize=6, handlelength=1.6, handletextpad=0.3, borderpad=0.3, labelspacing=0.3)
    boxed(leg)


def topo_panel(ax, topo):
    high = topo[topo.rate == MODELS[0][2]]
    # The facility MILP's moves from the base show the trade-off: more schedulers on the same
    # DCs cost SLO at flat carbon, fewer larger DCs buy SLO at a lower saving.
    it = high[high.policy == MILP].set_index("tag")
    if "base" in it.index:
        b = it.loc["base"]
        for tag, _, color in TOPOS[1:]:
            if tag in it.index:
                t = it.loc[tag]
                ax.annotate("", xy=(t.violation, t.saving), xytext=(b.violation, b.saving),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                            shrinkA=4, shrinkB=4, mutation_scale=8), zorder=2)
    for tag, _, color in TOPOS:
        for pol, _, mk in TOPO_POLICIES:
            r = high[(high.tag == tag) & (high.policy == pol)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot(r.violation, r.saving, mk, ms=8 if mk == "*" else 5.5, mfc=color, mec=INK,
                    mew=0.4, zorder=3)
    ax.set_xlabel("SLO violation (%)")
    ax.set_ylabel("Carbon saving vs RR (%)")
    pol_h = [Line2D([], [], ls="", marker=mk, ms=6.5 if mk == "*" else 4.8, mfc="#bdbcb8", mec=INK,
                    mew=0.5, label=lab) for _, lab, mk in TOPO_POLICIES]
    topo_h = [Patch(facecolor=c, edgecolor="none", label=lab) for _, lab, c in TOPOS]
    kw = dict(frameon=True, fancybox=False, fontsize=5.5, handlelength=1.2, handletextpad=0.3,
              borderpad=0.3, labelspacing=0.3)
    # Headroom above and below the points holds the two legends clear of points and arrows.
    lo, hi = high[high.policy != "round_robin"].saving.agg(["min", "max"])
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylim(lo - 3.6, hi + 6.4)
    leg = ax.legend(handles=pol_h, loc="lower right", ncol=len(pol_h),
                    borderaxespad=0.15,
                    **{**kw, "fontsize": 5, "columnspacing": 0.3, "handletextpad": 0.1})
    boxed(leg)
    ax.add_artist(leg)
    leg2 = ax.legend(handles=topo_h, loc="upper right", borderaxespad=0.15,
                     **{**kw, "fontsize": 5.5})
    boxed(leg2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "figures/sweep/overhead_poll_topology.png"))
    ap.add_argument("--width", type=float, default=5.0)
    ap.add_argument("--height", type=float, default=1.7)
    a = ap.parse_args()
    f4 = pd.read_csv(D / "ipdps_cells_f4.csv")
    sync = with_saving(pd.read_csv(D / "poll_cells_new.csv"), ["model", "tag"])
    asyn = with_saving(pd.read_csv(D / "poll_cells_async.csv"), ["model", "tag"])
    topo = with_saving(pd.read_csv(D / "topo_cells.csv"), ["tag"])
    # Overhead first, polling second, topology last. Spacer columns set the gaps: the one
    # before the overhead panel holds its policy labels, so (a) and (b) sit close.
    fig = plt.figure(figsize=(a.width, a.height))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.78, 0.36, 0.85, 0.36, 1.0], wspace=0.0)
    ax_ovh, ax_poll, ax_topo = fig.add_subplot(gs[0]), fig.add_subplot(gs[2]), fig.add_subplot(gs[4])
    poll_panel(ax_poll, sync, asyn)
    overhead_panel(ax_ovh, f4)
    topo_panel(ax_topo, topo)
    for ax in (ax_poll, ax_ovh, ax_topo):
        style(ax)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.tick_params(axis="both", which="major", labelsize=6.5)
    ax_poll.set_title("(b) Polling (mod. load)", fontsize=7.5)
    ax_ovh.set_title("(a) Sched. overhead (high)", fontsize=7.5)
    ax_topo.set_title("(c) Topology (Llama, high)", fontsize=7.5)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
