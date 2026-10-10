"""Workload-side figure: OSL source and long/mixed prompts.

(a) Carbon saving and SLO violation at high load with oracle OSLs (hollow) and
    OSLs from the MLP predictor (filled), arrow from one to the other, for the
    ILP-based schedulers. Colour identifies the model, as in Figs. 4 and 5.
    Llama-3.1-8B compares two runs on the same code (oracle_now, predosl);
    Mixtral-8x7B has no same-code oracle run, so its oracle side is the main grid.
(b) Mixtral-8x7B on long (GovReport) and mixed prompts: the facility MILP's carbon
    saving against SLO violation as the per-engine batch capacity grows, one line
    per workload at one rate, the capacity printed at the ends.

Greedy is cap-less G alpha=0 as in Fig. 4; its oracle side is the Fig. 4 cell.

Inputs: data/figures/osl_cells.csv, ipdps_cells_f4.csv, longmixed_cells.csv.

    python3 figures/sweep/plot_ipdps_workload.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

from plot_ipdps_frontier import INK, GRID  # also applies the shared rcParams

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/figures"
MILP = "facility_ilp_carbon_sla"
NOCAP = "carbon_aware_a0.0_nocap"
# Own palette, distinct from Figs. 4-6: magenta for Llama, dark cyan for Mixtral.
MODELS = [("llama3_8B", "Llama-3.1-8B", 4000, "#C2185B"),
          ("mixtral_8x7b", r"Mixtral-8$\times$7B", 2000, "#00838F")]
POLICIES = [("carbon_aware_a0.0_nocap", r"G, $\alpha$=0", "v"), ("ilp_carbon_sla", "Std ILP", "D"), ("iterative_ilp_carbon_sla", "Iter ILP", "p"),
            (MILP, "Fac. MILP", "*")]
# (workload, label, rate, colour, line style)
WORKLOADS = [("long", "Long, 50 req/s", 50, "#00838F", "-"),
             ("mixed", "Mixed, 200 req/s", 200, "#004D55", "--")]
# Capacity step drawn per workload in panel (b): long 12-32 by 4, mixed 16-64 by 8.
CAP_STEP = {"long": 4, "mixed": 8}
# Broken x-axis of panel (a): 0-15 at full scale, 15-25 cut, 25-50 compressed.
BREAK_LO, BREAK_HI, GAP, FAR_SCALE = 15.0, 25.0, 1.2, 0.35


def xmap(v):
    return v if v <= BREAK_LO else BREAK_LO + GAP + (v - BREAK_HI) * FAR_SCALE


def style(ax):
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def boxed(leg):
    leg.get_frame().set_edgecolor(INK)
    leg.get_frame().set_linewidth(0.5)


def osl_points():
    """(model, policy) -> (oracle row, predicted row), with saving and violation."""
    osl = pd.read_csv(D / "osl_cells.csv")
    f4 = pd.read_csv(D / "ipdps_cells_f4.csv")
    rr = f4[f4.policy == "round_robin"].set_index(["model", "rate"]).carbon_g
    out = {}
    for m, _, rate, _ in MODELS:
        pred = osl[(osl.model == m) & (osl.rate == rate) & (osl.osl == "predicted")]
        orac = osl[(osl.model == m) & (osl.rate == rate) & (osl.osl == "oracle")]
        if orac.empty:                       # Mixtral: oracle side from the main grid
            orac = f4[(f4.model == m) & (f4.rate == rate)]
        for pol, _, _ in POLICIES:
            # Cap-less greedy's oracle side is its Fig. 4 cell (key carbon_aware_a0.0 there);
            # the carbon_aware_a0.0 rows of osl_cells.csv ran it with a batch capacity.
            if pol == NOCAP:
                a = f4[(f4.model == m) & (f4.rate == rate) & (f4.policy == "carbon_aware_a0.0")]
            else:
                a = orac[orac.policy == pol]
            b = pred[pred.policy == pol]
            if a.empty or b.empty:
                continue
            pts = []
            for r in (a.iloc[0], b.iloc[0]):
                pts.append((100.0 - r.slo_pct, 100.0 * (1.0 - r.carbon_g / rr[(m, rate)])))
            out[(m, pol)] = pts
    return out


def osl_panel(ax):
    for (m, pol), (a, b) in osl_points().items():
        color = next(c for mm, _, _, c in MODELS if mm == m)
        mk = next(k for p, _, k in POLICIES if p == pol)
        ms = 9 if mk == "*" else 6.5
        edge, tip = 0.45 * ms, (0.12 * ms if mk == "*" else 0.45 * ms)
        ax.annotate("", xy=(xmap(b[0]), b[1]), xytext=(xmap(a[0]), a[1]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, shrinkA=edge,
                                    shrinkB=tip, mutation_scale=8), zorder=2)
        ax.plot(xmap(a[0]), a[1], mk, ms=ms, mfc="white", mec=color, mew=1.2, zorder=3)
        ax.plot(xmap(b[0]), b[1], mk, ms=ms, mfc=color, mec=INK, mew=0.4, zorder=3)
    ticks = [0, 5, 10, 15, 30, 40, 55]
    ax.set_xticks([xmap(t) for t in ticks])
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(-1, xmap(59))
    xb = BREAK_LO + GAP / 2
    ax.axvspan(xb - 0.3, xb + 0.3, color="white", zorder=1)
    for dx in (-0.18, 0.18):
        ax.plot([xb + dx - 0.12, xb + dx + 0.12], [-0.03, 0.03], transform=ax.get_xaxis_transform(),
                color=INK, lw=0.8, clip_on=False, zorder=5)
    ax.set_xlabel("SLO violation (%)")
    ax.set_ylabel("Carbon saving vs RR (%)")
    pol_h = [Line2D([], [], ls="", marker=mk, ms=7.5 if mk == "*" else 5.5, mfc="#bdbcb8",
                    mec=INK, mew=0.5, label=lab) for _, lab, mk in POLICIES]
    fill_h = [Line2D([], [], ls="", marker="o", ms=5.5, mfc="white", mec=INK, mew=0.9, label="Oracle OSL"),
              Line2D([], [], ls="", marker="o", ms=5.5, mfc=INK, mec=INK, label="Predicted OSL")]
    model_h = [Patch(facecolor=c, edgecolor="none", label=lab) for _, lab, _, c in MODELS]
    kw = dict(frameon=True, fancybox=False, fontsize=6, handletextpad=0.3, borderpad=0.3,
              labelspacing=0.3, columnspacing=0.6, handlelength=1.2)
    leg = ax.legend(handles=pol_h, ncol=2, loc="lower left", **kw)
    boxed(leg)
    ax.add_artist(leg)
    leg2 = ax.legend(handles=fill_h + model_h, ncol=1, loc="upper right", **kw)
    boxed(leg2)


def longmixed_panel(ax):
    lm = pd.read_csv(D / "longmixed_cells.csv")
    lm["wl"] = lm.tag.str.split("_").str[0]
    lm["cap"] = lm.tag.str.extract(r"cap(\d+)").astype(int)
    rr = lm[lm.policy == "round_robin"].set_index(["wl", "rate"]).carbon_g
    for wl, lab, rate, color, ls in WORKLOADS:
        s = lm[(lm.wl == wl) & (lm.rate == rate) & (lm.policy == MILP)].sort_values("cap")
        s = s[s.cap % CAP_STEP[wl] == 0]     # evenly spaced capacities only
        x = 100.0 - s.slo_pct
        y = 100.0 * (1.0 - s.carbon_g / rr[(wl, rate)])
        ax.plot(x, y, ls=ls, lw=1.2, color=color, marker="*", ms=8, mec=INK, mew=0.4, zorder=3)
        # Smallest and largest capacity labelled in empty space, an arrow to the exact point.
        offs = {"long": ((14, -3), (-44, -11)), "mixed": ((14, -1), (14, -4))}[wl]
        for (xi, yi, c), off in zip(((x.iloc[0], y.iloc[0], s.cap.iloc[0]),
                                     (x.iloc[-1], y.iloc[-1], s.cap.iloc[-1])), offs):
            ax.annotate(f"cap {c}", (xi, yi), xytext=off, textcoords="offset points",
                        fontsize=6, color=INK, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6, shrinkA=1,
                                        shrinkB=3, mutation_scale=6))
    # Hatch the band where the SLO violation stays below 5%.
    ax.axvspan(-1.5, 5, facecolor="none", edgecolor="#d3d2ce", hatch="///", lw=0, zorder=0)
    ax.text(2.0, 0.97, "<5%\nviol.", transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=5.5, color=INK, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.set_xlim(-1.5, None)
    ax.set_xlabel("SLO violation (%)")
    ax.set_ylabel("Fac. MILP carbon saving vs RR (%)")
    handles = [Line2D([], [], color=c, ls=ls, lw=1.2, marker="*", ms=7, mec=INK, mew=0.4, label=lab)
               for _, lab, _, c, ls in WORKLOADS]
    leg = ax.legend(handles=handles, loc="lower right", frameon=True, fancybox=False, fontsize=6,
                    handlelength=2.0, handletextpad=0.3, borderpad=0.3, labelspacing=0.3)
    boxed(leg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "figures/sweep/osl_longmixed.png"))
    ap.add_argument("--width", type=float, default=5.0)
    ap.add_argument("--height", type=float, default=1.9)
    a = ap.parse_args()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(a.width, a.height))
    osl_panel(ax1)
    longmixed_panel(ax2)
    for ax in (ax1, ax2):
        style(ax)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.tick_params(axis="both", which="major", labelsize=6.5)
    ax1.set_title("(a) OSL source (high load)", fontsize=7.5)
    ax2.set_title(r"(b) Prompt length (Mixtral-8$\times$7B)", fontsize=7.5)
    fig.subplots_adjust(wspace=0.38)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
