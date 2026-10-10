"""Sensitivity figure (v9 topology): PUE (left) and CI hour (right), two models.

Left: the carbon-SLO plane at high load for the ILP-based schedulers, uniform
PUE 1.1 (hollow) against real per-DC PUE (filled) on the same sites, with an
arrow from one to the other. Colour identifies the model.

Right: the facility MILP's carbon saving vs RR against the CI spread of the hour
(coefficient of variation of CI across the 16 DCs), one line per model and load.
Colour identifies the model, shade the load (light = low, dark = high).

Greedy is left out of both panels: these runs used it with a batch capacity,
unlike the cap-less greedy of the main comparison.

Inputs come from summarize_cells.py (tags: base, puereal, ci<hour>) and
data/figures/ci_hour_cv.csv (per-hour CI spread).

    python3 figures/sweep/plot_ipdps_sensitivity.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import pandas as pd

from plot_ipdps_frontier import INK, GRID  # also applies the shared rcParams

REPO = Path(__file__).resolve().parents[2]
# (model, label, {regime: rate}, light -> dark ramp, line style). The ramps are the
# validated per-model ramps of the TTFT row figure, so colour keeps one meaning.
MODELS = [
    ("llama3_8B", "Llama-3.1-8B", {"Low": 1200, "Moderate": 2400, "High": 4000},
     ("#8fb8e6", "#2a78d6", "#16427a"), "-"),
    ("mixtral_8x7b", r"Mixtral-8$\times$7B", {"Low": 800, "Moderate": 1200, "High": 2000},
     ("#efa27e", "#eb6834", "#9a3a12"), "--"),
]
MILP = "facility_ilp_carbon_sla"
PUE_POLICIES = [("carbon_aware_a0.0_nocap", r"G, $\alpha$=0", "v"),
                ("ilp_carbon_sla", "Std ILP", "D"),
                ("iterative_ilp_carbon_sla", "Iter ILP", "p"),
                (MILP, "Fac. MILP", "*")]


def load(cells, cv):
    df = pd.read_csv(cells)
    rr = df[df.policy == "round_robin"].set_index(["model", "tag", "rate"]).carbon_g.rename("rr_g")
    df = df.join(rr, on=["model", "tag", "rate"])
    df["saving"] = 100.0 * (1.0 - df.carbon_g / df.rr_g)
    df["violation"] = 100.0 - df.slo_pct
    return df.merge(pd.read_csv(cv)[["tag", "hour", "cv_pop"]], on="tag", how="left")


def style(ax):
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def boxed(leg):
    leg.get_frame().set_edgecolor(INK)          # thin square box, as the CV note
    leg.get_frame().set_linewidth(0.5)


# Broken x-axis of the PUE panel: 0-20 at full scale, 20-30 cut, 30-40 compressed.
BREAK_LO, BREAK_HI, GAP, FAR_SCALE = 20.0, 30.0, 1.5, 0.4
HIGHLIGHT = ("mixtral_8x7b", "facility_ilp_carbon_sla")   # pair ringed on the PUE panel


def xmap(v):
    return v if v <= BREAK_LO else BREAK_LO + GAP + (v - BREAK_HI) * FAR_SCALE


def pue_panel(ax, df):
    """Hollow = uniform PUE 1.1, filled = real PUE, arrow from one to the other."""
    for m, _, rates, ramp, _ in MODELS:
        color = ramp[1]
        rate = rates["High"]
        for pol, _, mk in PUE_POLICIES:
            pts = {}
            for tag in ("base", "puereal"):
                r = df[(df.model == m) & (df.tag == tag) & (df.rate == rate) & (df.policy == pol)]
                if not r.empty:
                    pts[tag] = r.iloc[0]
            if len(pts) < 2:
                continue
            a, b = pts["base"], pts["puereal"]
            if (m, pol) == HIGHLIGHT:
                # ring the uniform/real pair, centred between the two points
                cx = (xmap(a.violation) + xmap(b.violation)) / 2
                cy = (a.saving + b.saving) / 2
                ax.add_patch(Ellipse((cx, cy), width=3.8, height=abs(b.saving - a.saving) + 1.6,
                                     fill=False, ec=INK, lw=0.7, ls="-", zorder=4))
            ms = 9 if mk == "*" else 6.5
            edge = 0.45 * ms                     # about the marker radius, in points
            # A star's lower edge sits in the notch between its two legs, closer to its centre.
            tip = 0.12 * ms if mk == "*" else edge
            ax.annotate("", xy=(xmap(b.violation), b.saving), xytext=(xmap(a.violation), a.saving),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                        shrinkA=edge, shrinkB=tip, mutation_scale=8), zorder=2)
            ax.plot(xmap(a.violation), a.saving, mk, ms=ms, mfc="white", mec=color, mew=1.2, zorder=3)
            ax.plot(xmap(b.violation), b.saving, mk, ms=ms, mfc=color, mec=INK, mew=0.4, zorder=3)
    ticks = [0, 5, 10, 15, 20, 40, 55]          # 30 would collide with 20 across the break
    ax.set_xticks([xmap(t) for t in ticks])
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(-1, xmap(57))
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 4.5)                    # headroom so the two top boxes clear the points
    xb = BREAK_LO + GAP / 2                      # break mark on the x-axis, as on Fig. 4
    ax.axvspan(xb - 0.35, xb + 0.35, color="white", zorder=1)
    for dx in (-0.2, 0.2):
        ax.plot([xb + dx - 0.15, xb + dx + 0.15], [-0.03, 0.03], transform=ax.get_xaxis_transform(),
                color=INK, lw=0.8, clip_on=False, zorder=5)
    ax.set_xlabel("SLO violation (%)")
    ax.set_ylabel("Carbon saving vs RR (%)")
    pol_h = [Line2D([], [], ls="", marker=mk, ms=7.5 if mk == "*" else 5.5, mfc="#bdbcb8",
                    mec=INK, mew=0.5, label=lab) for _, lab, mk in PUE_POLICIES]
    fill_h = [Line2D([], [], ls="", marker="o", ms=5.5, mfc="white", mec=INK, mew=0.9,
                     label="Uniform PUE 1.1"),
              Line2D([], [], ls="", marker="o", ms=5.5, mfc=INK, mec=INK, label="Real per-DC PUE")]
    # Models as plain colour swatches, so they do not read as the hollow/filled PUE markers.
    model_h = [Patch(facecolor=ramp[1], edgecolor="none", label=lab)
               for _, lab, _, ramp, _ in MODELS]
    # Two boxes: the policies in two rows in the empty lower left (room for a fourth),
    # the PUE fill and the models one per row in the empty upper right.
    kw = dict(frameon=True, fancybox=False, fontsize=6, handletextpad=0.3, borderpad=0.3,
              labelspacing=0.3, columnspacing=0.6, handlelength=1.2)
    leg = ax.legend(handles=pol_h, ncol=2, loc="lower left", **{**kw, "fontsize": 5.6})
    boxed(leg)
    ax.add_artist(leg)
    # PUE fill in the upper right, models in the upper left, each its own small box.
    small = {**kw, "fontsize": 5.6, "handlelength": 1.0, "handletextpad": 0.25}
    leg2 = ax.legend(handles=fill_h, ncol=1, loc="upper right", borderaxespad=0.2, **small)
    boxed(leg2)
    ax.add_artist(leg2)
    leg3 = ax.legend(handles=model_h, ncol=1, loc="upper left", borderaxespad=0.2, **small)
    boxed(leg3)


def ci_panel(ax, df):
    """One evenly spaced slot per hour, ordered by CV, so hours with near-equal CV stay apart."""
    hours = df[df.tag.str.startswith("ci") | (df.tag == "base")]
    order = hours[["tag", "hour", "cv_pop"]].drop_duplicates().sort_values("cv_pop").reset_index(drop=True)
    slot = {t: i for i, t in enumerate(order.tag)}
    ends = []                                    # (y at the last hour, load name) per line
    for m, _, rates, ramp, ls in MODELS:
        for (regime, rate), color in zip(rates.items(), ramp):
            s = hours[(hours.model == m) & (hours.rate == rate) & (hours.policy == MILP)].copy()
            if s.empty:
                continue
            s["x"] = s.tag.map(slot)
            s = s.sort_values("x")
            ax.plot(s.x, s.saving, ls=ls, lw=1.0, color=color, marker="*", ms=7,
                    mec=INK, mfc=color, mew=0.4, zorder=3)
            if s.x.iloc[-1] == len(order) - 1 and len(s) > 1:
                ends.append([s.saving.iloc[-1], regime])
    for i, r in order.iterrows():
        ax.text(i, 0.99, "CV=" + f"{r.cv_pop:.3f}".lstrip("0"), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=5.5, rotation=90)   # hang from the top of the axes
    ax.axvline(slot["base"], color=INK, lw=0.6, ls=":", alpha=0.35, zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([pd.Timestamp(h).strftime("%-d %b '%y") + ("\n(base)" if t == "base" else "")
                        for t, h in zip(order.tag, order.hour)], fontsize=5.8, rotation=90)
    ax.tick_params(axis="x", length=0)
    # The ordering note sits between the title and the CV labels, boxed.
    ax.text((len(order) - 1) / 2, 1.02, r"lowest $\rightarrow$ highest CV", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=6, bbox=dict(boxstyle="square,pad=0.25", fc="white", ec=INK, lw=0.5))
    ax.set_ylabel("Fac. MILP carbon saving vs RR (%)")
    ax.set_ylim(0, 85)
    ax.set_xlim(-1.4, len(order) - 0.5)
    # No legend: the models share panel (a)'s colours, and each load is named once at the
    # left, with a bracket spanning both models' first points (the loads do not overlap there).
    first = order.tag.iloc[0]
    for regime in ("Low", "Moderate", "High"):
        ys = []
        for m, _, rates, _, _ in MODELS:
            r = hours[(hours.model == m) & (hours.tag == first) & (hours.rate == rates[regime])
                      & (hours.policy == MILP)]
            if not r.empty:
                ys.append(r.iloc[0].saving)
        if not ys:
            continue
        lo, hi = min(ys), max(ys)
        xb = -0.32
        ax.plot([xb, xb], [lo, hi], color=INK, lw=0.6, clip_on=False, zorder=2)
        for y in (lo, hi):
            ax.plot([xb, xb + 0.08], [y, y], color=INK, lw=0.6, clip_on=False, zorder=2)
        ax.text(xb - 0.06, (lo + hi) / 2, {"Moderate": "Mod."}.get(regime, regime), ha="right",
                va="center", fontsize=6, color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=str(REPO / "data/figures/ipdps_cells_sens.csv"))
    ap.add_argument("--cv", default=str(REPO / "data/figures/ci_hour_cv.csv"))
    ap.add_argument("--out", default=str(REPO / "figures/sweep/pue_ci_sensitivity.png"))
    ap.add_argument("--width", type=float, default=5.0)
    ap.add_argument("--height", type=float, default=1.8)
    a = ap.parse_args()
    df = load(a.cells, a.cv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(a.width, a.height))
    pue_panel(ax1, df)
    ci_panel(ax2, df)
    for ax in (ax1, ax2):
        style(ax)
    ax1.set_title("(a) PUE sensitivity (high load)", pad=16, fontsize=9)
    ax2.set_title("(b) CI sensitivity", pad=16, fontsize=9)
    fig.subplots_adjust(wspace=0.35)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
