"""IPDPS frontier figures: SLO violation and TTFT tails against carbon saving.

Reads the per-cell CSV written by summarize_cells.py and draws, for the moderate
and high load regime of each model (the classification in the
paper's setup section):

  ipdps_slo_vs_carbon.png      x = SLO violation (%), y = carbon saving vs RR (%)
  ipdps_ttft_vs_carbon.png     x = TTFT p50 to p99 (s), y = carbon saving vs RR (%)

Color identifies the model, marker the policy. Points are not connected.

    python3 figures/sweep/plot_ipdps_frontier.py [--csv data/figures/ipdps_cells.csv]
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

# Validated categorical slots (dataviz reference palette, light surface):
# blue / aqua / orange pass lightness, chroma, CVD (worst dE 9.2) and
# normal-vision checks. Aqua is below 3:1 against the surface, so every
# marker carries a dark edge and the legend names each model.
MODELS = [
    ("llama3_8B", 1, "Llama-3.1-8B", "#2a78d6"),
    ("qwen2.5_7B", 1, "Qwen2.5-7B", "#1baf7a"),
    ("mixtral_8x7b", 4, "Mixtral-8x7B", "#eb6834"),
]
REGIMES = [("moderate", "Moderate load"), ("high", "High load")]
# Low load is left to the text: every carbon-aware point sits below 0.5% violation there.
# Representative rate per regime, from the setup's occupancy classification.
RATES = {
    ("llama3_8B", 1): {"low": 1200, "moderate": 2400, "high": 4000},
    ("qwen2.5_7B", 1): {"low": 1200, "moderate": 2400, "high": 4000},
    ("mixtral_8x7b", 4): {"low": 800, "moderate": 1200, "high": 2000},
}
POLICIES = [  # (dir name, label, marker)
    ("round_robin", "RR", "x"),
    ("least_loaded", "LL", "+"),
    ("carbon_aware_a0.9", r"G, $\alpha$=0.9", "^"),
    ("carbon_aware_a0.5", r"G, $\alpha$=0.5", "s"),
    ("carbon_aware_a0.0", "G, $\\alpha$=0\n(MOER)", "v"),
    ("ilp_carbon_sla", "Std ILP", "D"),
    ("iterative_ilp_carbon_sla", "Iter ILP", "p"),   # pentagon: "P" read as LL's "+"
    ("facility_ilp_carbon_sla", "Fac. MILP", "*"),
    ("greenllm", "Green-LLM", "<"),
    ("moer", "MOER", "X"),
]
UNFILLED = {"x", "+"}
# x-axis limit of the SLO-violation scatter per regime. Points beyond it (cap-less
# greedy on Mixtral) sit on the right edge with their value printed beside them.
XMAX = {"moderate": 8.0, "high": 32.0}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "text.color": INK, "legend.labelcolor": INK,
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})


def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df.osl == "oracle"].copy()
    rr = (df[df.policy == "round_robin"]
          .set_index(["model", "tp", "rate"]).carbon_g.rename("rr_g"))
    df = df.join(rr, on=["model", "tp", "rate"])
    df["saving"] = 100.0 * (1.0 - df.carbon_g / df.rr_g)
    df["violation"] = 100.0 - df.slo_pct
    return df


def cell(df, model, tp, rate, policy):
    r = df[(df.model == model) & (df.tp == tp) & (df.rate == rate) & (df.policy == policy)]
    return None if r.empty else r.iloc[0]


def draw_points(ax, df, model, tp, rate, color, ycol, filled=True, ms=4.2, zo=3,
                span_from=None, xmax=None):
    """Every policy as an independent marker, no connecting lines."""
    for pol, _, mk in POLICIES:
        r = cell(df, model, tp, rate, pol)
        if r is None:
            continue
        if xmax is not None and round(r[ycol]) in xmax and r[ycol] > min(xmax.values()) / 1.17:
            # off-scale: drawn past the axis break, at the slot of its own value
            r = r.copy()
            r[ycol] = xmax[round(r[ycol])]
        if span_from is not None:
            # p90 -> p99 as one horizontal interval at the policy's saving;
            # the marker sits at p99.
            ax.plot([r[span_from], r[ycol]], [r.saving, r.saving], color=color,
                    lw=1.1, alpha=0.8, solid_capstyle="butt", zorder=zo - 1)
            # short vertical ticks at p50, p90 and p99
            ticks = [r[span_from], r[ycol]]
            if "ttft_p90_s" in r.index:
                ticks.insert(1, r["ttft_p90_s"])
            ax.plot(ticks, [r.saving] * len(ticks), ls="", marker="|", ms=5,
                    mew=1.1, color=color, zorder=zo - 1)
        if mk in UNFILLED:
            ax.plot(r[ycol], r.saving, mk, ms=ms + 1, mew=1.4, color=color, zorder=zo + 1)
        else:
            ax.plot(r[ycol], r.saving, mk, ms=ms + (1.5 if mk == "*" else 0),
                    mfc=color if filled else "white", mec=INK if filled else color,
                    mew=0.5 if filled else 0.9, alpha=0.9, zorder=zo)


def frame(ax, title, rates_note, xlabel, first=False):
    ax.set_title(title, color=INK, pad=4)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlabel)
    if first:
        ax.set_ylabel("Carbon saving vs RR (%)")


def rates_note(present, regime):
    parts = [f"{lab.split('-')[0]} {RATES[(m, tp)][regime]}"
             for m, tp, lab, _ in MODELS if (m, tp) in present]
    return "req/s: " + ", ".join(parts)


def legends(fig, present):
    model_h = [Line2D([], [], ls="", color=c, marker="o", ms=5, mec=INK, mew=0.5,
                      label=f"{lab} (TP={tp})")
               for m, tp, lab, c in MODELS if (m, tp) in present]
    pol_h = []
    for pol, lab, mk in POLICIES:
        if mk in UNFILLED:
            pol_h.append(Line2D([], [], ls="", marker=mk, ms=5, mew=1.3, color=INK2, label=lab))
        else:
            pol_h.append(Line2D([], [], ls="", marker=mk, ms=4.5 + (1.5 if mk == "*" else 0),
                                mfc="#bdbcb8", mec=INK, mew=0.5, label=lab))
    return model_h, pol_h


def figure(df, present, used_policies, ycol_pairs, ylabel, out, ylog=False, width=7.16,
           height=1.8, model_loc="lower right", hide=(), model_ax=0):
    present = {k for k in present if k[0] not in hide}
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(width, height), constrained_layout=False)
    for k, (regime, title) in enumerate(REGIMES):
        ax = axes[k]
        slots = None
        if len(ycol_pairs) == 1 and ycol_pairs[0][0] == "violation":
            # Values beyond XMAX get one slot each past a break in the x-axis,
            # labelled with their own value, in increasing order.
            lim = XMAX[regime]
            far = sorted({round(v) for m, tp, _, _ in MODELS if (m, tp) in present
                          for v in df[(df.model == m) & (df.tp == tp)
                                      & (df.rate == RATES[(m, tp)][regime])].violation
                          if v > lim})
            slots = {v: lim * (1.17 + 0.17 * i) for i, v in enumerate(far)}
        for m, tp, lab, c in MODELS:
            if (m, tp) not in present:
                continue
            rate = RATES[(m, tp)][regime]
            if len(ycol_pairs) == 2:
                # interval mode: marker at the first column, line back to the second
                draw_points(ax, df, m, tp, rate, c, ycol_pairs[0][0],
                            span_from=ycol_pairs[1][0])
            else:
                for ycol, filled in ycol_pairs:
                    draw_points(ax, df, m, tp, rate, c, ycol, filled=filled,
                                xmax=slots if ycol == "violation" else None)
        frame(ax, title, rates_note(present, regime), ylabel, first=(k == 0))
        if ylog:
            ax.set_xscale("log")
        if slots is not None:
            lim = XMAX[regime]
            right = max(slots.values(), default=lim) + 0.05 * lim
            ax.set_xlim(-0.03 * lim, right)
            ticks = [t for t in ax.get_xticks() if 0 <= t <= lim]
            ax.set_xticks(ticks + list(slots.values()))
            ax.set_xticklabels([f"{t:g}" for t in ticks] + [str(v) for v in slots])
            if slots:
                # break mark on the x-axis between the scale and the far slots
                xb = lim * 1.075
                for dx in (-0.012, 0.012):
                    ax.plot([xb + dx * lim - 0.01 * lim, xb + dx * lim + 0.01 * lim],
                            [-0.03, 0.03], transform=ax.get_xaxis_transform(),
                            color=INK2, lw=0.8, clip_on=False, zorder=5)
                ax.axvspan(lim * 1.06, lim * 1.09, color="white", zorder=1)
    model_h, pol_h = legends(fig, present)
    pol_h = [h for h, (pol, _, _) in zip(pol_h, POLICIES) if pol in used_policies]
    extra = []
    if len(ycol_pairs) == 2:
        extra = [Line2D([], [], color=INK2, lw=1.1, marker="|", ms=5, mew=1.1,
                        label="p50, p90, p99")]
    # Models (and the p90/p99 key) inside the left panel, in its empty lower
    # right; policies as one tight column just right of the last panel.
    if model_loc == "top":
        pass  # placed after subplots_adjust, as one row above the panels
    else:
        loc, anchor = (model_loc, None) if isinstance(model_loc, str) else model_loc
        axes[model_ax].legend(handles=model_h + extra, loc=loc, bbox_to_anchor=anchor,
                              frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=6,
                              handletextpad=0.3, borderpad=0.4, labelspacing=0.3)
    right = 1 - 1.05 / width
    # Margins in inches, so a shorter figure shrinks the plot area, not the text.
    fig.subplots_adjust(left=0.5 / width, right=right, top=1 - 0.2 / height,
                        bottom=0.45 / height, wspace=0.18 * 7.16 / width)
    if model_loc == "top":
        x0 = axes[0].get_position().x0
        x1 = axes[-1].get_position().x1
        fig.legend(handles=model_h + extra, loc="lower center", ncol=len(model_h + extra),
                   frameon=False, handletextpad=0.3, columnspacing=1.2,
                   bbox_to_anchor=((x0 + x1) / 2, 1.0))
    fig.legend(handles=pol_h, loc="center left", ncol=1, frameon=False, fontsize=6,
               handletextpad=0.25, labelspacing=0.5, borderaxespad=0.0,
               bbox_to_anchor=(right + 0.008, 0.55))
    # Trim the outer margin so the policy column ends at the figure edge.
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out + ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "data/figures/ipdps_cells_f4.csv"))
    ap.add_argument("--outdir", default=str(REPO / "figures/sweep"))
    ap.add_argument("--width", type=float, default=5.0, help="figure width in inches")
    ap.add_argument("--height", type=float, default=1.8, help="figure height in inches")
    ap.add_argument("--prefix", default="ipdps", help="output file prefix")
    a = ap.parse_args()
    df = load(a.csv)
    present = {(m, tp) for m, tp in df[["model", "tp"]].drop_duplicates().itertuples(index=False)}
    present &= set(RATES)
    used = set(df.policy)
    df["ttft_p50_s"] = df.ttft_p50 / 1000.0
    df["ttft_p90_s"] = df.ttft_p90 / 1000.0
    df["ttft_p99_s"] = df.ttft_p99 / 1000.0
    figure(df, present, used, [("violation", True)], "SLO violation (%)",
           os.path.join(a.outdir, f"{a.prefix}_slo_vs_carbon"), width=a.width, height=a.height,
           model_loc="lower right", model_ax=-1)
    figure(df, present, used, [("ttft_p99_s", True), ("ttft_p50_s", False)], "TTFT (s)",
           os.path.join(a.outdir, f"{a.prefix}_ttft_vs_carbon"), width=a.width, height=a.height,
           model_loc=("center left", (0.0, 0.56)),
           # Qwen shares Llama's trace and SLOs and tracks its TTFT tails;
           # it is left out here to keep the intervals readable.
           hide=("qwen2.5_7B",))


if __name__ == "__main__":
    main()


# Colour by policy family on the row figure, as shades of Llama's own blue so
# the row figure and the scatter agree on what colour means (the model).
# Light -> dark = carbon-blind baselines -> greedy -> ILP family; the ramp passes
# the validator's ordinal checks (one hue, monotone lightness, visible steps).
BASELINE_C, GREEDY_C, ILP_C = "#8fb8e6", "#2a78d6", "#16427a"
FAMILY_COLOR = {"round_robin": BASELINE_C, "least_loaded": BASELINE_C,
                "carbon_aware_a0.9": GREEDY_C, "carbon_aware_a0.5": GREEDY_C,
                "carbon_aware_a0.0": GREEDY_C, "ilp_carbon_sla": ILP_C,
                "iterative_ilp_carbon_sla": ILP_C, "facility_ilp_carbon_sla": ILP_C}


# Per-model light -> dark ramps for the row figure (baselines -> greedy -> ILP
# family), each in its model's hue from the scatter so colour keeps one meaning.
# Both ramps pass the validator's ordinal checks.
RAMPS = {("llama3_8B", 1): (BASELINE_C, GREEDY_C, ILP_C),
         ("qwen2.5_7B", 1): ("#8fd9bd", "#1baf7a", "#0d6b4a"),
         ("mixtral_8x7b", 4): ("#efa27e", "#eb6834", "#9a3a12")}
_FAMILY = {"round_robin": 0, "least_loaded": 0, "carbon_aware_a0.9": 1,
           "carbon_aware_a0.5": 1, "carbon_aware_a0.0": 1, "ilp_carbon_sla": 2,
           "iterative_ilp_carbon_sla": 2, "facility_ilp_carbon_sla": 2, "greenllm": 2}


def ttft_rows(df, panels=((("llama3_8B", 1), "high"), (("qwen2.5_7B", 1), "high"),
                          (("mixtral_8x7b", 4), "high")),
              out=None, width=5.0, height=1.6):
    """TTFT tails as one row per policy: p50-p99 interval, ticks at p50/p90,
    the policy marker at p99, and the carbon saving printed at the row's end.
    One panel per (model, regime); rows never collide."""
    pols = [(p, lab, mk) for p, lab, mk in POLICIES if p in _FAMILY and (df.policy == p).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height), sharey=True, squeeze=False)
    for k, (model, regime) in enumerate(panels):
        ax = axes[0][k]
        m, tp = model
        lab_m = next(lab for mm, t, lab, _ in MODELS if (mm, t) == model)
        ramp = RAMPS[model]
        rate = RATES[model][regime]
        xmax = 0.0
        for row, (pol, lab, mk) in enumerate(pols):
            r = cell(df, m, tp, rate, pol)
            if r is None:
                continue
            y = len(pols) - 1 - row
            color = ramp[_FAMILY[pol]]
            p50, p90, p99 = r.ttft_p50 / 1e3, r.ttft_p90 / 1e3, r.ttft_p99 / 1e3
            xmax = max(xmax, p99)
            ax.plot([p50, p99], [y, y], color=color, lw=1.1, alpha=0.9,
                    solid_capstyle="butt", zorder=2)
            ax.plot([p50, p90], [y, y], ls="", marker="|", ms=5.5, mew=0.7,
                    color=color, zorder=3)
            if mk in UNFILLED:
                ax.plot(p99, y, mk, ms=5, mew=1.4, color=color, zorder=4)
            else:
                ax.plot(p99, y, mk, ms=4.6 + (1.5 if mk == "*" else 0), mfc=color,
                        mec=INK, mew=0.5, zorder=4)
            ax.text(p99, y, f"  {r.saving:.0f}%", va="center", ha="left",
                    fontsize=6, color=INK)
        ax.set_xlim(0, xmax * 1.28)
        ax.xaxis.set_major_locator(MultipleLocator(2))   # same 2 s step on every panel
        ax.set_ylim(-0.5, len(pols) - 0.5)
        for yy in range(len(pols) - 1):   # faint separators between rows
            ax.axhline(yy + 0.5, color=INK, alpha=0.08, lw=0.6, zorder=1)
        ax.set_title(f"{lab_m} (TP={tp})", color=INK, pad=4)
        ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("TTFT (s)")
        ax.set_yticks(range(len(pols)))
        # A two-line label ("G, alpha=0" over "(MOER)") does not fit one row: the
        # first line moves up a little and the second is set small beneath it.
        labs = [lab for _, lab, _ in pols][::-1]
        ax.set_yticklabels([lab.split("\n")[0] for lab in labs])
        ax.tick_params(axis="y", length=0)
        if k == 0:
            for y, lab in enumerate(labs):
                if "\n" in lab:
                    ax.get_yticklabels()[y].set_visible(False)
                    main, small = lab.split("\n")
                    # same line: the small part right-aligned at the tick, the main
                    # part ending where the small part begins
                    t = ax.annotate(small, xy=(0, y), xycoords=("axes fraction", "data"),
                                    xytext=(-3.5, 0), textcoords="offset points",
                                    ha="right", va="center", fontsize=5, color=INK,
                                    annotation_clip=False)
                    w = t.get_window_extent(fig.canvas.get_renderer()).width * 72 / fig.dpi
                    ax.annotate(main + " ", xy=(0, y), xycoords=("axes fraction", "data"),
                                xytext=(-3.5 - w, 0), textcoords="offset points",
                                ha="right", va="center", fontsize=6.5, color=INK,
                                annotation_clip=False)
    fig.subplots_adjust(left=0.2, right=0.99, top=1 - 0.2 / height,
                        bottom=0.5 / height, wspace=0.08)
    out = out or str(REPO / "figures/sweep/ipdps_ttft_rows.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__" and os.environ.get("TTFT_ROWS"):
    _df = load(str(REPO / "data/figures/ipdps_cells_f4.csv"))
    ttft_rows(_df)
