from __future__ import annotations
import argparse
import json
import os
import sys
from textwrap import fill

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

HERE   = os.path.dirname(os.path.abspath(__file__))
OUTFIG = os.path.join(HERE, "outputs", "figures")
OUTTAB = os.path.join(HERE, "outputs", "tables")
OUTDAT = os.path.join(HERE, "outputs", "data")
for d in [OUTFIG, OUTTAB, OUTDAT]:
    os.makedirs(d, exist_ok=True)

REGIME_SHORT = {
    "neutral_wf":       "Drift",
    "selected_wf":      "Directional\nselection",
    "moran":            "Moran\ndynamics",
    "freq_dep":         "Freq.-dependent\nselection",
    "eco_evol":         "Eco-evolutionary\nfeedback",
    "group_structured": "Group-structured\nselection",
}
REGIME_LONG = {
    "neutral_wf":       "Random genetic drift",
    "selected_wf":      "Directional selection",
    "moran":            "Moran selection dynamics",
    "freq_dep":         "Frequency-dependent selection",
    "eco_evol":         "Eco-evolutionary feedback",
    "group_structured": "Group-structured selection",
}
TARGET_SHORT = {
    "transmission": "Transmission",
    "persistence":  "Persistence",
    "dominance":    "Dominance",
    "ecology":      "Resource",
}
TARGET_LONG = {
    "transmission": "Allele transmission",
    "persistence":  "Lineage persistence",
    "dominance":    "Allele dominance",
    "ecology":      "Resource level",
}
OBS_SHORT = {
    "gene":     "Gene",
    "lineage":  "Lineage",
    "organism": "Organism",
    "org_eco":  "Org.+Eco.",
}
OBS_LONG = {
    "gene":     "Gene composition",
    "lineage":  "Lineage composition",
    "organism": "Organism fitness",
    "org_eco":  "Organism + ecology",
}
IV_SHORT = {
    "gene":     "Gene",
    "lineage":  "Lineage",
    "organism": "Organism",
    "ecology":  "Ecology",
}
IV_LONG = {
    "gene":     "Gene intervention",
    "lineage":  "Lineage intervention",
    "organism": "Organism intervention",
    "ecology":  "Ecology intervention",
}

ALL_REGIMES = ["neutral_wf", "selected_wf", "moran", "freq_dep", "eco_evol", "group_structured"]
SEL_REGIMES = ["selected_wf", "moran", "freq_dep", "eco_evol", "group_structured"]
ALL_TARGETS = ["transmission", "persistence", "dominance", "ecology"]
BASE_OBS = ["gene", "lineage", "organism"]
ECO_OBS  = ["gene", "lineage", "organism", "org_eco"]
ALL_OBS  = ["gene", "lineage", "organism", "org_eco"]
ALL_IVS  = ["gene", "lineage", "organism", "ecology"]

def observers_for_regime(regime):
    return ECO_OBS if regime == "eco_evol" else BASE_OBS

OBS_C = {
    "gene":     "#3A78B5",
    "lineage":  "#48A37C",
    "organism": "#D47A2C",
    "org_eco":  "#C47AA8",
}
IV_C = {
    "gene":     "#3A78B5",
    "lineage":  "#48A37C",
    "organism": "#D47A2C",
    "ecology":  "#C47AA8",
}

sns.set_theme(
    context="paper", style="whitegrid", font="DejaVu Serif",
    rc={
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
        "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "grid.alpha": 0.8,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8, "figure.dpi": 220, "savefig.dpi": 220,
    },
)

def finish_axis(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=3, width=0.8)
    ax.grid(axis="x", color="#D9D9D9", lw=0.6, alpha=0.7)
    ax.grid(axis="y", visible=False)

def panel_label(ax, s):
    ax.text(-0.10, 1.02, s, transform=ax.transAxes,
            fontsize=16, fontweight="bold", ha="left", va="top", clip_on=False)

def smart_xlim(vals_lo, vals_hi, pad_frac=0.10, min_pad=0.02):
    vals_lo = np.asarray(vals_lo, dtype=float); vals_hi = np.asarray(vals_hi, dtype=float)
    vals_lo = vals_lo[np.isfinite(vals_lo)]; vals_hi = vals_hi[np.isfinite(vals_hi)]
    if len(vals_lo) == 0 or len(vals_hi) == 0: return -0.05, 0.10
    lo = np.nanmin(vals_lo); hi = np.nanmax(vals_hi)
    span = hi - lo; pad = max(min_pad, pad_frac * span if span > 0 else min_pad)
    return lo - pad, hi + pad

def ci_point(ax, x, lo, hi, y, color, marker="o", ms=8, lw=2.4):
    if np.isnan(x) or np.isnan(lo) or np.isnan(hi): return
    ax.plot([lo, hi], [y, y], color=color, lw=lw, alpha=0.35, solid_capstyle="round", zorder=2)
    ax.scatter([x], [y], s=ms**2, color=color, edgecolor="white", linewidth=0.8, marker=marker, zorder=3)

def wrap(s, n=32): return fill(s, width=n)

def target_applicable(regime, target):
    return not (target == "ecology" and regime != "eco_evol")

def observer_applicable(regime, obs):
    return not (obs == "org_eco" and regime != "eco_evol")

def intervention_applicable(regime, interv):
    return not (interv == "ecology" and regime != "eco_evol")

# ── Data access ──────────────────────────────────────────────────────────────

def load_results(path: str):
    if not os.path.exists(path):
        print(f"[INFO] {path} not found.", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)

def _pred(results, regime, target, level):
    if results:
        for r in results.get("pred_results", []):
            if r["regime"] == regime and r["target"] == target and r["level"] == level:
                return float(r["score"]), float(r["ci_lo"]), float(r["ci_hi"])
    return np.nan, np.nan, np.nan

def _caus(results, regime, target, interv):
    if results:
        for r in results.get("causal_results", []):
            if r["regime"] == regime and r["target"] == target and r["interv_class"] == interv:
                return float(r["gap"]), float(r["ci_lo"]), float(r["ci_hi"])
    return np.nan, np.nan, np.nan

def all_pred(results, regime, target):
    obs_list = observers_for_regime(regime)
    return {o: _pred(results, regime, target, o) for o in obs_list if observer_applicable(regime, o)}

def all_caus(results, regime, target):
    return {iv: _caus(results, regime, target, iv) for iv in ALL_IVS if intervention_applicable(regime, iv)}

# ── Helper: metric name ───────────────────────────────────────────────────────

# ── Dissociation / non-dissociation cell extractors ──────────────────────────

def get_dissociation_cells(results):
    if results is None:
        return []
    pred_rows    = results.get("pred_results", [])
    caus_rows    = results.get("causal_results", [])
    verdicts     = results.get("verdicts", {})
    pred_winner  = verdicts.get("pred_winner", {})
    causal_winner= verdicts.get("causal_winner", {})
    dissociation = verdicts.get("dissociation", {})

    cells = []
    for rg in ALL_REGIMES:
        for tg in ALL_TARGETS:
            if not target_applicable(rg, tg): continue
            if not dissociation.get(rg, {}).get(tg, False): continue
            pw = pred_winner.get(rg, {}).get(tg, "NONE")
            cw = causal_winner.get(rg, {}).get(tg, "NONE")
            if pw == "NONE" or cw == "NONE": continue
            prow = next((r for r in pred_rows
                         if r["regime"]==rg and r["target"]==tg and r["level"]==pw), None)
            crow = next((r for r in caus_rows
                         if r["regime"]==rg and r["target"]==tg and r["interv_class"]==cw), None)
            if prow is None or crow is None: continue
            cells.append({
                "regime": rg, "target": tg,
                "pred_winner": pw,
                "pred_score": float(prow["score"]),
                "pred_lo":    float(prow["ci_lo"]),
                "pred_hi":    float(prow["ci_hi"]),
                "caus_winner": cw,
                "caus_gap": float(crow["gap"]),
                "caus_lo":  float(crow["ci_lo"]),
                "caus_hi":  float(crow["ci_hi"]),
            })
    return cells


def get_nondiss_cells(results):
    if results is None:
        return []
    pred_rows    = results.get("pred_results", [])
    verdicts     = results.get("verdicts", {})
    pred_winner  = verdicts.get("pred_winner", {})
    dissociation = verdicts.get("dissociation", {})

    cells = []
    for rg in ALL_REGIMES:
        if rg == "neutral_wf": continue
        for tg in ALL_TARGETS:
            if not target_applicable(rg, tg): continue
            if dissociation.get(rg, {}).get(tg, False): continue
            pw = pred_winner.get(rg, {}).get(tg, "NONE")
            if pw == "NONE": continue
            prow = next((r for r in pred_rows
                         if r["regime"]==rg and r["target"]==tg and r["level"]==pw), None)
            if prow is None: continue
            cells.append({
                "regime": rg, "target": tg,
                "pred_winner": pw,
                "pred_score": float(prow["score"]),
                "pred_lo":    float(prow["ci_lo"]),
                "pred_hi":    float(prow["ci_hi"]),
            })
    return cells

# ── Figure 1: Conceptual overview ────────────────────────────────────────────

def fig1_schematic(outpath):
    fig = plt.figure(figsize=(8.8, 5.2))
    gs  = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.48, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); panel_label(ax, "A")
    ax.set_title("Levels of biological description", pad=6)
    levels = [
        ("Gene composition",    OBS_C["gene"],     0.82),
        ("Lineage composition", OBS_C["lineage"],  0.60),
        ("Organism fitness",    OBS_C["organism"], 0.38),
        ("Organism + ecology",  OBS_C["org_eco"],  0.16),
    ]
    for i, (lab, col, y) in enumerate(levels):
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.08, y-0.08), 0.82, 0.12,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=col, edgecolor=col, alpha=0.15, lw=1.6, transform=ax.transAxes))
        ax.text(0.49, y-0.02, lab, transform=ax.transAxes,
                ha="center", va="center", color=col, fontweight="bold", fontsize=9)
        if i < len(levels)-1:
            ax.annotate("", xy=(0.49, y-0.09), xytext=(0.49, y-0.16),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#888888"))
    ax.text(0.49, 0.03, "greater integration downward",
            transform=ax.transAxes, ha="center", fontsize=8, color="#777777", style="italic")

    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); panel_label(ax, "B")
    ax.set_title("Predictive access", pad=6)
    ax.text(0.5, 0.88, r"$\phi(s_{t_0})$", transform=ax.transAxes, ha="center", fontsize=10)
    ax.text(0.5, 0.10, r"future target $y(T)$", transform=ax.transAxes, ha="center", fontsize=10)
    ax.annotate("", xy=(0.5, 0.18), xytext=(0.5, 0.80),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", lw=1.7, color="#444444"))
    ax.text(0.72, 0.50, r"cross-validated $R^2$",
            transform=ax.transAxes, ha="center", fontsize=9, color="#444444")
    for x, y, col in [(0.20, 0.72, OBS_C["gene"]),
                       (0.22, 0.47, OBS_C["organism"]),
                       (0.18, 0.25, OBS_C["org_eco"])]:
        ax.add_patch(mpatches.Circle((x, y), 0.035, transform=ax.transAxes,
                                     facecolor=col, edgecolor="white", lw=0.8))
        ax.annotate("", xy=(0.46, 0.52), xytext=(x+0.035, y),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=col))

    ax = fig.add_subplot(gs[0, 2]); ax.axis("off"); panel_label(ax, "C")
    ax.set_title("Causal access", pad=6)
    ax.text(0.5, 0.88, r"state at $t_0$", transform=ax.transAxes, ha="center", fontsize=10)
    ax.text(0.5, 0.10, r"$\Delta \bar{y} = \bar{y}_{int} - \bar{y}_{ctrl}$",
            transform=ax.transAxes, ha="center", fontsize=10)
    for x, lab, col in zip([0.20,0.50,0.80], ["Gene","Organism","Ecology"],
                             [OBS_C["gene"], OBS_C["organism"], OBS_C["org_eco"]]):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x-0.10, 0.48), 0.20, 0.12,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=col, edgecolor=col, alpha=0.15, lw=1.3, transform=ax.transAxes))
        ax.text(x, 0.54, lab, transform=ax.transAxes,
                ha="center", va="center", color=col, fontweight="bold", fontsize=8.5)
        ax.annotate("", xy=(x, 0.46), xytext=(x, 0.80),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color=col))
        ax.annotate("", xy=(0.5, 0.18), xytext=(x, 0.46),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=0.9, color=col,
                                    connectionstyle="arc3,rad=0.15"))

    ax = fig.add_subplot(gs[1, :]); ax.axis("off"); panel_label(ax, "D")
    ax.set_title("Illustrative dissociation: eco-evolutionary feedback × resource", pad=6)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.18), 0.36, 0.55, boxstyle="round,pad=0.03,rounding_size=0.04",
        transform=ax.transAxes, facecolor=OBS_C["org_eco"], edgecolor=OBS_C["org_eco"],
        alpha=0.12, lw=1.6))
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.59, 0.18), 0.36, 0.55, boxstyle="round,pad=0.03,rounding_size=0.04",
        transform=ax.transAxes, facecolor=OBS_C["gene"], edgecolor=OBS_C["gene"],
        alpha=0.12, lw=1.6))
    ax.text(0.23, 0.62, "Best predictor", transform=ax.transAxes,
            ha="center", fontsize=10, color=OBS_C["org_eco"], fontweight="bold")
    ax.text(0.23, 0.40, "Organism + ecology",
            transform=ax.transAxes, ha="center", fontsize=9, color=OBS_C["org_eco"])
    ax.text(0.77, 0.62, "Strongest causal handle", transform=ax.transAxes,
            ha="center", fontsize=10, color=OBS_C["gene"], fontweight="bold")
    ax.text(0.77, 0.40, "Gene composition",
            transform=ax.transAxes, ha="center", fontsize=9, color=OBS_C["gene"])
    ax.annotate("", xy=(0.57, 0.45), xytext=(0.43, 0.45),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="<|-|>", lw=2.0, color="#666666"))
    ax.text(0.50, 0.54, "dissociation", transform=ax.transAxes,
            ha="center", fontsize=10, color="#666666", style="italic")
    ax.text(0.50, 0.08, "Best forecast and best intervention live at different levels.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#777777")

    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.04); plt.close(fig)
    print(f"  {outpath}")

# ── Figure 2: Flagship ────────────────────────────────────────────────────────

def fig2_flagship(results, outpath):
    rg = "eco_evol"
    tg = "ecology"
    pd = all_pred(results, rg, tg)
    cd = all_caus(results, rg, tg)

    fig, axes = plt.subplots(
        1, 2, figsize=(9.4, 4.0),
        gridspec_kw={"wspace": 0.62}
    )
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.18, top=0.80)

    # -------------------------
    # A. Predictive ranking
    # -------------------------
    ax = axes[0]
    order = observers_for_regime(rg)[::-1]
    ymap = {o: i for i, o in enumerate(order)}

    pred_vals = [(pd[o][0], o) for o in order if o in pd and not np.isnan(pd[o][0])]
    best_pred = max(pred_vals, key=lambda x: x[0])[1] if pred_vals else None

    if best_pred is not None:
        y_best = ymap[best_pred]
        ax.axhspan(y_best - 0.45, y_best + 0.45, color=OBS_C[best_pred], alpha=0.05, zorder=0)

    pred_hi_vals = []
    pred_lo_vals = []
    for o in order:
        y = ymap[o]
        s, lo, hi = pd.get(o, (np.nan, np.nan, np.nan))
        if np.isnan(s):
            continue
        pred_lo_vals.append(lo)
        pred_hi_vals.append(hi)
        ci_point(ax, s, lo, hi, y, OBS_C[o], marker="o", ms=8, lw=2.0)
        ax.text(
            hi + 0.010, y, f"{s:.3f}",
            va="center", ha="left", fontsize=8,
            color=OBS_C[o], fontweight="bold"
        )

    ax.axvline(0, color="#B0B0B0", lw=0.9, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([OBS_LONG[o] for o in order])
    ax.set_xlabel(r"Cross-validated $R^2$ (95\% CI)")
    ax.set_title("Predictive ranking", pad=8, fontsize=11)

    xlo = min(pred_lo_vals + [0.0]) - 0.03
    xhi = max(pred_hi_vals + [0.55]) + 0.06
    ax.set_xlim(xlo, xhi)

    ax.grid(axis="x", alpha=0.35)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=10)

    ax.text(
        -0.14, 1.04, "A",
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        ha="left", va="top"
    )

    # -------------------------
    # B. Causal ranking
    # -------------------------
    ax = axes[1]
    order_iv = ALL_IVS[::-1]
    ymap2 = {iv: i for i, iv in enumerate(order_iv)}

    caus_vals = [(abs(cd[iv][0]), iv) for iv in order_iv if iv in cd and not np.isnan(cd[iv][0])]
    best_caus = max(caus_vals, key=lambda x: x[0])[1] if caus_vals else None

    if best_caus is not None:
        y_best = ymap2[best_caus]
        ax.axhspan(y_best - 0.45, y_best + 0.45, color=IV_C[best_caus], alpha=0.05, zorder=0)

    caus_lo_vals, caus_hi_vals = [], []
    for iv in order_iv:
        y = ymap2[iv]
        g, lo, hi = cd.get(iv, (np.nan, np.nan, np.nan))
        if np.isnan(g):
            continue
        caus_lo_vals.append(lo)
        caus_hi_vals.append(hi)
        ci_point(ax, g, lo, hi, y, IV_C[iv], marker="D", ms=8, lw=2.0)

        if g >= 0:
            txtx, ha = hi + 0.003, "left"
        else:
            txtx, ha = lo - 0.003, "right"

        ax.text(
            txtx, y, f"{g:+.4f}",
            va="center", ha=ha, fontsize=8,
            color=IV_C[iv], fontweight="bold"
        )

    ax.axvline(0, color="#B0B0B0", lw=0.9, ls="--")
    ax.set_yticks(range(len(order_iv)))
    ax.set_yticklabels([IV_LONG[iv] for iv in order_iv])
    ax.set_xlabel(r"Causal gap: intervention - control (95\% CI)")
    ax.set_title("Causal ranking", pad=8, fontsize=11)

    xlo = min(caus_lo_vals + [-0.02]) - 0.010
    xhi = max(caus_hi_vals + [0.03]) + 0.010
    ax.set_xlim(xlo, xhi)

    ax.grid(axis="x", alpha=0.35)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=10)

    ax.text(
        -0.14, 1.04, "B",
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        ha="left", va="top"
    )

    # -------------------------
    # Global title
    # -------------------------
    fig.suptitle(
        "Eco-evolutionary feedback, resource target\n"
        "Eco summary forecasts best, but gene intervention is the reliable causal handle",
        y=0.965, fontsize=11.5, fontweight="semibold"
    )

    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  {outpath}")

# ── Figure 3: Dissociation panel (results-driven) ────────────────────────────

def fig3_dissociation(results, outpath):
    cells = sorted(
        get_dissociation_cells(results),
        key=lambda c: abs(c["caus_gap"]),
        reverse=True
    )

    if not cells:
        fig, ax = plt.subplots(figsize=(7.0, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No confirmed dissociation cells",
                ha="center", va="center", fontsize=12)
        fig.savefig(outpath, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        print(f"  {outpath}")
        return

    labels = [f"{REGIME_LONG[c['regime']]}  ×  {TARGET_SHORT[c['target']]}" for c in cells]
    y = np.arange(len(cells))[::-1]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(9.4, 0.48 * len(cells) + 1.9),
        sharey=True,
        gridspec_kw={"wspace": 0.10, "width_ratios": [1.05, 1.0]}
    )
    fig.subplots_adjust(left=0.34, right=0.98, top=0.84, bottom=0.14)

    # -------------------------
    # A. Predictive side
    # -------------------------
    ax = axes[0]
    pred_los, pred_his = [], []

    for i, c in enumerate(cells):
        yy = y[i]
        col = OBS_C[c["pred_winner"]]
        ci_point(ax, c["pred_score"], c["pred_lo"], c["pred_hi"], yy, col, marker="o", ms=7.5, lw=2.0)
        pred_los.append(c["pred_lo"])
        pred_his.append(c["pred_hi"])
        ax.text(
            c["pred_hi"] + 0.010, yy, f"{c['pred_score']:.3f}",
            va="center", ha="left", fontsize=7.5,
            color=col, fontweight="bold"
        )

    xmin = min(pred_los + [0.0]) - 0.04
    xmax = max(pred_his + [0.85]) + 0.06
    ax.set_xlim(xmin, xmax)
    ax.axvline(0, color="#B0B0B0", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Predictive score ($R^2$ or AUC)")
    ax.set_title("Predictive side", pad=6, fontsize=10.5)
    ax.grid(axis="x", alpha=0.30)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=8)

    ax.text(
        -0.10, 1.02, "A",
        transform=ax.transAxes,
        fontsize=16, fontweight="bold",
        ha="left", va="top"
    )

    # -------------------------
    # B. Causal side
    # -------------------------
    ax = axes[1]
    caus_los, caus_his = [], []

    for i, c in enumerate(cells):
        yy = y[i]
        col = IV_C[c["caus_winner"]]
        ci_point(ax, c["caus_gap"], c["caus_lo"], c["caus_hi"], yy, col, marker="D", ms=7.5, lw=2.0)
        caus_los.append(c["caus_lo"])
        caus_his.append(c["caus_hi"])

        if c["caus_gap"] >= 0:
            txtx, ha = c["caus_hi"] + 0.0035, "left"
        else:
            txtx, ha = c["caus_lo"] - 0.0035, "right"

        ax.text(
            txtx, yy, f"{c['caus_gap']:+.3f}",
            va="center", ha=ha, fontsize=7.5,
            color=col, fontweight="bold"
        )

    xmin = min(caus_los + [-0.01]) - 0.01
    xmax = max(caus_his + [0.12]) + 0.01
    ax.set_xlim(xmin, xmax)
    ax.axvline(0, color="#B0B0B0", lw=0.9, ls="--")
    ax.set_xlabel("Causal winner gap (95% CI)")
    ax.set_title("Causal side", pad=6, fontsize=10.5)
    ax.grid(axis="x", alpha=0.30)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", left=False, labelleft=False)

    ax.text(
        -0.10, 1.02, "B",
        transform=ax.transAxes,
        fontsize=16, fontweight="bold",
        ha="left", va="top"
    )

    # legend outside the plotting area
    handles = [
        mpatches.Patch(color=OBS_C[k], label=OBS_LONG[k])
        for k in sorted(set(c["pred_winner"] for c in cells), key=lambda x: ALL_OBS.index(x))
    ]
    axes[0].legend(
        handles=handles,
        title="Predictive winner",
        loc="lower right",
        frameon=True,
        edgecolor="#DDDDDD",
        framealpha=0.95,
        fontsize=7,
        title_fontsize=8
    )

    fig.suptitle(
        f"Confirmed dissociation cells (n = {len(cells)})\n"
        "Predictive and causal winners differ within fixed targets",
        y=0.95, fontsize=11, fontweight="bold"
    )

    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  {outpath}")

# ── Figure 4: Predictive heatmap ─────────────────────────────────────────────

def fig4_pred_map(results, outpath):
    NA = -2          # structurally not applicable
    NONE = -1        # applicable, but no confirmed winner

    winner_idx = np.full((len(ALL_REGIMES), len(ALL_TARGETS)), NONE, dtype=float)
    score_mat  = np.full((len(ALL_REGIMES), len(ALL_TARGETS)), np.nan)
    obs_index  = {o: i for i, o in enumerate(ALL_OBS)}
    verdicts   = results.get("verdicts", {}) if results else {}
    pred_winner = verdicts.get("pred_winner", {})

    for ri, rg in enumerate(ALL_REGIMES):
        for ti, tg in enumerate(ALL_TARGETS):
            if not target_applicable(rg, tg):
                winner_idx[ri, ti] = NA
                continue

            o_best = pred_winner.get(rg, {}).get(tg, "NONE")
            if o_best == "NONE":
                winner_idx[ri, ti] = NONE
                continue

            s_best, _, _ = _pred(results, rg, tg, o_best)
            if np.isnan(s_best):
                winner_idx[ri, ti] = NONE
                continue

            winner_idx[ri, ti] = obs_index[o_best]
            score_mat[ri, ti] = s_best

    annot = np.empty_like(score_mat, dtype=object)
    for i in range(score_mat.shape[0]):
        for j in range(score_mat.shape[1]):
            if winner_idx[i, j] == NA:
                annot[i, j] = "N/A"
            elif winner_idx[i, j] == NONE:
                annot[i, j] = "--"
            else:
                annot[i, j] = f"{score_mat[i, j]:.2f}"

    cmap = ["#E6E6E6", "#FFFFFF"] + [OBS_C[o] for o in ALL_OBS]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.subplots_adjust(left=0.23, right=0.82, top=0.84, bottom=0.18)

    sns.heatmap(
        winner_idx + 2,   # shifts NA=-2->0, NONE=-1->1, observers->2..5
        ax=ax,
        cmap=sns.color_palette(cmap, as_cmap=True),
        vmin=-0.5,
        vmax=len(cmap)-0.5,
        annot=annot,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        cbar=False,
        square=False,
    )

    ax.set_xticks(np.arange(len(ALL_TARGETS)) + 0.5)
    ax.set_xticklabels([TARGET_SHORT[t] for t in ALL_TARGETS], rotation=15, ha="right")
    ax.set_yticks(np.arange(len(ALL_REGIMES)) + 0.5)
    ax.set_yticklabels([REGIME_LONG[r] for r in ALL_REGIMES], rotation=0)
    ax.set_title(
        "Predictive winner across regimes and targets\n"
        "(numeric = best predictive score; N/A = not applicable; -- = no confirmed winner)",
        pad=10,
    )
    panel_label(ax, "A")

    handles = [
        mpatches.Patch(color="#E6E6E6", label="Not applicable"),
        mpatches.Patch(color="#FFFFFF", label="No confirmed winner"),
    ] + [
        mpatches.Patch(color=OBS_C[o], label=OBS_LONG[o]) for o in ALL_OBS
    ]

    ax.legend(
        handles=handles,
        title="Cell meaning",
        bbox_to_anchor=(1.01, 1.00),
        loc="upper left",
        frameon=False,
        borderaxespad=0.0,
    )

    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  {outpath}")

# ── Figure 5: Causal heatmap ──────────────────────────────────────────────────

def fig5_caus_map(results, outpath):
    NA = -2
    NONE = -1

    winner_idx = np.full((len(SEL_REGIMES), len(ALL_TARGETS)), NONE, dtype=float)
    gap_mat    = np.full((len(SEL_REGIMES), len(ALL_TARGETS)), np.nan)
    iv_index   = {iv: i for i, iv in enumerate(ALL_IVS)}
    verdicts   = results.get("verdicts", {}) if results else {}
    causal_winner = verdicts.get("causal_winner", {})

    for ri, rg in enumerate(SEL_REGIMES):
        for ti, tg in enumerate(ALL_TARGETS):
            if not target_applicable(rg, tg):
                winner_idx[ri, ti] = NA
                continue

            iv_best = causal_winner.get(rg, {}).get(tg, "NONE")
            if iv_best == "NONE":
                winner_idx[ri, ti] = NONE
                continue

            g_best, _, _ = _caus(results, rg, tg, iv_best)
            if np.isnan(g_best):
                winner_idx[ri, ti] = NONE
                continue

            winner_idx[ri, ti] = iv_index[iv_best]
            gap_mat[ri, ti] = g_best

    annot = np.empty_like(gap_mat, dtype=object)
    for i in range(gap_mat.shape[0]):
        for j in range(gap_mat.shape[1]):
            if winner_idx[i, j] == NA:
                annot[i, j] = "N/A"
            elif winner_idx[i, j] == NONE:
                annot[i, j] = "--"
            else:
                annot[i, j] = f"{gap_mat[i, j]:+.3f}"

    cmap = ["#E6E6E6", "#FFFFFF"] + [IV_C[iv] for iv in ALL_IVS]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.subplots_adjust(left=0.23, right=0.82, top=0.84, bottom=0.18)

    sns.heatmap(
        winner_idx + 2,
        ax=ax,
        cmap=sns.color_palette(cmap, as_cmap=True),
        vmin=-0.5,
        vmax=len(cmap)-0.5,
        annot=annot,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        cbar=False,
        square=False,
    )

    ax.set_xticks(np.arange(len(ALL_TARGETS)) + 0.5)
    ax.set_xticklabels([TARGET_SHORT[t] for t in ALL_TARGETS], rotation=15, ha="right")
    ax.set_yticks(np.arange(len(SEL_REGIMES)) + 0.5)
    ax.set_yticklabels([REGIME_LONG[r] for r in SEL_REGIMES], rotation=0)
    panel_label(ax, "A")
    ax.set_title(
        "Causal winner across selection-driven regimes and targets\n"
        "(numeric = winning causal gap; N/A = not applicable; -- = no confirmed winner)",
        pad=10,
    )

    handles = [
        mpatches.Patch(color="#E6E6E6", label="Not applicable"),
        mpatches.Patch(color="#FFFFFF", label="No confirmed winner"),
    ] + [
        mpatches.Patch(color=IV_C[iv], label=IV_LONG[iv]) for iv in ALL_IVS
    ]

    ax.legend(
        handles=handles,
        title="Cell meaning",
        bbox_to_anchor=(1.01, 1.00),
        loc="upper left",
        frameon=False,
        borderaxespad=0.0,
    )

    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  {outpath}")

# ── Figure 6: Neutral control ─────────────────────────────────────────────────

def fig6_neutral(results, outpath):
    fig, axes = plt.subplots(
        1, 2, figsize=(8.8, 3.6),
        gridspec_kw={"wspace": 0.34}
    )
    fig.subplots_adjust(left=0.30, right=0.98, top=0.80, bottom=0.19)

    for ax, target, lbl in zip(axes, ["transmission", "persistence"], ["A", "B"]):
        pd = all_pred(results, "neutral_wf", target)
        order = observers_for_regime("neutral_wf")[::-1]

        los, his = [], []
        for i, o in enumerate(order):
            s, lo, hi = pd.get(o, (np.nan, np.nan, np.nan))
            if not np.isnan(s):
                los.append(lo)
                his.append(hi)
                ci_point(ax, s, lo, hi, i, OBS_C[o], ms=7.5, lw=2.0)

                shown = 0.0 if abs(s) < 5e-4 else s
                ax.text(
                    hi + 0.008, i, f"{shown:.3f}",
                    va="center", ha="left",
                    fontsize=7.5, color=OBS_C[o]
                )

        xmin = min(los + [0.0]) - 0.03
        xmax = max(his + [0.18 if target == "persistence" else 0.38]) + 0.04

        ax.set_xlim(xmin, xmax)
        ax.axvline(0, color="#B0B0B0", lw=0.9, ls="--")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([OBS_LONG[o] for o in order])
        ax.set_xlabel(r"$R^2$ (95% CI)")
        ax.set_title(TARGET_LONG[target], pad=10, fontsize=11)
        ax.grid(axis="x", alpha=0.28)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0, pad=8)

        ax.text(
            -0.10, 1.02, lbl,
            transform=ax.transAxes,
            fontsize=16, fontweight="bold",
            ha="left", va="top"
        )

    fig.suptitle(
        "Random drift control: prediction via temporal autocorrelation, not dissociation",
        y=0.98, fontsize=12, fontweight="bold"
    )
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  {outpath}")

# ── Figure 7: Boundary cells (results-driven) ─────────────────────────────────

def fig7_boundary(results, outpath, max_cells=4):
    cells = sorted(
        get_nondiss_cells(results),
        key=lambda c: c["pred_score"],
        reverse=True
    )[:max_cells]

    if not cells:
        fig, ax = plt.subplots(figsize=(7.0, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No boundary cells to display",
                ha="center", va="center", fontsize=12)
        fig.savefig(outpath, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        print(f"  {outpath}")
        return

    n = len(cells)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(8.2, 1.65 * n + 0.55),
        gridspec_kw={"hspace": 0.70}
    )
    if n == 1:
        axes = [axes]

    fig.subplots_adjust(left=0.23, right=0.98, top=0.90, bottom=0.10)

    for row, (ax, c, lbl) in enumerate(zip(axes, cells, ["A", "B", "C", "D"][:n])):
        pd = all_pred(results, c["regime"], c["target"])
        order = observers_for_regime(c["regime"])[::-1]

        los, his = [], []
        for i, o in enumerate(order):
            s, lo, hi = pd.get(o, (np.nan, np.nan, np.nan))
            if not np.isnan(s):
                los.append(lo)
                his.append(hi)
                ci_point(ax, s, lo, hi, i, OBS_C[o], ms=7.0, lw=2.0)

        # tighter x-limits for dominance-like near-1 scores
        xmin = min(los + [0.0]) - 0.03
        xmax = max(his + [c["pred_score"]]) + 0.05
        ax.set_xlim(xmin, xmax)

        ax.axvline(0, color="#B0B0B0", lw=0.9, ls="--")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([OBS_SHORT[o] for o in order])
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0, pad=6)

        title = f"{REGIME_LONG[c['regime']]}\n× {TARGET_SHORT[c['target']]}"
        ax.set_title(title, fontsize=10.2, pad=10)

        if row == n - 1:
            ax.set_xlabel("Predictive score ($R^2$ or AUC)")
        else:
            ax.set_xlabel("")

        ax.text(
            -0.10, 1.02, lbl,
            transform=ax.transAxes,
            fontsize=15, fontweight="bold",
            ha="left", va="top"
        )

        ax.text(
            0.97, 0.06, "no confirmed dissociation",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=7.5, color="#8E8E8E", style="italic"
        )

    fig.suptitle(
        "Boundary cells: predictive access present but no confirmed dissociation",
        y=0.98, fontsize=12, fontweight="bold"
    )
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  {outpath}")

# ── Table 1 (results-driven) ──────────────────────────────────────────────────

def table1_latex(results, outpath):
    cells = sorted(
        get_dissociation_cells(results),
        key=lambda c: abs(c["caus_gap"]),
        reverse=True,
    )

    def fmt_pred(c):
        metric = "AUC" if c["target"] == "dominance" else r"$R^2$"
        return rf"{metric}: ${c['pred_score']:.3f}$ [{c['pred_lo']:.3f}, {c['pred_hi']:.3f}]"

    def fmt_caus(c):
        return rf"${c['caus_gap']:+.3f}$ [{c['caus_lo']:+.3f}, {c['caus_hi']:+.3f}]"

    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Regime & Target & Predictive winner & Predictive score & Causal winner & Causal gap (95\% CI) \\",
        r"\midrule",
    ]

    for c in cells:
        lines.append(
            f"{REGIME_LONG[c['regime']]} & "
            f"{TARGET_LONG[c['target']]} & "
            f"{OBS_SHORT[c['pred_winner']]} & "
            f"{fmt_pred(c)} & "
            f"{IV_SHORT[c['caus_winner']]} & "
            f"{fmt_caus(c)} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
    ]

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  {outpath}")
    
# ── Macros (results-driven) ───────────────────────────────────────────────────

def write_macros(results, outpath):
    cells = sorted(get_dissociation_cells(results), key=lambda c: abs(c["caus_gap"]), reverse=True)

    def cmd(name, value):
        return rf"\newcommand{{\{name}}}{{{value}}}"

    def get_cell(regime, target):
        for c in cells:
            if c["regime"] == regime and c["target"] == target:
                return c
        return None

    flag = get_cell("eco_evol", "ecology")
    fd   = get_cell("freq_dep", "persistence")
    grp  = get_cell("group_structured", "transmission")

    lines = [
        "% Auto-generated by paper_analysis.py",
        cmd("NcellsDiss", len(cells)),
        cmd("NRegimes", len(ALL_REGIMES)),
        cmd("NTargets", len(ALL_TARGETS)),
        cmd("Npop", "200"),
        cmd("Nalleles", "4"),
        cmd("Tmax", "50"),
        cmd("Tobs", "20"),
        cmd("MutRate", "0.005"),
        cmd("SelCoeff", "0.05"),
        cmd("RidgeAlpha", "1.0"),
        cmd("CVFolds", "5"),
        cmd("FreqDepAlpha", "0.5"),
        cmd("EcoK", "1.0"),
        cmd("EcoRgrowth", "0.3"),
        cmd("NGroups", "10"),
        cmd("GroupSize", "20"),
        cmd("SWithin", "0.05"),
        cmd("SBetween", "0.08"),
        cmd("Nreps", "100000"),
        cmd("Nintervs", "100000"),
        cmd("DeltaGene", "0.10"),
        cmd("DeltaLin", "0.02"),
        cmd("AlleleTol", "0.05"),
    ]

    if flag is not None:
        lines += [
            cmd("FlagPredScore", f"{flag['pred_score']:.3f}"),
            cmd("FlagPredLo",    f"{flag['pred_lo']:.3f}"),
            cmd("FlagPredHi",    f"{flag['pred_hi']:.3f}"),
            cmd("FlagCausGap",   f"{flag['caus_gap']:+.3f}"),
            cmd("FlagCausLo",    f"{flag['caus_lo']:+.3f}"),
            cmd("FlagCausHi",    f"{flag['caus_hi']:+.3f}"),
        ]

    if fd is not None:
        lines += [
            cmd("FdCausGap", f"{fd['caus_gap']:+.3f}"),
        ]

    if grp is not None:
        lines += [
            cmd("GrpPredScore", f"{grp['pred_score']:.3f}"),
            cmd("GrpCausGap",   f"{grp['caus_gap']:+.3f}"),
        ]

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json")
    args = parser.parse_args()

    print(f"Loading: {args.results}")
    results = load_results(args.results)

    print("\nMain-text figures:")
    fig1_schematic(os.path.join(OUTFIG, "fig1_schematic.pdf"))
    fig2_flagship(results, os.path.join(OUTFIG, "fig2_flagship.pdf"))
    fig3_dissociation(results, os.path.join(OUTFIG, "fig3_dissociation.pdf"))
    fig4_pred_map(results, os.path.join(OUTFIG, "fig4_pred_map.pdf"))
    fig5_caus_map(results, os.path.join(OUTFIG, "fig5_caus_map.pdf"))
    fig6_neutral(results, os.path.join(OUTFIG, "fig6_neutral.pdf"))
    fig7_boundary(results, os.path.join(OUTFIG, "fig7_boundary.pdf"))

    print("\nTables and macros:")
    table1_latex(results, os.path.join(OUTTAB, "table1.tex"))
    write_macros(results, os.path.join(OUTDAT, "macros.tex"))

    print("\nDone.")

if __name__ == "__main__":
    main()
