#!/usr/bin/env python3
from __future__ import annotations

"""
nbody_paper.py — rebuilt figure/tables driver for the 3D N-body paper

This rewrite is deliberately conservative:
- it uses the exact stress-battery outputs/observable names from nbody_stress.py
- it avoids silent blank figures by validating data before plotting
- it keeps figure filenames stable for paper.tex
- it makes Fig. 12 use a genuine local velocity-dispersion construction
  consistent with the stress observable class
"""

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from nbody_3d import (
    _HAS_NUMBA,
    _worker_init,
    integrate_leapfrog,
)
from nbody_stress import (
    COARSE_OBS,
    FINE_OBS,
    StressConfig,
    analyse,
    get_initial_conditions,
    get_simconfig,
    min_image,
    run_stress,
)

# -----------------------------------------------------------------------------
# Output layout
# -----------------------------------------------------------------------------

DATA_DIR = os.path.join("outputs", "data")
TABLE_DIR = os.path.join("outputs", "tables")
FIG_DIR = os.path.join("outputs", "figures")

# -----------------------------------------------------------------------------
# Battery / paper constants
# -----------------------------------------------------------------------------

PAPER_N = [256, 512, 1024, 2048]
PAPER_EPS = [0.02, 0.05, 0.10]
PAPER_K = 16
PAPER_MODELS = ["direct_isolated", "pm_periodic"]
PAPER_INITS = ["bimodal3d", "hernquist3d", "plummer3d", "cold_clumpy3d"]
PAPER_STEPS = 600
PAPER_REPS = 30
H_EARLY = 100
H_MID = 300

SHOWCASE_N = 512
SHOWCASE_SEED = 2000
PRIMARY_TARGET = "ΔC8-early"

IC_ORDER = ["bimodal3d", "hernquist3d", "plummer3d", "cold_clumpy3d"]
IC_LABELS = {
    "bimodal3d": "bimodal",
    "hernquist3d": "Hernquist",
    "plummer3d": "Plummer",
    "cold_clumpy3d": "cold-clumpy",
}
IC_COLORS = {
    "bimodal3d": "#1b7837",
    "hernquist3d": "#762a83",
    "plummer3d": "#2166ac",
    "cold_clumpy3d": "#d6604d",
}
MODEL_LABELS = {
    "direct_isolated": "direct-isolated",
    "pm_periodic": "PM-periodic",
}
MODEL_COLORS = {
    "direct_isolated": "#2166ac",
    "pm_periodic": "#8c510a",
}
PRED_COLORS = {
    "CoarseG8": "#2166ac",
    "kNN-all": "#d6604d",
    "ClosePairs": "#b2182b",
    "VelDisp": "#1a9641",
    "FoF-groups": "#762a83",
}
EPS_LS = {0.02: "-", 0.05: "--", 0.10: ":"}
EPS_MK = {0.02: "o", 0.05: "s", 0.10: "^"}

STYLE = {
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in [DATA_DIR, TABLE_DIR, FIG_DIR]:
        os.makedirs(d, exist_ok=True)


def savefig(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def draw_missing(ax: plt.Axes, title: Optional[str] = None, text: str = "Data unavailable") -> None:
    ax.clear()
    if title:
        ax.set_title(title)
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_alpha(0.3)


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def fmt_r(v: Optional[float], nd: int = 3) -> str:
    return "---" if v is None else f"{v:+.{nd}f}"


def fmt_ci(lo: Optional[float], hi: Optional[float], nd: int = 2) -> str:
    if lo is None or hi is None:
        return "[---, ---]"
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _json_default(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def load_csv_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                if v == "":
                    parsed[k] = None
                    continue
                try:
                    if k in {"model", "init", "status", "message"}:
                        parsed[k] = v
                    elif "." in v or "e" in v.lower():
                        parsed[k] = float(v)
                    else:
                        parsed[k] = int(v)
                except Exception:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def write_csv_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write.")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_configs(reps: int = PAPER_REPS, steps: int = PAPER_STEPS) -> List[StressConfig]:
    seeds = [2000 + i for i in range(reps)]
    return [
        StressConfig(
            model=model,
            init=init,
            seed=seed,
            n=n,
            steps=steps,
            eps=eps,
            k_fine=PAPER_K,
            h_early=H_EARLY,
            h_mid=H_MID,
        )
        for n, eps, model, init, seed in product(PAPER_N, PAPER_EPS, PAPER_MODELS, PAPER_INITS, seeds)
    ]


def run_battery(workers: int, configs: List[StressConfig], use_numba: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total = len(configs)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(use_numba,),
    ) as ex:
        futs = {ex.submit(run_stress, cfg, use_numba): cfg for cfg in configs}
        with tqdm(total=total, unit="run", ncols=100) as pbar:
            for fut in as_completed(futs):
                rows.append(fut.result())
                pbar.update(1)
    rows.sort(key=lambda r: (r["n"], r["eps"], r["model"], r["init"], r["seed"]))
    return rows


def make_key(model: str, init: str, n: int, eps: float, k: int = PAPER_K) -> str:
    return f"N={n}|eps={eps}|k={k}|model={model}|init={init}"


def get_cell(analysis: Dict[str, Dict[str, Any]], model: str, init: str, n: int, eps: float) -> Dict[str, Any]:
    return analysis.get(make_key(model, init, n, eps), {})


def get_metric(cell: Dict[str, Any], pred: str, tgt: str = PRIMARY_TARGET) -> Optional[float]:
    return safe_float(cell.get(f"r_{pred}_{tgt}"))


def get_ci(cell: Dict[str, Any], pred: str, tgt: str = PRIMARY_TARGET) -> Tuple[Optional[float], Optional[float]]:
    return safe_float(cell.get(f"ci_lo_{pred}_{tgt}")), safe_float(cell.get(f"ci_hi_{pred}_{tgt}"))


def best_fine_name(cell: Dict[str, Any]) -> Optional[str]:
    vals = []
    for _, pred_name in FINE_OBS:
        v = get_metric(cell, pred_name)
        if v is not None:
            vals.append((abs(v), pred_name))
    return max(vals)[1] if vals else None


def best_coarse_name(cell: Dict[str, Any]) -> Optional[str]:
    vals = []
    for _, pred_name in COARSE_OBS:
        v = get_metric(cell, pred_name)
        if v is not None:
            vals.append((abs(v), pred_name))
    return max(vals)[1] if vals else None


def ok_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("status") == "ok"]


def filter_rows(
    rows: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    init: Optional[str] = None,
    n: Optional[int] = None,
    eps: Optional[float] = None,
) -> List[Dict[str, Any]]:
    out = ok_rows(rows)
    if model is not None:
        out = [r for r in out if r.get("model") == model]
    if init is not None:
        out = [r for r in out if r.get("init") == init]
    if n is not None:
        out = [r for r in out if int(r.get("n")) == int(n)]
    if eps is not None:
        out = [r for r in out if abs(float(r.get("eps")) - float(eps)) < 1e-12]
    return out


def projected_density_image(pos: np.ndarray, box_size: float = 2.0, grid: int = 128, periodic: bool = False) -> np.ndarray:
    if pos is None or len(pos) == 0:
        return np.zeros((grid, grid))
    x = pos[:, 0]
    y = pos[:, 1]
    if periodic:
        x = np.mod(x, box_size)
        y = np.mod(y, box_size)
        mask = np.ones(len(x), dtype=bool)
    else:
        mask = (x >= 0.0) & (x < box_size) & (y >= 0.0) & (y < box_size)
    img, _, _ = np.histogram2d(
        x[mask],
        y[mask],
        bins=grid,
        range=[[0.0, box_size], [0.0, box_size]],
    )
    return np.log10(img + 0.5)


def _run_showcase_sim(init: str, seed: int, n: int, eps: float, steps: int) -> Dict[Any, np.ndarray]:
    cfg_s = StressConfig(
        model="direct_isolated",
        init=init,
        seed=seed,
        n=n,
        steps=steps,
        eps=eps,
        k_fine=PAPER_K,
        h_early=H_EARLY,
        h_mid=H_MID,
    )
    sc = get_simconfig(cfg_s)
    pos0, vel0 = get_initial_conditions(cfg_s)
    mass = 1.0 / n
    snap_steps = sorted({0, H_EARLY, H_MID, steps})
    snaps = integrate_leapfrog(pos0, vel0, mass, sc, snap_steps, False)
    out = {s: snaps[s][0] for s in snap_steps if s in snaps}
    out["vel0"] = vel0
    return out


def compute_local_veldisp_per_particle(
    pos: np.ndarray,
    vel: np.ndarray,
    k: int,
    periodic: bool = False,
    box_size: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if pos is None or vel is None or len(pos) < 2:
        return np.empty((0, 3)), np.array([])
    if periodic:
        inside = np.ones(len(pos), dtype=bool)
    else:
        inside = np.all((pos >= 0.0) & (pos < box_size), axis=1)
    pos_in = pos[inside]
    vel_in = vel[inside]
    if len(pos_in) < 2:
        return np.empty((0, 3)), np.array([])
    k_eff = min(k, len(pos_in) - 1)
    dx = pos_in[:, None, :] - pos_in[None, :, :]
    if periodic:
        dx = min_image(dx, box_size)
    r2 = np.sum(dx * dx, axis=-1)
    np.fill_diagonal(r2, np.inf)
    nn_idx = np.argpartition(r2, kth=k_eff - 1, axis=1)[:, :k_eff]
    v_nn = vel_in[nn_idx]
    v_mean = np.mean(v_nn, axis=1, keepdims=True)
    local_std = np.sqrt(np.mean(np.sum((v_nn - v_mean) ** 2, axis=-1), axis=1))
    return pos_in, local_std

# -----------------------------------------------------------------------------
# Macros and tables
# -----------------------------------------------------------------------------

def write_macros(analysis: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]], path: str) -> None:
    bim = get_cell(analysis, "direct_isolated", "bimodal3d", 1024, 0.05)
    her_small = get_cell(analysis, "direct_isolated", "hernquist3d", 1024, 0.02)
    her_big = get_cell(analysis, "direct_isolated", "hernquist3d", 1024, 0.10)
    plu_small = get_cell(analysis, "direct_isolated", "plummer3d", 1024, 0.02)
    drift_vals = [safe_float(r.get("energy_rel_drift")) for r in filter_rows(rows, model="direct_isolated")]
    drift_vals = [v for v in drift_vals if v is not None]
    drift_median = float(np.median(drift_vals)) if drift_vals else 0.0
    drift_max = float(np.max(drift_vals)) if drift_vals else 0.0

    br = get_metric(bim, "CoarseG8")
    blo, bhi = get_ci(bim, "CoarseG8")
    hv = get_metric(her_small, "VelDisp")
    hvlo, hvhi = get_ci(her_small, "VelDisp")
    pv = get_metric(plu_small, "VelDisp")
    pvlo, pvhi = get_ci(plu_small, "VelDisp")

    mapping = {
        "TotalRuns": str(len(rows)),
        "NCells": str(len(analysis)),
        "NReplicates": str(PAPER_REPS),
        "NBootstrap": "1000",
        "NSteps": str(PAPER_STEPS),
        "HEarly": str(H_EARLY),
        "HMid": str(H_MID),
        "NParticleMin": str(min(PAPER_N)),
        "NParticleMax": str(max(PAPER_N)),
        "EpsMin": f"{min(PAPER_EPS):.2f}",
        "EpsMax": f"{max(PAPER_EPS):.2f}",
        "NICs": str(len(PAPER_INITS)),
        "NModels": str(len(PAPER_MODELS)),
        "ShowcaseN": str(SHOWCASE_N),
        "BimodalCoarseR": fmt_r(br),
        "BimodalCoarseCILo": "---" if blo is None else f"{blo:+.2f}",
        "BimodalCoarseCIHi": "---" if bhi is None else f"{bhi:+.2f}",
        "BimodalBestFineR": fmt_r(safe_float(bim.get("best_fine_r"))),
        "BimodalGap": fmt_r(
            (safe_float(bim.get("best_coarse_r")) or 0.0) - (safe_float(bim.get("best_fine_r")) or 0.0)
        ),
        "BimodalNRange": f"{min(PAPER_N)}--{max(PAPER_N)}",
        "BimodalNScaleMin": str(min(PAPER_N)),
        "BimodalNScaleMax": str(max(PAPER_N)),
        "HernquistVelDispR": fmt_r(hv),
        "HernquistVelDispCILo": "---" if hvlo is None else f"{hvlo:+.2f}",
        "HernquistVelDispCIHi": "---" if hvhi is None else f"{hvhi:+.2f}",
        "HernquistKNNR": fmt_r(get_metric(her_small, "kNN-all")),
        "HernquistVelDispEpsTen": fmt_r(get_metric(her_big, "VelDisp")),
        "PlummerVelDispR": fmt_r(pv),
        "PlummerVelDispCILo": "---" if pvlo is None else f"{pvlo:+.2f}",
        "PlummerVelDispCIHi": "---" if pvhi is None else f"{pvhi:+.2f}",
        "EnergyDriftMedian": f"{drift_median:.2e}",
        "EnergyDriftMax": f"{drift_max:.2e}",
    }
    with open(path, "w") as f:
        for k, v in mapping.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def write_verdict_summary(analysis: Dict[str, Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        f.write("\\begin{tabular}{lllrrrrl}\n")
        f.write("\\toprule\n")
        f.write("Model & IC & $\\epsilon$ & $r_{\\rm CG8}$ & Best fine & $r_{\\rm fine}$ & Gap & Verdict\\\\\n")
        f.write("\\midrule\n")
        for model in PAPER_MODELS:
            for init in IC_ORDER:
                for eps in PAPER_EPS:
                    cell = get_cell(analysis, model, init, 1024, eps)

                    # Display CG8 explicitly in its own column
                    rcg8 = get_metric(cell, "CoarseG8")

                    # Best fine observable
                    bfn = best_fine_name(cell)
                    rf = get_metric(cell, bfn) if bfn else None

                    # GAP MUST USE THE ACTUAL BEST COARSE COMPARATOR USED FOR THE VERDICT
                    bc = safe_float(cell.get("best_coarse_r"))
                    gap = None
                    if bc is not None and rf is not None:
                        gap = abs(rf) - abs(bc)

                    verdict = cell.get("verdict", "---")

                    f.write(
                        f"{MODEL_LABELS[model]} & {IC_LABELS[init]} & {eps:.2f} & "
                        f"{fmt_r(rcg8)} & {bfn or '---'} & {fmt_r(rf)} & {fmt_r(gap)} & {verdict}\\\\\n"
                    )
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_n_scaling(analysis: Dict[str, Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        f.write("\\begin{tabular}{llrrrr}\n")
        f.write("\\toprule\n")
        f.write("IC & $N$ & $r_{\\rm CG8}(\\epsilon=0.02)$ & $r_{\\rm CG8}(0.05)$ & $r_{\\rm CG8}(0.10)$ & best fine max\\\\\n")
        f.write("\\midrule\n")
        for init in IC_ORDER:
            for n in PAPER_N:
                vals = []
                bests = []
                for eps in PAPER_EPS:
                    cell = get_cell(analysis, "direct_isolated", init, n, eps)
                    vals.append(fmt_r(get_metric(cell, "CoarseG8")))
                    bf = safe_float(cell.get("best_fine_r"))
                    if bf is not None:
                        bests.append(abs(bf))
                bfmax = f"{max(bests):.3f}" if bests else "---"
                f.write(f"{IC_LABELS[init]} & {n} & {vals[0]} & {vals[1]} & {vals[2]} & {bfmax}\\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_cond_fine(analysis: Dict[str, Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        f.write("\\begin{tabular}{llrrrr}\n")
        f.write("\\toprule\n")
        f.write("IC & $\\epsilon$ & $r_{\\rm CG8}$ & $r_{\\rm VelDisp}$ & $r_{\\rm kNN}$ & Verdict\\\\\n")
        f.write("\\midrule\n")
        for init in ["hernquist3d", "plummer3d"]:
            for eps in PAPER_EPS:
                cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
                f.write(
                    f"{IC_LABELS[init]} & {eps:.2f} & {fmt_r(get_metric(cell, 'CoarseG8'))} & "
                    f"{fmt_r(get_metric(cell, 'VelDisp'))} & {fmt_r(get_metric(cell, 'kNN-all'))} & "
                    f"{cell.get('verdict', '---')}\\\\\n"
                )
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_diagnostics(rows: List[Dict[str, Any]], path: str) -> None:
    direct_rows = filter_rows(rows, model="direct_isolated")
    drifts = [safe_float(r.get("energy_rel_drift")) for r in direct_rows]
    drifts = [v for v in drifts if v is not None]
    vir0 = [safe_float(r.get("virial_0")) for r in direct_rows]
    virf = [safe_float(r.get("virial_f")) for r in direct_rows]
    vir0 = [v for v in vir0 if v is not None]
    virf = [v for v in virf if v is not None]
    with open(path, "w") as f:
        f.write("\\begin{tabular}{lrr}\n")
        f.write("\\toprule\n")
        f.write("Diagnostic & Median & Max/Min\\\\\n")
        f.write("\\midrule\n")
        f.write(f"Relative energy drift & {np.median(drifts):.2e} & {np.max(drifts):.2e}\\\\\n" if drifts else "Relative energy drift & --- & ---\\\\\n")
        f.write(f"Initial virial ratio & {np.median(vir0):.3f} & {np.min(vir0):.3f}/{np.max(vir0):.3f}\\\\\n" if vir0 else "Initial virial ratio & --- & ---\\\\\n")
        f.write(f"Final virial ratio & {np.median(virf):.3f} & {np.min(virf):.3f}/{np.max(virf):.3f}\\\\\n" if virf else "Final virial ratio & --- & ---\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def fig01_ic_gallery(showcase_pos0: Dict[str, np.ndarray]) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.0))
    for ax, init in zip(axes.flat, IC_ORDER):
        pos = showcase_pos0.get(init)
        if pos is None:
            draw_missing(ax, IC_LABELS[init])
            continue
        img = projected_density_image(pos, periodic=False)
        ax.imshow(img.T, origin="lower", extent=[0, 2, 0, 2], cmap="magma", aspect="equal")
        ax.set_title(IC_LABELS[init])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(f"Initial-condition gallery ($N={SHOWCASE_N}$, projected density at $t=0$)")
    savefig(fig, "fig01_ic_gallery.pdf")


def fig02_snapshots(showcase_snaps: Dict[str, Dict[Any, np.ndarray]]) -> None:
    plt.rcParams.update(STYLE)
    cases = [
        ("bimodal3d", 0.05),
        ("hernquist3d", 0.02),
        ("hernquist3d", 0.10),
    ]
    steps = [0, H_EARLY, H_MID, PAPER_STEPS]
    fig, axes = plt.subplots(len(cases), len(steps), figsize=(12.5, 8.0))
    if len(cases) == 1:
        axes = np.array([axes])
    for i, (init, eps) in enumerate(cases):
        key = f"{init}_{eps:.2f}"
        snap = showcase_snaps.get(key)
        for j, step in enumerate(steps):
            ax = axes[i, j]
            if snap is None or step not in snap:
                draw_missing(ax, f"{IC_LABELS[init]}, ε={eps:.2f}" if j == 0 else None)
                continue
            img = projected_density_image(snap[step], periodic=False)
            ax.imshow(img.T, origin="lower", extent=[0, 2, 0, 2], cmap="magma", aspect="equal")
            if i == 0:
                ax.set_title(f"t = {step}")
            if j == 0:
                ax.set_ylabel(f"{IC_LABELS[init]}\nε={eps:.2f}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Time evolution of projected density for showcase runs")
    savefig(fig, "fig02_snapshots.pdf")


def fig03_verdict_map(analysis: Dict[str, Dict[str, Any]]) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, model in zip(axes, PAPER_MODELS):
        mat = np.full((len(IC_ORDER), len(PAPER_EPS)), np.nan)
        labels = np.empty((len(IC_ORDER), len(PAPER_EPS)), dtype=object)
        for i, init in enumerate(IC_ORDER):
            for j, eps in enumerate(PAPER_EPS):
                cell = get_cell(analysis, model, init, 1024, eps)
                bf = safe_float(cell.get("best_fine_r"))
                bc = safe_float(cell.get("best_coarse_r"))
                if bf is not None and bc is not None:
                    mat[i, j] = bf - bc
                labels[i, j] = cell.get("verdict", "---")
        im = ax.imshow(mat, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-0.35, vmax=0.35)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xticks(range(len(PAPER_EPS)))
        ax.set_xticklabels([f"{eps:.2f}" for eps in PAPER_EPS])
        ax.set_yticks(range(len(IC_ORDER)))
        ax.set_yticklabels([IC_LABELS[i] for i in IC_ORDER])
        ax.set_xlabel(r"$\epsilon$")
        for i in range(len(IC_ORDER)):
            for j in range(len(PAPER_EPS)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.2f}\n{labels[i,j]}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(r"$|r_{\rm best\ fine}| - |r_{\rm best\ coarse}|$")
    fig.suptitle("Verdict map at $N=1024$: fine advantage by IC family and softening")
    savefig(fig, "fig03_verdict_map.pdf")


def fig04_bimodal_anchor(analysis: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax1, ax2 = axes

    rs = filter_rows(rows, model="direct_isolated", init="bimodal3d", n=1024, eps=0.05)
    x = np.array([r["coarse_g8_0"] for r in rs], dtype=float) if rs else np.array([])
    y = np.array([r["d_coarse_g8_early"] for r in rs], dtype=float) if rs else np.array([])
    if len(x) >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
        ax1.scatter(x, y, color=IC_COLORS["bimodal3d"], alpha=0.75, s=18)
        p = np.polyfit(x, y, 1)
        xs = np.linspace(np.min(x), np.max(x), 100)
        ax1.plot(xs, p[0] * xs + p[1], color="0.25", lw=2)
        ax1.set_xlabel(r"Initial coarse density variance $\sigma_\rho^2(G8)$")
        ax1.set_ylabel(r"Future $\Delta C_8^{\rm early}$")
        ax1.set_title("Bimodal anchor: direct-isolated, $N=1024$, $\\epsilon=0.05$")
    else:
        draw_missing(ax1, "Bimodal anchor")

    coarse = []
    fine = []
    for eps in PAPER_EPS:
        cell = get_cell(analysis, "direct_isolated", "bimodal3d", 1024, eps)
        coarse.append(get_metric(cell, "CoarseG8"))
        fine_name = best_fine_name(cell)
        fine.append(get_metric(cell, fine_name) if fine_name else None)
    ax2.plot(PAPER_EPS, [np.nan if v is None else abs(v) for v in coarse], marker="o", color=PRED_COLORS["CoarseG8"], label="CoarseG8")
    ax2.plot(PAPER_EPS, [np.nan if v is None else abs(v) for v in fine], marker="s", color="0.35", label="best fine")
    ax2.set_xlabel(r"$\epsilon$")
    ax2.set_ylabel(r"$|r|$")
    ax2.set_title("Bimodal coarse dominance across softening")
    ax2.legend(frameon=False)
    savefig(fig, "fig04_bimodal_anchor.pdf")


def fig05_cond_fine(analysis: Dict[str, Dict[str, Any]]) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    for ax, init in zip(axes, ["hernquist3d", "plummer3d"]):
        for pred in ["CoarseG8", "VelDisp", "kNN-all"]:
            vals = []
            lo = []
            hi = []
            for eps in PAPER_EPS:
                cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
                vals.append(get_metric(cell, pred))
                clo, chi = get_ci(cell, pred)
                lo.append(clo)
                hi.append(chi)
            yy = np.array([np.nan if v is None else v for v in vals], dtype=float)
            ax.plot(PAPER_EPS, yy, marker=EPS_MK[PAPER_EPS[0]] if pred == "CoarseG8" else None, color=PRED_COLORS.get(pred, "0.4"), label=pred)
            lo_arr = np.array([np.nan if v is None else v for v in lo], dtype=float)
            hi_arr = np.array([np.nan if v is None else v for v in hi], dtype=float)
            if np.any(np.isfinite(lo_arr)) and np.any(np.isfinite(hi_arr)):
                ax.fill_between(PAPER_EPS, lo_arr, hi_arr, color=PRED_COLORS.get(pred, "0.4"), alpha=0.15)
        ax.axhline(0.0, color="0.7", lw=1, ls="--")
        ax.set_title(IC_LABELS[init])
        ax.set_xlabel(r"$\epsilon$")
    axes[0].set_ylabel(r"Pearson $r$ with $\Delta C_8^{\rm early}$")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Conditional fine-leaning candidates: concentrated profiles at $N=1024$")
    savefig(fig, "fig05_cond_fine.pdf")


def fig06_eps_transition(analysis: Dict[str, Dict[str, Any]]) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for init in IC_ORDER:
        vals = []
        for eps in PAPER_EPS:
            cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
            bf = safe_float(cell.get("best_fine_r"))
            bc = safe_float(cell.get("best_coarse_r"))
            vals.append(np.nan if (bf is None or bc is None) else abs(bc) - abs(bf))
        ax.plot(PAPER_EPS, vals, marker="o", color=IC_COLORS[init], label=IC_LABELS[init])
    ax.axhline(0.0, color="0.6", lw=1, ls="--")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$|r_{\rm coarse}| - |r_{\rm best\ fine}|$")
    ax.set_title("Coarse advantage vs softening ($N=1024$, direct-isolated)")
    ax.legend(frameon=False, ncol=2)
    savefig(fig, "fig06_eps_transition.pdf")


def fig07_n_scaling(analysis: Dict[str, Dict[str, Any]]) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.4), sharey=True)
    for ax, init in zip(axes, IC_ORDER):
        for eps in PAPER_EPS:
            coarse_vals = []
            fine_vals = []
            for n in PAPER_N:
                cell = get_cell(analysis, "direct_isolated", init, n, eps)
                coarse_vals.append(get_metric(cell, "CoarseG8"))
                bf = safe_float(cell.get("best_fine_r"))
                fine_vals.append(bf)
            ax.plot(PAPER_N, coarse_vals, marker=EPS_MK[eps], ls=EPS_LS[eps], color=IC_COLORS[init], label=f"coarse ε={eps:.2f}")
            ax.plot(PAPER_N, fine_vals, marker=EPS_MK[eps], ls=EPS_LS[eps], color="0.5", alpha=0.55)
        ax.axhline(0.0, color="0.75", lw=1, ls="--")
        ax.set_title(IC_LABELS[init], color=IC_COLORS[init], fontweight="bold")
        ax.set_xlabel(r"$N$")
        ax.set_xticks(PAPER_N)
    axes[0].set_ylabel(r"Pearson $r$ with $\Delta C_8^{\rm early}$")
    handles = [
        plt.Line2D([0], [0], color="k", lw=1.8, label=r"$r_{\rm CG8}$ (colored)"),
        plt.Line2D([0], [0], color="0.5", lw=1.8, label="best fine obs"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("N-scaling: coarse predictor vs best fine predictor (direct-isolated)", y=1.06)
    savefig(fig, "fig07_n_scaling.pdf")


def fig08_model_comparison(analysis: Dict[str, Dict[str, Any]]) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5), sharey=True)
    for ax, init in zip(axes, IC_ORDER):
        for model in PAPER_MODELS:
            vals = [get_metric(get_cell(analysis, model, init, 1024, eps), "CoarseG8") for eps in PAPER_EPS]
            ax.plot(PAPER_EPS, vals, marker="o", color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.axhline(0.0, color="0.75", lw=1, ls="--")
        ax.set_title(IC_LABELS[init], color=IC_COLORS[init], fontweight="bold")
        ax.set_xlabel(r"$\epsilon$")
    axes[0].set_ylabel(r"$r_{\rm CG8}$ vs $\Delta C_8^{\rm early}$")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Force-model comparison at $N=1024$")
    savefig(fig, "fig08_model_comparison.pdf")


def fig09_diagnostics(rows: List[Dict[str, Any]]) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    ax1, ax2, ax3 = axes
    direct_rows = filter_rows(rows, model="direct_isolated")
    drifts = [safe_float(r.get("energy_rel_drift")) for r in direct_rows]
    drifts = np.array([v for v in drifts if v is not None], dtype=float)
    if len(drifts) > 0:
        ax1.hist(drifts, bins=20, color="#4c78a8", alpha=0.85)
        ax1.set_xscale("log")
        ax1.set_title("Relative energy drift")
        ax1.set_xlabel(r"$|\Delta E|/|E_0|$")
    else:
        draw_missing(ax1, "Relative energy drift")

    vir0 = np.array([safe_float(r.get("virial_0")) for r in direct_rows], dtype=float)
    vir0 = vir0[np.isfinite(vir0)]
    virf = np.array([safe_float(r.get("virial_f")) for r in direct_rows], dtype=float)
    virf = virf[np.isfinite(virf)]
    if len(vir0) and len(virf):
        ax2.hist(vir0, bins=18, alpha=0.55, label="initial")
        ax2.hist(virf, bins=18, alpha=0.55, label="final")
        ax2.set_title("Virial-ratio distribution")
        ax2.set_xlabel(r"$Q = 2K/|U|$")
        ax2.legend(frameon=False)
    else:
        draw_missing(ax2, "Virial-ratio distribution")

    x = []
    y = []
    c = []
    for init in IC_ORDER:
        rs = filter_rows(rows, model="direct_isolated", init=init, n=1024, eps=0.05)
        x.extend([safe_float(r.get("energy_rel_drift")) for r in rs])
        y.extend([safe_float(r.get("virial_f")) for r in rs])
        c.extend([IC_COLORS[init]] * len(rs))
    xx = np.array([v for v in x if v is not None], dtype=float)
    yy = np.array([v for v in y if v is not None], dtype=float)
    if len(xx) == len(yy) and len(xx) > 0:
        ax3.scatter(xx, yy, s=15, alpha=0.6, color=c[:len(xx)], edgecolors="none")
        ax3.set_xscale("log")
        ax3.set_xlabel(r"$|\Delta E|/|E_0|$")
        ax3.set_ylabel(r"final virial ratio")
        ax3.set_title("Energy drift vs final virial ratio")
    else:
        draw_missing(ax3, "Energy drift vs final virial ratio")
    savefig(fig, "fig09_diagnostics.pdf")


def fig10_summary_matrix(analysis: Dict[str, Dict[str, Any]]) -> None:
    plt.rcParams.update(STYLE)
    model = "direct_isolated"
    n_ref = 1024
    obs_classes = [
        ("CoarseG8", r"Coarse" "\n" r"$\sigma_\rho^2(G8)$"),
        ("kNN-all", "Fine pos.\nkNN-all"),
        ("ClosePairs", "Fine pos.\nClose pairs"),
        ("VelDisp", "Fine kin.\nVelDisp"),
    ]
    mat = np.full((len(IC_ORDER), len(obs_classes) * len(PAPER_EPS)), np.nan)
    for i, init in enumerate(IC_ORDER):
        for j, (pred, _) in enumerate(obs_classes):
            for k, eps in enumerate(PAPER_EPS):
                cell = get_cell(analysis, model, init, n_ref, eps)
                v = get_metric(cell, pred)
                if v is not None:
                    mat[i, j * len(PAPER_EPS) + k] = v
    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    im = ax.imshow(mat, origin="upper", cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(IC_ORDER)))
    ax.set_yticklabels([IC_LABELS[i] for i in IC_ORDER], fontsize=11)
    col_labels = [rf"$\epsilon={eps:.2f}$" for _ in obs_classes for eps in PAPER_EPS]
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right")
    for divider in range(1, len(obs_classes)):
        ax.axvline(divider * len(PAPER_EPS) - 0.5, color="white", lw=2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.55 else "0.1")
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    centers = [j * len(PAPER_EPS) + (len(PAPER_EPS) - 1) / 2 for j in range(len(obs_classes))]
    ax2.set_xticks(centers)
    ax2.set_xticklabels([lbl for _, lbl in obs_classes], fontsize=10)
    ax2.tick_params(length=0)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"Pearson $r$ with $\Delta C_8^{\rm early}$")
    ax.set_title(f"Summary matrix at $N={n_ref}$, direct-isolated")
    savefig(fig, "fig10_summary_matrix.pdf")


def fig11_bimodal_mechanism(showcase_pos0: Dict[str, np.ndarray], analysis: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax1, ax2 = axes
    pos0 = showcase_pos0.get("bimodal3d")
    if pos0 is None:
        draw_missing(ax1, "Bimodal initial state")
    else:
        x = pos0[:, 0]
        y = pos0[:, 1]
        ax1.scatter(x, y, s=4, alpha=0.6, color=IC_COLORS["bimodal3d"], edgecolors="none")
        grid = 8
        for v in np.linspace(0.0, 2.0, grid + 1):
            ax1.axvline(v, color="0.75", lw=0.6)
            ax1.axhline(v, color="0.75", lw=0.6)
        ax1.set_xlim(0, 2)
        ax1.set_ylim(0, 2)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_title("Bimodal showcase: projected particles + coarse 8×8 grid")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

    for n, marker in [(256, "o"), (1024, "s")]:
        rs = filter_rows(rows, model="direct_isolated", init="bimodal3d", n=n, eps=0.05)
        if rs:
            xx = np.array([r["coarse_g8_0"] for r in rs], dtype=float)
            yy = np.array([r["d_coarse_g8_early"] for r in rs], dtype=float)
            ax2.scatter(xx, yy, s=22, alpha=0.65, marker=marker, label=f"N={n}", edgecolors="none")
    ax2.set_xlabel(r"$\sigma_\rho^2(G8)$ at $t=0$")
    ax2.set_ylabel(r"$\Delta C_8^{\rm early}$")
    ax2.set_title("Bimodal anchor across particle count")
    ax2.legend(frameon=False)
    savefig(fig, "fig11_bimodal_mechanism.pdf")


def fig12_veldisp_mechanism(showcase_snaps: Dict[str, Dict[Any, np.ndarray]]) -> None:
    """
    Scientifically matched Fig. 12:
    - Hernquist showcase only
    - compare epsilon = 0.02 vs 0.10
    - use per-particle local velocity-dispersion values consistent with the
      stress-battery VelDisp observable class

    Panels per row:
      (1) projected density
      (2) projected map of mean local VelDisp
      (3) radial profile of local VelDisp vs projected radius
    """
    plt.rcParams.update(STYLE)
    cases = [("hernquist3d", 0.02), ("hernquist3d", 0.10)]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.6), constrained_layout=True)

    for i, (init, eps) in enumerate(cases):
        ax1, ax2, ax3 = axes[i]
        key = f"{init}_{eps:.2f}"
        sc = showcase_snaps.get(key)
        title = rf"{IC_LABELS[init]}, $\epsilon={eps:.2f}$"

        if sc is None or 0 not in sc or "vel0" not in sc:
            draw_missing(ax1, title)
            draw_missing(ax2)
            draw_missing(ax3)
            continue

        pos0 = np.asarray(sc[0], dtype=float)
        vel0 = np.asarray(sc["vel0"], dtype=float)

        if pos0.ndim != 2 or pos0.shape[1] != 3 or vel0.ndim != 2 or vel0.shape != pos0.shape:
            draw_missing(ax1, title, text="Malformed showcase arrays")
            draw_missing(ax2)
            draw_missing(ax3)
            continue

        img = projected_density_image(pos0, periodic=False)
        ax1.imshow(img.T, origin="lower", extent=[0, 2, 0, 2], cmap="magma", aspect="equal")
        ax1.set_title(title)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        pos_in, local_std = compute_local_veldisp_per_particle(pos0, vel0, PAPER_K, periodic=False, box_size=2.0)
        if len(local_std) == 0:
            draw_missing(ax2, "Projected local VelDisp map")
            draw_missing(ax3, "Radial VelDisp profile")
            continue

        x = pos_in[:, 0]
        y = pos_in[:, 1]
        r_proj = np.sqrt((x - 1.0) ** 2 + (y - 1.0) ** 2)

        gridsz = 28
        xedges = np.linspace(0.0, 2.0, gridsz + 1)
        yedges = np.linspace(0.0, 2.0, gridsz + 1)
        counts, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
        sums, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=local_std)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_local = np.where(counts > 0, sums / counts, np.nan)
        im = ax2.imshow(
            mean_local.T,
            origin="lower",
            extent=[0, 2, 0, 2],
            aspect="equal",
            cmap="viridis",
        )
        ax2.set_title("Projected local VelDisp")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
        cbar.set_label("mean local VelDisp")

        nbins = 24
        bins = np.linspace(np.min(r_proj), np.max(r_proj), nbins + 1)
        idx = np.digitize(r_proj, bins) - 1
        rc, mu, sd = [], [], []
        for b in range(nbins):
            m = idx == b
            if np.sum(m) >= 6:
                rc.append(0.5 * (bins[b] + bins[b + 1]))
                mu.append(np.mean(local_std[m]))
                sd.append(np.std(local_std[m]))
        if rc:
            rc = np.asarray(rc)
            mu = np.asarray(mu)
            sd = np.asarray(sd)
            ax3.plot(rc, mu, color=PRED_COLORS["VelDisp"], lw=2)
            ax3.fill_between(rc, mu - sd, mu + sd, color=PRED_COLORS["VelDisp"], alpha=0.2)
            ax3.set_title("Radial local VelDisp profile")
            ax3.set_xlabel(r"projected radius from box center")
            ax3.set_ylabel("local VelDisp")
        else:
            draw_missing(ax3, "Radial VelDisp profile")

    savefig(fig, "fig12_veldisp_mechanism.pdf")


def fig13_eps_boundary(analysis: Dict[str, Dict[str, Any]]) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    a_ref = 0.20
    for init in IC_ORDER:
        x = np.array([eps / a_ref for eps in PAPER_EPS], dtype=float)
        y = []
        for eps in PAPER_EPS:
            cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
            bf = safe_float(cell.get("best_fine_r"))
            bc = safe_float(cell.get("best_coarse_r"))
            y.append(np.nan if (bf is None or bc is None) else abs(bc) - abs(bf))
        ax.plot(x, y, marker="o", color=IC_COLORS[init], label=IC_LABELS[init])
    ax.axhline(0.0, color="0.55", lw=1, ls="--")
    ax.axvspan(0.10, 0.50, color="0.92", zorder=0)
    ax.text(0.41, 0.01, "coarse wins  ↑", color="0.45", fontsize=9)
    ax.text(0.41, -0.015, "fine wins  ↓", color="0.45", fontsize=9)
    ax.text(0.29, -0.028, "transition\nzone", color="0.5", fontsize=8, ha="center")
    ax.set_xlabel(r"Softening-to-scale-radius ratio $\epsilon/a$")
    ax.set_ylabel(r"$|r_{\rm coarse}| - |r_{\rm best\ fine}|$")
    ax.set_title(r"Verdict boundary in $\epsilon/a$ ($N=1024$, direct-isolated)")
    ax.legend(frameon=False, loc="best")
    savefig(fig, "fig13_eps_boundary.pdf")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--replicates", type=int, default=PAPER_REPS)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--use-numba", dest="use_numba", action="store_true", default=True)
    parser.add_argument("--no-numba", dest="use_numba", action="store_false")
    args = parser.parse_args()

    if args.use_numba and not _HAS_NUMBA:
        print("numba not found — falling back to NumPy")
        args.use_numba = False

    ensure_dirs()
    battery_csv = os.path.join(DATA_DIR, "paper_battery.csv")

    if args.no_run:
        if not os.path.exists(battery_csv):
            raise FileNotFoundError(f"--no-run was given but {battery_csv} does not exist.")
        print(f"Loading existing battery from {battery_csv}")
        rows = load_csv_rows(battery_csv)
    else:
        configs = build_configs(reps=args.replicates)
        print(f"Running battery: {len(configs)} runs")
        rows = run_battery(args.workers, configs, args.use_numba)
        write_csv_rows(battery_csv, rows)

    ok_count = sum(1 for r in rows if r.get("status") == "ok")
    print(f"ok runs: {ok_count}/{len(rows)}")
    if ok_count == 0:
        raise RuntimeError("No successful stress runs. Refusing to generate blank paper figures.")

    analysis = analyse(rows, n_boot=args.n_boot)
    if not analysis:
        raise RuntimeError("Analysis dictionary is empty. Refusing to generate blank paper figures.")

    with open(os.path.join(DATA_DIR, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=_json_default)

    showcase_specs = [
        ("bimodal3d", 0.05),
        ("hernquist3d", 0.02),
        ("hernquist3d", 0.10),
        ("plummer3d", 0.05),
        ("cold_clumpy3d", 0.05),
    ]
    showcase_pos0: Dict[str, np.ndarray] = {}
    showcase_snaps: Dict[str, Dict[Any, np.ndarray]] = {}
    for init, eps in showcase_specs:
        key = f"{init}_{eps:.2f}"
        snaps = _run_showcase_sim(init, SHOWCASE_SEED, SHOWCASE_N, eps, PAPER_STEPS)
        showcase_snaps[key] = snaps
        if 0 in snaps:
            if init not in showcase_pos0 or abs(eps - 0.05) < 1e-12:
                showcase_pos0[init] = snaps[0]

    write_macros(analysis, rows, os.path.join(DATA_DIR, "paper_macros.tex"))
    write_verdict_summary(analysis, os.path.join(TABLE_DIR, "verdict_summary.tex"))
    write_n_scaling(analysis, os.path.join(TABLE_DIR, "n_scaling.tex"))
    write_cond_fine(analysis, os.path.join(TABLE_DIR, "cond_fine.tex"))
    write_diagnostics(rows, os.path.join(TABLE_DIR, "diagnostics.tex"))

    print("Generating figures...")
    fig01_ic_gallery(showcase_pos0)
    fig02_snapshots(showcase_snaps)
    fig03_verdict_map(analysis)
    fig04_bimodal_anchor(analysis, rows)
    fig05_cond_fine(analysis)
    fig06_eps_transition(analysis)
    fig07_n_scaling(analysis)
    fig08_model_comparison(analysis)
    fig09_diagnostics(rows)
    fig10_summary_matrix(analysis)
    fig11_bimodal_mechanism(showcase_pos0, analysis, rows)
    fig12_veldisp_mechanism(showcase_snaps)
    fig13_eps_boundary(analysis)
    print("Done.")


if __name__ == "__main__":
    main()