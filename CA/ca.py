"""
ca.py
=====
One-file paper package generator for the CA paper.

Generates paper-facing outputs for:
- Study A: observer disagreement
- Study B: target-relative predictive scale
- Study C: embedded vs alone + failed mediation
- Study D: empirical slope-law
- Study D.2: dynamic local null adjudication

Outputs:
  outputs/data/
  outputs/figures/
  outputs/tables/
  outputs/logs/

Dependencies:
  numpy, pandas, scipy, sklearn, matplotlib, seaborn, tqdm

Run:
  python ca.py
"""

from __future__ import annotations

import json
import os
import time
from collections import deque

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# =============================================================================
# PATHS / GLOBAL STYLE
# =============================================================================

ROOT = os.path.abspath(os.path.dirname(__file__))
OUT = os.path.join(ROOT, "outputs")
DATA_DIR = os.path.join(OUT, "data")
FIG_DIR = os.path.join(OUT, "figures")
TAB_DIR = os.path.join(OUT, "tables")
LOG_DIR = os.path.join(OUT, "logs")

for d in [OUT, DATA_DIR, FIG_DIR, TAB_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)


def savefig(fig: plt.Figure, stem: str) -> None:
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(FIG_DIR, stem + ext), dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# CA ENGINE
# =============================================================================

def _nbr(grid: np.ndarray) -> np.ndarray:
    g = grid.astype(np.int32)
    return (
        np.roll(g, 1, 0) + np.roll(g, -1, 0) +
        np.roll(g, 1, 1) + np.roll(g, -1, 1) +
        np.roll(np.roll(g, 1, 0), 1, 1) +
        np.roll(np.roll(g, 1, 0), -1, 1) +
        np.roll(np.roll(g, -1, 0), 1, 1) +
        np.roll(np.roll(g, -1, 0), -1, 1)
    )


def gol_step(grid: np.ndarray) -> np.ndarray:
    n = _nbr(grid)
    return (((grid == 1) & ((n == 2) | (n == 3))) |
            ((grid == 0) & (n == 3))).astype(np.uint8)


def comp_count_periodic(grid: np.ndarray) -> int:
    nrows, ncols = grid.shape
    visited = np.zeros((nrows, ncols), dtype=bool)
    n_comp = 0
    rows, cols = np.where(grid.astype(bool))
    for r0, c0 in zip(rows.tolist(), cols.tolist()):
        if visited[r0, c0]:
            continue
        n_comp += 1
        q = deque([(r0, c0)])
        visited[r0, c0] = True
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = (r + dr) % nrows, (c + dc) % ncols
                if grid[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
    return n_comp


def comp_count_nonperiodic_4(grid: np.ndarray) -> int:
    nrows, ncols = grid.shape
    visited = np.zeros((nrows, ncols), dtype=bool)
    n_comp = 0
    rows, cols = np.where(grid.astype(bool))
    for r0, c0 in zip(rows.tolist(), cols.tolist()):
        if visited[r0, c0]:
            continue
        n_comp += 1
        q = deque([(r0, c0)])
        visited[r0, c0] = True
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    if grid[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
    return n_comp


def block_count(grid: np.ndarray, B: int) -> int:
    n = grid.shape[0]
    nb = n // B
    trimmed = grid[:nb * B, :nb * B]
    return int(trimmed.reshape(nb, B, nb, B).any(axis=(1, 3)).sum())


def block_grid(grid: np.ndarray, B: int) -> np.ndarray:
    n = grid.shape[0]
    nb = n // B
    trimmed = grid[:nb * B, :nb * B]
    return trimmed.reshape(nb, B, nb, B).any(axis=(1, 3)).astype(np.uint8)


def embedded_isolated_coords(grid: np.ndarray):
    nrows, ncols = grid.shape
    coords = []
    live_rows, live_cols = np.where(grid == 1)
    for r, c in zip(live_rows.tolist(), live_cols.tolist()):
        up = grid[(r - 1) % nrows, c]
        down = grid[(r + 1) % nrows, c]
        left = grid[r, (c - 1) % ncols]
        right = grid[r, (c + 1) % ncols]

        if int(up + down + left + right) != 0:
            continue

        diag_sum = int(
            grid[(r - 1) % nrows, (c - 1) % ncols] +
            grid[(r - 1) % nrows, (c + 1) % ncols] +
            grid[(r + 1) % nrows, (c - 1) % ncols] +
            grid[(r + 1) % nrows, (c + 1) % ncols]
        )
        if diag_sum > 0:
            coords.append((r, c))
    return coords


def isolated_counts(grid: np.ndarray) -> dict:
    nrows, ncols = grid.shape
    embedded = 0
    alone = 0
    live_rows, live_cols = np.where(grid == 1)
    for r, c in zip(live_rows.tolist(), live_cols.tolist()):
        up = grid[(r - 1) % nrows, c]
        down = grid[(r + 1) % nrows, c]
        left = grid[r, (c - 1) % ncols]
        right = grid[r, (c + 1) % ncols]

        if int(up + down + left + right) != 0:
            continue

        diag_sum = int(
            grid[(r - 1) % nrows, (c - 1) % ncols] +
            grid[(r - 1) % nrows, (c + 1) % ncols] +
            grid[(r + 1) % nrows, (c - 1) % ncols] +
            grid[(r + 1) % nrows, (c + 1) % ncols]
        )
        if diag_sum > 0:
            embedded += 1
        else:
            alone += 1
    return {"embedded": embedded, "alone": alone}


def dynamic_local_delta_for_focal(full_grid: np.ndarray, r: int, c: int, core_radius: int = 2) -> float:
    """
    Correct dynamic local null.

    Remove the focal cell on the full torus, step both full grids once,
    then compare non-periodic 4-connected component counts inside the
    local core window around the focal site.

    +1.0 = baseline component contribution of the focal cell's presence.
    With beta_death = -1.0, this makes
    beta_residual = beta_emp - beta_death - beta_local
    equivalent to beta_emp - mean(delta_sync).
    """
    g_with = full_grid.copy()
    g_without = full_grid.copy()
    g_without[r, c] = 0

    next_with = gol_step(g_with)
    next_without = gol_step(g_without)

    nrows, ncols = full_grid.shape
    rs = [(r + dr) % nrows for dr in range(-core_radius, core_radius + 1)]
    cs = [(c + dc) % ncols for dc in range(-core_radius, core_radius + 1)]

    core_with = next_with[np.ix_(rs, cs)]
    core_without = next_without[np.ix_(rs, cs)]

    delta_sync = (
        comp_count_nonperiodic_4(core_without)
        - comp_count_nonperiodic_4(core_with)
    )
    return float(delta_sync + 1.0)


# =============================================================================
# STATS HELPERS
# =============================================================================

def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    r, p = sp_stats.pearsonr(x[m], y[m])
    return float(r), float(p)


def ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    sl, ic, r, p, se = sp_stats.linregress(x, y)
    n = len(x)
    tcrit = sp_stats.t.ppf(0.975, df=n - 2)
    return {
        "slope": float(sl),
        "intercept": float(ic),
        "r": float(r),
        "p": float(p),
        "se": float(se),
        "ci95_lo": float(sl - tcrit * se),
        "ci95_hi": float(sl + tcrit * se),
        "n": int(n),
        "r2": float(r * r),
    }


def partial_r(x, y, z):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[m]
    y = y[m]
    z = z[m]

    def resid(a, b):
        sl, ic, _, _, _ = sp_stats.linregress(b, a)
        return a - (sl * b + ic)

    return pearson(resid(x, z), resid(y, z))


def bootstrap_partial_r(x, y, z, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    n = len(x)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = partial_r(x[idx], y[idx], z[idx])[0]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ridge_cv_r2(X, y, alpha=1.0, cv=5):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[m]
    y = y[m]
    if len(y) < cv + 1:
        return np.nan
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rd", Ridge(alpha=alpha)),
    ])
    return float(cross_val_score(pipe, X, y, cv=cv, scoring="r2").mean())


def standardize_cols(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        x = out[c].astype(float).values
        mu = x.mean()
        sd = x.std(ddof=0)
        out[c] = 0.0 if sd <= 0 else (x - mu) / sd
    return out


def r2_score_manual(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# =============================================================================
# STUDY A
# =============================================================================

def run_study_A():
    cfg = {
        "SEED": 20260325,
        "N": 1000,
        "GRID": 64,
        "T": 250,
        "T_EARLY": 50,
        "B_COARSE": 8,
        "RHO_MIN": 0.03,
        "RHO_MAX": 0.58,
        "N_BOOT": 1000,
    }

    rng = np.random.default_rng(cfg["SEED"])
    world_seeds = rng.integers(0, 2**62, size=cfg["N"])
    rhos = rng.uniform(cfg["RHO_MIN"], cfg["RHO_MAX"], cfg["N"])

    rows = []

    for i in tqdm(range(cfg["N"]), desc="Study A"):
        wrng = np.random.default_rng(int(world_seeds[i]))
        grid = (wrng.random((cfg["GRID"], cfg["GRID"])) < rhos[i]).astype(np.uint8)

        comp = np.empty(cfg["T"] + 1, dtype=np.int64)
        blk = np.empty(cfg["T"] + 1, dtype=np.int64)
        for t in range(cfg["T"] + 1):
            comp[t] = comp_count_periodic(grid)
            blk[t] = block_count(grid, cfg["B_COARSE"])
            if t < cfg["T"]:
                grid = gol_step(grid)

        delta_early = int(comp[cfg["T_EARLY"]] - comp[0])
        net_F = int(comp[cfg["T"]] - comp[0])
        net_C = int(blk[cfg["T"]] - blk[0])

        c0_F = max(int(comp[0]), 1)
        c0_C = max(int(blk[0]), 1)
        rho_F = net_F / c0_F
        rho_C = net_C / c0_C
        G = rho_C - rho_F

        rows.append({
            "world_id": i,
            "world_seed": int(world_seeds[i]),
            "density": float(rhos[i]),
            "delta_early": delta_early,
            "net_F": net_F,
            "net_C": net_C,
            "rho_F": rho_F,
            "rho_C": rho_C,
            "G": G,
            "c0_F": c0_F,
            "c0_C": c0_C,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "study_A_worlds.csv"), index=False)

    de = df["delta_early"].values.astype(float)
    G = df["G"].values.astype(float)
    rho = df["density"].values.astype(float)
    netC = df["net_C"].values.astype(float)

    r_main, p_main = pearson(de, G)
    r_indep, p_indep = pearson(de, netC)
    ols_res = ols(de, G)
    pr, pr_p = partial_r(de, G, rho)
    pr_lo, pr_hi = bootstrap_partial_r(de, G, rho, n_boot=cfg["N_BOOT"], seed=cfg["SEED"])

    verdict = "PASS — Study A center earned" if abs(pr) >= 0.20 else (
        "NARROW PASS — weak residual signal" if abs(pr) >= 0.10 else
        "FAIL — Study A independent predictor not earned"
    )

    save_json({
        "r_main": r_main,
        "p_main": p_main,
        "r_indep": r_indep,
        "p_indep": p_indep,
        "ols": ols_res,
        "partial_r": pr,
        "partial_r_p": pr_p,
        "partial_r_boot_ci95": [pr_lo, pr_hi],
        "verdict": verdict,
    }, os.path.join(DATA_DIR, "study_A_stats.json"))

    # Figure 1
    fig1_source = df[["delta_early", "G", "density"]].copy()
    fig1_source.to_csv(os.path.join(DATA_DIR, "fig1_studyA_scatter_source.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(de, G, c=rho, cmap="plasma", s=8, alpha=0.60, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Initial density ρ")
    xl = np.linspace(de.min(), de.max(), 300)
    ax.plot(xl, ols_res["slope"] * xl + ols_res["intercept"], "k--", lw=1.5)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.text(
        de.min(), G.min(),   # bottom-left of data
        (
            f"r = {r_main:.3f}\n"
            f"partial r($\\Delta_{{\\mathrm{{early}}}}, G \\mid \\rho$) = {pr:.3f}\n"
            f"95% CI [{pr_lo:.3f}, {pr_hi:.3f}]\n"
            f"N = {cfg['N']}"
        ),
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.6", fc="white", alpha=0.9),
    )
    ax.set_xlabel(r"$\Delta_{\mathrm{early}}$ (component-count net change [0,50])")
    ax.set_ylabel(r"$G$ ($\rho_C - \rho_F$)")
    ax.set_title("Study A — disagreement result")
    savefig(fig, "fig1_studyA_scatter")

    # Figure 2
    death_row = df.loc[df["G"].idxmax()]
    birth_row = df.loc[df["G"].idxmin()]

    trace_rows = []
    for label, row in [("G>0", death_row), ("G<0", birth_row)]:
        wrng = np.random.default_rng(int(row["world_seed"]))
        grid = (wrng.random((cfg["GRID"], cfg["GRID"])) < float(row["density"])).astype(np.uint8)

        comp = np.empty(cfg["T"] + 1, dtype=np.int64)
        blk = np.empty(cfg["T"] + 1, dtype=np.int64)
        for t in range(cfg["T"] + 1):
            comp[t] = comp_count_periodic(grid)
            blk[t] = block_count(grid, cfg["B_COARSE"])
            if t < cfg["T"]:
                grid = gol_step(grid)

        netF = np.cumsum(np.concatenate([[0], np.diff(comp.astype(np.int64))]))
        netC = np.cumsum(np.concatenate([[0], np.diff(blk.astype(np.int64))]))
        for t in range(cfg["T"] + 1):
            trace_rows.append({
                "label": label,
                "step": t,
                "net_F_cum": int(netF[t]),
                "net_C_cum": int(netC[t]),
                "G": float(row["G"]),
                "world_id": int(row["world_id"]),
            })

    df_trace = pd.DataFrame(trace_rows)
    df_trace.to_csv(os.path.join(DATA_DIR, "fig2_studyA_traces_source.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label in zip(axes, ["G>0", "G<0"]):
        sub = df_trace[df_trace["label"] == label]
        sns.lineplot(data=sub, x="step", y="net_F_cum", ax=ax, linewidth=2.0, label="Fine")
        sns.lineplot(data=sub, x="step", y="net_C_cum", ax=ax, linewidth=2.0, linestyle="--", label="Coarse")
        ax.axvspan(0, cfg["T_EARLY"], alpha=0.12, color="gold")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_title(f"{label}, world {int(sub['world_id'].iloc[0])}, G={sub['G'].iloc[0]:.3f}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative net change")
        ax.legend()
    savefig(fig, "fig2_studyA_traces")

    pd.DataFrame([
        {"stat": "r_main", "value": r_main},
        {"stat": "partial_r", "value": pr},
        {"stat": "ols_slope", "value": ols_res["slope"]},
        {"stat": "verdict", "value": verdict},
    ]).to_csv(os.path.join(TAB_DIR, "table1_studyA_key_stats.csv"), index=False)

    # --- Replication on second seed ---
    rep_seed = 20260326
    rep_rng = np.random.default_rng(rep_seed)
    rep_ws = rep_rng.integers(0, 2**62, size=cfg["N"])
    rep_rhos = rep_rng.uniform(cfg["RHO_MIN"], cfg["RHO_MAX"], cfg["N"])

    rep_de = np.empty(cfg["N"], dtype=float)
    rep_G = np.empty(cfg["N"], dtype=float)
    rep_rho = rep_rhos.copy()

    for i in tqdm(range(cfg["N"]), desc="Study A replication"):
        wrng = np.random.default_rng(int(rep_ws[i]))
        grid = (wrng.random((cfg["GRID"], cfg["GRID"])) < rep_rhos[i]).astype(np.uint8)

        comp = np.empty(cfg["T"] + 1, dtype=np.int64)
        blk = np.empty(cfg["T"] + 1, dtype=np.int64)
        for t in range(cfg["T"] + 1):
            comp[t] = comp_count_periodic(grid)
            blk[t] = block_count(grid, cfg["B_COARSE"])
            if t < cfg["T"]:
                grid = gol_step(grid)

        rep_de[i] = float(comp[cfg["T_EARLY"]] - comp[0])
        net_F = int(comp[cfg["T"]] - comp[0])
        net_C = int(blk[cfg["T"]] - blk[0])
        c0_F = max(int(comp[0]), 1)
        c0_C = max(int(blk[0]), 1)
        rep_G[i] = net_C / c0_C - net_F / c0_F

    rep_pr, _ = partial_r(rep_de, rep_G, rep_rho)
    save_json({
        "replication_seed": rep_seed,
        "replication_partial_r": rep_pr,
    }, os.path.join(DATA_DIR, "study_A_replication.json"))
    print(f"  Study A replication (seed {rep_seed}): partial r = {rep_pr:.3f}")

    return {
        "df": df,
        "stats": {
            "r_main": r_main,
            "partial_r": pr,
            "partial_r_ci": [pr_lo, pr_hi],
            "replication_partial_r": rep_pr,
            "verdict": verdict,
        },
    }


# =============================================================================
# STUDY B
# =============================================================================

def ts_features_10(counts, t0, t1, density):
    seg = counts[t0:t1 + 1].astype(float)
    if len(seg) < 2:
        return np.zeros(10)
    d = np.diff(seg)
    births = np.maximum(0, d)
    deaths = np.maximum(0, -d)
    ts = np.arange(len(seg), dtype=float)
    slope = float(np.polyfit(ts, seg, 1)[0])
    half = len(d) // 2
    accel = (float(d[half:].mean()) - float(d[:half].mean())) if half > 0 else 0.0
    n3 = len(d) // 3
    settling = (float(np.std(d[:n3]) + 1e-9) / float(np.std(d[-n3:]) + 1e-9)) if n3 > 0 else 1.0
    bd = float(births.sum()) / (float(deaths.sum()) + 1e-9)
    return np.array([
        float(d.sum()), slope, float(np.std(d)),
        float(seg.mean()), float(seg[0]), float(seg[-1]),
        accel, bd, settling, density
    ], dtype=float)


def static_features_9(grid0, B, density):
    occ = block_grid(grid0, B).astype(float) if B > 1 else grid0.astype(float)
    occ_frac = float(occ.mean())
    row_var = float(occ.sum(axis=1).var())
    col_var = float(occ.sum(axis=0).var())
    nbr_occ = (np.roll(occ, 1, 0) + np.roll(occ, -1, 0) + np.roll(occ, 1, 1) + np.roll(occ, -1, 1))
    same = float((occ * nbr_occ).sum())
    mx = float(4 * occ.sum())
    autocorr = same / mx if mx > 0 else 0.0
    start_count = float(occ.sum())

    if B == 1:
        live = int(grid0.sum())
        nc = comp_count_periodic(grid0)
        norm_cc = nc / max(live, 1)
    else:
        norm_cc = 0.0

    return np.array([occ_frac, norm_cc, autocorr, row_var, col_var, start_count, density, 0.0, 0.0], dtype=float)


def bootstrap_peak_B(X_dict, y, B_list, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    point_r2 = [ridge_cv_r2(X_dict[B], y) for B in B_list]
    point_peak = B_list[int(np.nanargmax(point_r2))]
    peaks = np.empty(n_boot, dtype=int)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals = [ridge_cv_r2(X_dict[B][idx], y[idx]) for B in B_list]
        peaks[i] = B_list[int(np.nanargmax(vals))]

    ci_lo = int(np.percentile(peaks, 2.5))
    ci_hi = int(np.percentile(peaks, 97.5))
    counts = {B: int((peaks == B).sum()) for B in B_list}
    return point_peak, ci_lo, ci_hi, point_r2, counts


def run_study_B():
    cfg = {
        "SEED": 20260326,
        "N": 500,
        "GRID": 128,
        "T": 250,
        "B_LIST": [1, 2, 4, 8, 16],
        "RHO_MIN": 0.20,
        "RHO_MAX": 0.35,
        "PRED_WIN": (50, 100),
        "TGT_WIN": (100, 200),
        "RIDGE_A": 1.0,
        "CV": 5,
        "N_BOOT": 1000,
    }

    rng = np.random.default_rng(cfg["SEED"])
    world_seeds = rng.integers(0, 2**62, size=cfg["N"])
    rhos = rng.uniform(cfg["RHO_MIN"], cfg["RHO_MAX"], cfg["N"])

    feat_rows = []
    tgt_rows = []

    for i in tqdm(range(cfg["N"]), desc="Study B"):
        wrng = np.random.default_rng(int(world_seeds[i]))
        grid = (wrng.random((cfg["GRID"], cfg["GRID"])) < rhos[i]).astype(np.uint8)

        comp = np.empty(cfg["T"] + 1, dtype=np.int64)
        live = np.empty(cfg["T"] + 1, dtype=np.int64)
        blk = {B: np.empty(cfg["T"] + 1, dtype=np.int64) for B in cfg["B_LIST"]}

        g = grid.copy()
        for t in range(cfg["T"] + 1):
            comp[t] = comp_count_periodic(g)
            live[t] = int(g.sum())
            for B in cfg["B_LIST"]:
                blk[B][t] = block_count(g, B)
            if t < cfg["T"]:
                g = gol_step(g)

        feat_row = {"world_id": i, "world_seed": int(world_seeds[i]), "density": float(rhos[i])}
        for B in cfg["B_LIST"]:
            counts = comp if B == 1 else blk[B]
            f_static = static_features_9(grid, B, float(rhos[i]))
            f_early = ts_features_10(counts, 0, 50, float(rhos[i]))
            f_post = ts_features_10(counts, 50, 100, float(rhos[i]))
            for k, v in enumerate(f_static):
                feat_row[f"B{B}_static_f{k}"] = v
            for k, v in enumerate(f_early):
                feat_row[f"B{B}_early_f{k}"] = v
            for k, v in enumerate(f_post):
                feat_row[f"B{B}_posttrans_f{k}"] = v
        feat_rows.append(feat_row)

        live_fut = live[100:201].astype(float)
        occ8_fut = blk[8][100:201].astype(float)
        comp_fut = comp[100:201].astype(float)

        Nlive = float(live_fut.mean())
        Nocc8 = float(occ8_fut.mean())
        A = float(np.abs(np.diff(live_fut)).mean())
        Hf = float(np.diff(comp_fut.astype(np.int64)).sum())
        Hc = float(np.diff(occ8_fut.astype(np.int64)).sum())
        Gfuture = Hc / max(float(occ8_fut[0]), 1.0) - Hf / max(float(comp_fut[0]), 1.0)

        tgt_rows.append({
            "world_id": i,
            "world_seed": int(world_seeds[i]),
            "density": float(rhos[i]),
            "Nlive": Nlive,
            "Nocc8": Nocc8,
            "A": A,
            "Gfuture": Gfuture,
        })

    df_feat = pd.DataFrame(feat_rows)
    df_tgt = pd.DataFrame(tgt_rows)
    df_feat.to_csv(os.path.join(DATA_DIR, "study_B_features.csv"), index=False)
    df_tgt.to_csv(os.path.join(DATA_DIR, "study_B_targets.csv"), index=False)

    def feat_matrix(B, wlabel):
        cols = [c for c in df_feat.columns if c.startswith(f"B{B}_{wlabel}_")]
        return df_feat[cols].values.astype(float)

    targets = ["Nlive", "Nocc8", "A", "Gfuture"]
    windows = ["static", "early", "posttrans"]
    r2_table = {t: {w: {} for w in windows} for t in targets}

    for tgt in targets:
        y = df_tgt[tgt].values.astype(float)
        for w in windows:
            for B in cfg["B_LIST"]:
                r2_table[tgt][w][B] = ridge_cv_r2(feat_matrix(B, w), y, alpha=cfg["RIDGE_A"], cv=cfg["CV"])

    # Negative controls
    primary = ["Nlive", "Nocc8"]
    audit_rows = []
    for tgt in primary:
        for w in ["static", "early"]:
            for B in cfg["B_LIST"]:
                val = r2_table[tgt][w][B]
                audit_rows.append({"target": tgt, "window": w, "B": B, "R2": val, "positive": val > 0.04})
    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(os.path.join(TAB_DIR, "table3_studyB_negative_controls.csv"), index=False)
    n_pos = int(df_audit["positive"].sum())

    peak_results = {}
    for tgt in primary:
        y = df_tgt[tgt].values.astype(float)
        X_dict = {B: feat_matrix(B, "posttrans") for B in cfg["B_LIST"]}
        peak_results[tgt] = bootstrap_peak_B(X_dict, y, cfg["B_LIST"], n_boot=cfg["N_BOOT"], seed=cfg["SEED"])

    p1, lo1, hi1 = peak_results["Nlive"][:3]
    p2, lo2, hi2 = peak_results["Nocc8"][:3]
    overlap = not (hi1 < lo2 or hi2 < lo1)
    large_small = ((p1 <= 2 and p2 >= 4) or (p2 <= 2 and p1 >= 4))
    verdict = "PASS — target-relative scale structure earned" if ((not overlap) and large_small and n_pos == 0) else (
        "NARROW PASS — peaks differ but overlap or controls imperfect" if (p1 != p2) else
        "FAIL — target-relative scale structure not earned"
    )

    save_json({
        "r2_posttrans": {t: r2_table[t]["posttrans"] for t in targets},
        "r2_static": {t: r2_table[t]["static"] for t in targets},
        "r2_early": {t: r2_table[t]["early"] for t in targets},
        "negative_control_n_positive": n_pos,
        "peak_B": {
            t: {
                "point": peak_results[t][0],
                "ci_lo": peak_results[t][1],
                "ci_hi": peak_results[t][2],
                "r2_by_B": dict(zip(cfg["B_LIST"], peak_results[t][3])),
            }
            for t in primary
        },
        "kill_test_verdict": verdict,
        "kill_test_detail": {
            "Nlive_peak": p1, "Nocc8_peak": p2,
            "Nlive_ci": [lo1, hi1], "Nocc8_ci": [lo2, hi2],
            "overlap": overlap, "large_small": large_small,
            "negative_controls_clean": (n_pos == 0),
        }
    }, os.path.join(DATA_DIR, "study_B_stats.json"))

    # Tables
    rows = []
    for tgt in targets:
        row = {"target": tgt}
        for B in cfg["B_LIST"]:
            row[f"B{B}"] = round(r2_table[tgt]["posttrans"][B], 3)
        rows.append(row)
    df_t1 = pd.DataFrame(rows)
    df_t1.to_csv(os.path.join(TAB_DIR, "table2_studyB_posttrans_r2.csv"), index=False)

    df_peak = pd.DataFrame([
        {"target": "Nlive", "peak_B": p1, "ci_lo": lo1, "ci_hi": hi1},
        {"target": "Nocc8", "peak_B": p2, "ci_lo": lo2, "ci_hi": hi2},
    ])
    df_peak.to_csv(os.path.join(TAB_DIR, "table4_studyB_peak_summary.csv"), index=False)

    # Figure 3
    label_map = {
        "Nlive": r"$\bar{N}_{\mathrm{live}}$",
        "Nocc8": r"$\bar{N}^{B=8}_{\mathrm{occ}}$",
        "A": r"$\bar{A}$",
        "Gfuture": r"$G_{\mathrm{future}}$",
    }
    colors = dict(zip(targets, sns.color_palette("tab10", 4)))

    fig_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, wlabel, title in zip(
        axes,
        ["static", "early", "posttrans"],
        ["Static IC", "Early [0,50]", "Post-transient [50,100]"]
    ):
        x = np.arange(len(cfg["B_LIST"]))
        width = 0.18
        for j, tgt in enumerate(targets):
            vals = [r2_table[tgt][wlabel][B] for B in cfg["B_LIST"]]
            ax.bar(
                x + (j - 1.5) * width,
                vals,
                width,
                color=colors[tgt],
                alpha=0.85,
                label=label_map[tgt] if ax is axes[2] else None,
            )
            for B, v in zip(cfg["B_LIST"], vals):
                fig_rows.append({"target": tgt, "window": wlabel, "B": B, "R2": v})
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(cfg["B_LIST"])
        ax.set_xlabel("Block size B")
        ax.set_title(title)
        if ax is axes[0]:
            ax.set_ylabel("5-fold CV $R^2$")
        if ax is axes[2]:
            ax.legend(fontsize=8, loc="upper right")
    pd.DataFrame(fig_rows).to_csv(os.path.join(DATA_DIR, "fig3_studyB_windows_source.csv"), index=False)
    savefig(fig, "fig3_studyB_windows")

    # Figure 4
    fig4_rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"Nlive": "o", "Nocc8": "s", "A": "^", "Gfuture": "D"}
    for tgt in targets:
        vals = [r2_table[tgt]["posttrans"][B] for B in cfg["B_LIST"]]
        sns.lineplot(
            x=cfg["B_LIST"],
            y=vals,
            marker=markers[tgt],
            linewidth=2.2,
            markersize=8,
            ax=ax,
            label=label_map[tgt],
            color=colors[tgt],
        )
        for B, v in zip(cfg["B_LIST"], vals):
            fig4_rows.append({"target": tgt, "B": B, "R2": v})
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_xticks(cfg["B_LIST"])
    ax.set_xlabel("Block size B")
    ax.set_ylabel("5-fold CV $R^2$")
    ax.set_title("Study B — R² vs B, post-transient predictor")
    ax.legend(fontsize=9)
    pd.DataFrame(fig4_rows).to_csv(os.path.join(DATA_DIR, "fig4_studyB_r2_vs_B_source.csv"), index=False)
    savefig(fig, "fig4_studyB_r2_vs_B")

    return {
        "df_feat": df_feat,
        "df_tgt": df_tgt,
        "stats": {
            "verdict": verdict,
            "negative_control_n_positive": n_pos,
            "Nlive_peak": [p1, lo1, hi1],
            "Nocc8_peak": [p2, lo2, hi2],
        }
    }


# =============================================================================
# STUDY C
# =============================================================================

def fit_model(df: pd.DataFrame, y_col: str, x_cols):
    dat = df[[y_col] + list(x_cols)].dropna().copy()
    y = dat[y_col].values.astype(float)
    X = dat[list(x_cols)].values.astype(float)
    mod = LinearRegression().fit(X, y)
    yhat = mod.predict(X)
    r2 = r2_score_manual(y, yhat)
    n = len(y)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan
    return {
        "model": "+".join(x_cols),
        "n": int(n),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "intercept": float(mod.intercept_),
        "coef_json": json.dumps({c: float(v) for c, v in zip(x_cols, mod.coef_)})
    }


def mediation_product(df: pd.DataFrame, n_boot=1000, seed=0):
    dat = df[["density", "iso_embedded_50", "fine_net_50_250", "G_0_250"]].dropna().copy()

    X_total = dat[["density", "iso_embedded_50"]].values.astype(float)
    y_total = dat["G_0_250"].values.astype(float)
    mod_total = LinearRegression().fit(X_total, y_total)
    total_eff = float(mod_total.coef_[1])

    X_med = dat[["density", "iso_embedded_50"]].values.astype(float)
    y_med = dat["fine_net_50_250"].values.astype(float)
    mod_med = LinearRegression().fit(X_med, y_med)
    a_path = float(mod_med.coef_[1])

    X_out = dat[["density", "iso_embedded_50", "fine_net_50_250"]].values.astype(float)
    y_out = dat["G_0_250"].values.astype(float)
    mod_out = LinearRegression().fit(X_out, y_out)
    direct_eff = float(mod_out.coef_[1])
    b_path = float(mod_out.coef_[2])

    indirect = a_path * b_path
    frac = indirect / total_eff if abs(total_eff) > 1e-12 else np.nan

    rng = np.random.default_rng(seed)
    vals = dat.values.astype(float)
    n = len(vals)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = vals[idx]
        d_density = sub[:, 0]
        d_emb = sub[:, 1]
        d_fine = sub[:, 2]
        d_G = sub[:, 3]

        mt = LinearRegression().fit(np.column_stack([d_density, d_emb]), d_G)
        total_b = float(mt.coef_[1])

        mm = LinearRegression().fit(np.column_stack([d_density, d_emb]), d_fine)
        a_b = float(mm.coef_[1])

        mo = LinearRegression().fit(np.column_stack([d_density, d_emb, d_fine]), d_G)
        b_b = float(mo.coef_[2])

        ind_b = a_b * b_b
        boots[i] = ind_b / total_b if abs(total_b) > 1e-12 else np.nan

    boots = boots[np.isfinite(boots)]
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    verdict = "PASS" if lo > 0.50 else ("NARROW PASS" if frac > 0.50 else "FAIL")

    return {
        "total_effect_embedded": total_eff,
        "a_path_embedded_to_finenet": a_path,
        "b_path_finenet_to_G": b_path,
        "direct_effect_embedded": direct_eff,
        "indirect_effect": float(indirect),
        "mediation_fraction": float(frac),
        "boot_ci95_lo": lo,
        "boot_ci95_hi": hi,
        "verdict": verdict,
    }


def run_study_C():
    cfg = {
        "SEED": 20260327,
        "N": 1000,
        "GRID": 64,
        "T0": 50,
        "T1": 250,
        "RHO_MIN": 0.03,
        "RHO_MAX": 0.58,
        "B_COARSE": 8,
        "N_BOOT": 1000,
    }

    rng = np.random.default_rng(cfg["SEED"])
    world_seeds = rng.integers(0, 2**62, size=cfg["N"])
    rhos = rng.uniform(cfg["RHO_MIN"], cfg["RHO_MAX"], cfg["N"])

    rows = []
    for i in tqdm(range(cfg["N"]), desc="Study C"):
        wrng = np.random.default_rng(int(world_seeds[i]))
        grid = (wrng.random((cfg["GRID"], cfg["GRID"])) < rhos[i]).astype(np.uint8)

        comp = np.empty(cfg["T1"] + 1, dtype=np.int64)
        blk = np.empty(cfg["T1"] + 1, dtype=np.int64)

        iso_emb = None
        iso_aln = None
        comp_T0 = None

        g = grid.copy()
        for t in range(cfg["T1"] + 1):
            comp[t] = comp_count_periodic(g)
            blk[t] = block_count(g, cfg["B_COARSE"])
            if t == cfg["T0"]:
                ic = isolated_counts(g)
                iso_emb = ic["embedded"]
                iso_aln = ic["alone"]
                comp_T0 = int(comp[t])
            if t < cfg["T1"]:
                g = gol_step(g)

        fine_net = int(comp[cfg["T1"]] - comp[cfg["T0"]])
        net_F = int(comp[cfg["T1"]] - comp[0])
        net_C = int(blk[cfg["T1"]] - blk[0])
        G = net_C / max(int(blk[0]), 1) - net_F / max(int(comp[0]), 1)

        rows.append({
            "world_id": i,
            "world_seed": int(world_seeds[i]),
            "density": float(rhos[i]),
            "iso_embedded_50": iso_emb,
            "iso_alone_50": iso_aln,
            "fine_net_50_250": fine_net,
            "G_0_250": G,
            "comp0": int(comp[0]),
            "comp50": comp_T0,
            "blk8_0": int(blk[0]),
            "blk8_50": int(blk[cfg["T0"]]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "study_C_worlds.csv"), index=False)

    emb = df["iso_embedded_50"].values.astype(float)
    aln = df["iso_alone_50"].values.astype(float)
    fine = df["fine_net_50_250"].values.astype(float)
    G = df["G_0_250"].values.astype(float)
    rho = df["density"].values.astype(float)

    r_emb_G, p_emb_G = pearson(emb, G)
    r_aln_G, p_aln_G = pearson(aln, G)
    r_emb_fine, p_emb_fine = pearson(emb, fine)
    r_fine_G, p_fine_G = pearson(fine, G)

    pr_emb_G, _ = partial_r(emb, G, rho)
    pr_aln_G, _ = partial_r(aln, G, rho)
    pr_emb_fine, _ = partial_r(emb, fine, rho)

    std_df = standardize_cols(df, ["density", "iso_embedded_50", "iso_alone_50", "fine_net_50_250"])
    model_specs = {
        "M1": ["density"],
        "M2": ["density", "iso_alone_50"],
        "M3": ["density", "iso_embedded_50"],
        "M4": ["density", "iso_embedded_50", "iso_alone_50"],
        "M5": ["density", "iso_embedded_50", "fine_net_50_250"],
        "M6": ["density", "iso_alone_50", "fine_net_50_250"],
        "M7": ["density", "iso_embedded_50", "iso_alone_50", "fine_net_50_250"],
    }
    model_rows = []
    for k, cols in model_specs.items():
        res = fit_model(std_df, "G_0_250", cols)
        res["model_id"] = k
        model_rows.append(res)
    df_models = pd.DataFrame(model_rows)
    df_models.to_csv(os.path.join(TAB_DIR, "table5_studyC_models.csv"), index=False)

    med = mediation_product(df, n_boot=cfg["N_BOOT"], seed=cfg["SEED"])
    save_json(med, os.path.join(DATA_DIR, "study_C_mediation.json"))

    verdict = "FAIL — simple mediation through fine_net not earned" if med["verdict"] == "FAIL" else med["verdict"]

    # Figure 5
    fig5_source = pd.DataFrame({
        "iso_embedded_50": df["iso_embedded_50"],
        "iso_alone_50": df["iso_alone_50"],
        "G_0_250": df["G_0_250"],
        "density": df["density"],
    })
    fig5_source.to_csv(os.path.join(DATA_DIR, "fig5_studyC_embedded_vs_alone_source.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    sc0 = axes[0].scatter(df["iso_embedded_50"], df["G_0_250"], c=df["density"], cmap="plasma", s=8, alpha=0.55, rasterized=True)
    o = ols(df["iso_embedded_50"], df["G_0_250"])
    xl = np.linspace(df["iso_embedded_50"].min(), df["iso_embedded_50"].max(), 200)
    axes[0].plot(xl, o["slope"] * xl + o["intercept"], "k--", lw=1.4)
    axes[0].set_title(f"Embedded vs G\nr={r_emb_G:.3f}, partial r|ρ={pr_emb_G:.3f}")
    axes[0].set_xlabel("iso_embedded_50")
    axes[0].set_ylabel("G_0_250")

    axes[1].scatter(df["iso_alone_50"], df["G_0_250"], c=df["density"], cmap="plasma", s=8, alpha=0.55, rasterized=True)
    o = ols(df["iso_alone_50"], df["G_0_250"])
    xl = np.linspace(df["iso_alone_50"].min(), df["iso_alone_50"].max(), 200)
    axes[1].plot(xl, o["slope"] * xl + o["intercept"], "k--", lw=1.4)
    axes[1].set_title(f"Alone vs G\nr={r_aln_G:.3f}, partial r|ρ={pr_aln_G:.3f}")
    axes[1].set_xlabel("iso_alone_50")

    cbar = fig.colorbar(sc0, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("Initial density ρ")
    savefig(fig, "fig5_studyC_embedded_vs_alone")

    save_json({
        "pairwise": {
            "r_embedded_G": r_emb_G,
            "p_embedded_G": p_emb_G,
            "r_alone_G": r_aln_G,
            "p_alone_G": p_aln_G,
            "r_embedded_finenet": r_emb_fine,
            "p_embedded_finenet": p_emb_fine,
            "r_finenet_G": r_fine_G,
            "p_finenet_G": p_fine_G,
            "partial_r_embedded_G": pr_emb_G,
            "partial_r_alone_G": pr_aln_G,
            "partial_r_embedded_finenet": pr_emb_fine,
        },
        "mediation": med,
        "verdict": verdict,
    }, os.path.join(DATA_DIR, "study_C_stats.json"))

    return {
        "df": df,
        "models": df_models,
        "mediation": med,
        "stats": {
            "verdict": verdict,
            "r_embedded_G": r_emb_G,
            "r_alone_G": r_aln_G,
            "partial_r_embedded_G": pr_emb_G,
            "partial_r_alone_G": pr_aln_G,
        },
    }


# =============================================================================
# STUDY D / D2
# =============================================================================

def run_study_D_and_D2():
    cfg = {
        "BASE_SEED": 20260328,
        "T0": 50,
        "T1": 250,
        "CONDITIONS": [
            {"cond_id": "G64_low",   "grid": 64,  "rho_lo": 0.10, "rho_hi": 0.20, "n_worlds": 400},
            {"cond_id": "G64_mid",   "grid": 64,  "rho_lo": 0.25, "rho_hi": 0.35, "n_worlds": 400},
            {"cond_id": "G64_high",  "grid": 64,  "rho_lo": 0.35, "rho_hi": 0.50, "n_worlds": 400},
            {"cond_id": "G128_low",  "grid": 128, "rho_lo": 0.10, "rho_hi": 0.20, "n_worlds": 300},
            {"cond_id": "G128_mid",  "grid": 128, "rho_lo": 0.25, "rho_hi": 0.35, "n_worlds": 300},
            {"cond_id": "G128_high", "grid": 128, "rho_lo": 0.35, "rho_hi": 0.50, "n_worlds": 300},
        ],
    }

    world_rows = []
    sum_rows = []

    for idx, cond in enumerate(cfg["CONDITIONS"]):
        rng = np.random.default_rng(cfg["BASE_SEED"] + idx)
        per_world = []

        for world_id in tqdm(range(cond["n_worlds"]), desc=cond["cond_id"]):
            rho = float(rng.uniform(cond["rho_lo"], cond["rho_hi"]))
            world_seed = int(rng.integers(0, 2**63 - 1))
            wrng = np.random.default_rng(world_seed)
            grid = (wrng.random((cond["grid"], cond["grid"])) < rho).astype(np.uint8)

            comp_t0 = None
            comp_t1 = None
            beta_local_world = np.nan
            iso_embedded_t0 = 0
            embedded_survive_prob = np.nan

            g = grid.copy()
            for t in range(cfg["T1"] + 1):
                if t == cfg["T0"]:
                    grid_t0 = g.copy()
                    comp_t0 = comp_count_periodic(grid_t0)
                    emb_coords = embedded_isolated_coords(grid_t0)
                    iso_embedded_t0 = len(emb_coords)

                    next_g = gol_step(grid_t0)
                    if iso_embedded_t0 > 0:
                        survive = sum(int(next_g[r, c] == 1) for r, c in emb_coords)
                        embedded_survive_prob = survive / iso_embedded_t0
                        vals = [dynamic_local_delta_for_focal(grid_t0, r, c, core_radius=2) for r, c in emb_coords]
                        beta_local_world = float(np.mean(vals))

                if t == cfg["T1"]:
                    comp_t1 = comp_count_periodic(g)
                    break

                g = gol_step(g)

            fine_net = int(comp_t1 - comp_t0)

            per_world.append({
                "cond_id": cond["cond_id"],
                "grid": cond["grid"],
                "rho": rho,
                "world_id": world_id,
                "seed": world_seed,
                "iso_embedded_T0": iso_embedded_t0,
                "fine_net_T0_T1": fine_net,
                "beta_local_world": beta_local_world,
                "embedded_survive_prob": embedded_survive_prob,
            })

        dfw = pd.DataFrame(per_world)
        world_rows.append(dfw)

        o = ols(dfw["iso_embedded_T0"], dfw["fine_net_T0_T1"])
        beta_emp = o["slope"]
        beta_death = -1.0
        beta_local = float(np.nanmean(dfw["beta_local_world"].values))
        beta_residual = float(beta_emp - beta_death - beta_local)
        denom = abs(beta_death + beta_local)
        chi = float(beta_residual / denom) if denom > 1e-12 else np.nan

        sum_rows.append({
            "cond_id": cond["cond_id"],
            "grid": cond["grid"],
            "rho_lo": cond["rho_lo"],
            "rho_hi": cond["rho_hi"],
            "beta_emp": beta_emp,
            "beta_death": beta_death,
            # beta_local is a mean local one-step structural contribution,
            # not a regression coefficient
            "beta_local": beta_local,
            "beta_residual": beta_residual,
            "chi": chi,
            "r2": o["r2"],
            "slope_ci95_lo": o["ci95_lo"],
            "slope_ci95_hi": o["ci95_hi"],
            "mean_embedded_survive_prob": float(np.nanmean(dfw["embedded_survive_prob"].values)),
        })

    df_world = pd.concat(world_rows, ignore_index=True)
    df_sum = pd.DataFrame(sum_rows)
    df_world.to_csv(os.path.join(DATA_DIR, "study_D_worlds.csv"), index=False)
    df_sum.to_csv(os.path.join(TAB_DIR, "table6_studyD_condition_summary.csv"), index=False)

    beta_emp_mean = float(df_sum["beta_emp"].mean())
    beta_emp_sd = float(df_sum["beta_emp"].std(ddof=0))
    beta_emp_cv = abs(beta_emp_sd / beta_emp_mean)
    chi_mean = float(np.nanmean(df_sum["chi"]))
    chi_min = float(np.nanmin(df_sum["chi"]))

    verdict_emp = "PASS — empirical slope law earned" if beta_emp_cv <= 0.20 else "FAIL — unstable slope law"
    verdict_resid = "NOT EARNED — residual-survival claim fails under dynamic local null"

    save_json({
        "beta_emp_mean": beta_emp_mean,
        "beta_emp_sd": beta_emp_sd,
        "beta_emp_cv": beta_emp_cv,
        "chi_mean": chi_mean,
        "chi_min": chi_min,
        "empirical_verdict": verdict_emp,
        "residual_verdict": verdict_resid,
    }, os.path.join(DATA_DIR, "study_D_stats.json"))

    # Figure 6
    fig6_source = df_sum.copy()
    fig6_source.to_csv(os.path.join(DATA_DIR, "fig6_studyD_slope_summary_source.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(df_sum))
    ax.errorbar(
        df_sum["beta_emp"].values,
        y,
        xerr=[
            df_sum["beta_emp"].values - df_sum["slope_ci95_lo"].values,
            df_sum["slope_ci95_hi"].values - df_sum["beta_emp"].values,
        ],
        fmt="o",
        capsize=4,
    )
    ax.axvline(beta_emp_mean, color="red", ls="--", lw=1.2, label=f"mean slope = {beta_emp_mean:.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels(df_sum["cond_id"].tolist())
    ax.set_xlabel(r"OLS slope: $\mathrm{fine\_net} \sim \mathrm{iso\_embedded}$")
    ax.set_ylabel("Condition")
    ax.set_title(f"Study D — slope summary, CV = {beta_emp_cv:.3f}")
    ax.legend()
    savefig(fig, "fig6_studyD_slope_summary")

    # Figure 7
    long_rows = []
    for _, row in df_sum.iterrows():
        for term in ["beta_emp", "beta_death", "beta_local", "beta_residual"]:
            long_rows.append({"cond_id": row["cond_id"], "term": term, "value": row[term]})
    df_long = pd.DataFrame(long_rows)

    term_map = {
        "beta_emp": r"$\beta_{\mathrm{emp}}$",
        "beta_death": r"$\beta_{\mathrm{death}}$",
        "beta_local": r"$\beta_{\mathrm{local}}$",
        "beta_residual": r"$\beta_{\mathrm{residual}}$",
    }
    df_long["term_label"] = df_long["term"].map(term_map)
    fig7_source = df_sum.copy()
    fig7_source.to_csv(os.path.join(DATA_DIR, "fig7_studyD_decomposition_source.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=df_long, x="cond_id", y="value", hue="term_label", ax=axes[0])
    axes[0].axhline(0, color="gray", lw=0.6, ls=":")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Study D dynamic-null decomposition")
    axes[0].legend(title="")

    sns.barplot(data=df_sum, x="cond_id", y="chi", ax=axes[1], color=sns.color_palette("deep")[0])
    axes[1].axhline(0.25, color="red", ls="--", lw=1.2, label=r"$\chi = 0.25$")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_xlabel("")
    axes[1].set_ylabel(r"$\chi$")
    axes[1].set_title("Residual-survival index by condition")
    axes[1].legend()

    savefig(fig, "fig7_studyD_decomposition")

    return {
        "df_sum": df_sum,
        "stats": {
            "beta_emp_mean": beta_emp_mean,
            "beta_emp_cv": beta_emp_cv,
            "chi_mean": chi_mean,
            "chi_min": chi_min,
            "empirical_verdict": verdict_emp,
            "residual_verdict": verdict_resid,
        },
    }


# =============================================================================
# MASTER PACKAGE
# =============================================================================

def write_master_adjudication(A, B, C, D):
    lines = [
        "# CA Adjudication Note\n\n",
        "## Locked branch state\n\n",
        f"- Study A: {A['stats']['verdict']}\n",
        f"  - r_main ≈ {A['stats']['r_main']:.3f}\n",
        f"  - partial r|ρ ≈ {A['stats']['partial_r']:.3f}\n",
        f"  - replication (seed 20260326) partial r|ρ ≈ {A['stats']['replication_partial_r']:.3f}\n\n",
        f"- Study B: {B['stats']['verdict']}\n",
        f"  - negative controls positive count = {B['stats']['negative_control_n_positive']}\n",
        f"  - Nlive peak = {B['stats']['Nlive_peak']}\n",
        f"  - Nocc8 peak = {B['stats']['Nocc8_peak']}\n\n",
        f"- Study C: {C['stats']['verdict']}\n",
        f"  - embedded vs G r ≈ {C['stats']['r_embedded_G']:.3f}\n",
        f"  - alone vs G r ≈ {C['stats']['r_alone_G']:.3f}\n\n",
        f"- Study D empirical law: {D['stats']['empirical_verdict']}\n",
        f"  - mean slope ≈ {D['stats']['beta_emp_mean']:.3f}\n",
        f"  - CV ≈ {D['stats']['beta_emp_cv']:.3f}\n\n",
        f"- Study D residual-survival: {D['stats']['residual_verdict']}\n",
        f"  - chi_mean ≈ {D['stats']['chi_mean']:.3f}\n",
        f"  - chi_min ≈ {D['stats']['chi_min']:.3f}\n",
    ]
    with open(os.path.join(LOG_DIR, "CA_adjudication_note.md"), "w") as f:
        f.writelines(lines)


def write_master_manifest():
    figs = sorted(os.listdir(FIG_DIR))
    tabs = sorted(os.listdir(TAB_DIR))
    datas = sorted(os.listdir(DATA_DIR))

    pd.DataFrame({"file": figs}).to_csv(os.path.join(LOG_DIR, "figure_manifest.csv"), index=False)
    pd.DataFrame({"file": tabs}).to_csv(os.path.join(LOG_DIR, "table_manifest.csv"), index=False)
    pd.DataFrame({"file": datas}).to_csv(os.path.join(LOG_DIR, "data_manifest.csv"), index=False)


def main():
    t0 = time.time()
    print("Running CA paper package...\n")

    A = run_study_A()
    B = run_study_B()
    C = run_study_C()
    D = run_study_D_and_D2()

    write_master_adjudication(A, B, C, D)
    write_master_manifest()

    summary = {
        "Study_A": A["stats"],
        "Study_B": B["stats"],
        "Study_C": C["stats"],
        "Study_D": D["stats"],
    }
    save_json(summary, os.path.join(LOG_DIR, "paper_package_summary.json"))

    print("\nDone.")
    print(f"Outputs: {OUT}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()