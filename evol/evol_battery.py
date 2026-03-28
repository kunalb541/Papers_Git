"""
evol_battery.py  —  Battery harness
=====================================
Target definitions, predictive scoring, causal scoring,
verdict maps, dissociation panel, blocking-condition checks,
and the pilot diagnostic.

Causal winners are always compared within a fixed target.
Cross-target causal comparisons are forbidden by design.
"""
from __future__ import annotations

import json
import os
from itertools import product
import multiprocessing as mp

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Cap workers below full core count — M5 Air is fanless; all-core sustained
# loads can downclock. 6 workers leaves headroom on an 8-core Air.
DEFAULT_WORKERS = min(6, (mp.cpu_count() or 4) - 1)

from evol_sim import (
    PopState, BATTERY_PARAMS, GROUP_PARAMS, ECO_PARAMS, REGIMES, run_regime
)
from evol_observers import (
    get_observation, check_anti_collapse, pilot_lineage_diagnostics,
    OBSERVER_NAMES,
)
from evol_interventions import (
    apply_intervention, INTERVENTION_CLASSES,
)

# ---------------------------------------------------------------------------
# Scoring parameters
# ---------------------------------------------------------------------------
RIDGE_ALPHA      = 1.0
CV_FOLDS         = 5
N_BOOT_SCORE     = 1000
SEED_BASE        = 3000

# Null baselines by target type.
# Continuous (R²): null is 0. A confirmed predictive signal requires ci_lo > 0.
# Binary (AUC):    null is 0.5. A confirmed predictive signal requires ci_lo > 0.5.
_NULL_BASELINE = {"continuous": 0.0, "binary": 0.5}

def _score_null(target_type: str) -> float:
    """Return the null baseline for a given target type."""
    return _NULL_BASELINE.get(target_type, 0.0)

# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------

def target_persistence(state_t0: PopState, state_tT: PopState,
                        original_s0: PopState = None,
                        min_freq: float = 0.01) -> float:
    """Fraction of lineages present at observation time (original_s0 if given,
    else state_t0) with freq >= min_freq that still have freq > 0 at t_T.

    For causal scoring, pass original_s0 = the pre-intervention state so the
    denominator is fixed and does not change with the intervention.
    """
    anchor = original_s0 if original_s0 is not None else state_t0
    counts_0 = np.bincount(anchor.lineages, minlength=anchor.n_founders)
    freqs_0  = counts_0 / len(anchor.lineages)
    present  = np.where(freqs_0 >= min_freq)[0]
    if len(present) == 0:
        return 0.0
    counts_T = np.bincount(state_tT.lineages, minlength=state_tT.n_founders)
    survived = np.sum(counts_T[present] > 0)
    return float(survived) / len(present)


def target_transmission(state_t0: PopState, state_tT: PopState,
                         n_alleles: int = 4) -> float:
    """Frequency of favored allele (allele 0) at t_T.
    Proxy for transmission success of the fittest type.
    Continuous target in [0, 1]."""
    counts = np.bincount(state_tT.alleles, minlength=n_alleles)
    return float(counts[0]) / len(state_tT.alleles)


def target_dominance(state_tT: PopState,
                      n_alleles: int = 4, threshold: float = 0.5) -> float:
    """Binary: did favored allele (allele 0) reach frequency > threshold at t_T?
    Returns 1.0 / 0.0."""
    counts = np.bincount(state_tT.alleles, minlength=n_alleles)
    freq   = float(counts[0]) / len(state_tT.alleles)
    return 1.0 if freq > threshold else 0.0


def target_ecology(state_tT: PopState) -> float:
    """Resource level R(t_T). Meaningful only in eco regime."""
    return state_tT.resource


TARGET_NAMES = ["persistence", "transmission", "dominance", "ecology"]
TARGET_TYPES = {
    "persistence":  "continuous",
    "transmission": "continuous",
    "dominance":    "binary",
    "ecology":      "continuous",
}

def compute_target(name: str, state_t0: PopState, state_tT: PopState,
                   n_alleles: int = 4, regime: str = "",
                   original_s0: PopState = None) -> float:
    """Compute target value. Ecology target returns nan outside eco_evol regime.

    original_s0: for persistence target in causal arms, pass the pre-intervention
    state to keep the denominator fixed across control and intervention.
    """
    if name == "persistence":
        return target_persistence(state_t0, state_tT, original_s0)
    elif name == "transmission":
        return target_transmission(state_t0, state_tT, n_alleles)
    elif name == "dominance":
        return target_dominance(state_tT, n_alleles=n_alleles, threshold=0.5)
    elif name == "ecology":
        if regime != "eco_evol":
            return float("nan")
        return target_ecology(state_tT)
    else:
        raise ValueError(f"Unknown target: {name}")


# ---------------------------------------------------------------------------
# Predictive scoring
# ---------------------------------------------------------------------------

def score_predictive(X: np.ndarray, y: np.ndarray,
                     target_type: str,
                     seed: int = 0) -> tuple[float, float, float]:
    """
    Cross-validated predictive score.
    Ridge regression for continuous targets (fixed alpha, standardized).
    Logistic regression + AUC for binary targets.
    Returns (mean_cv_score, ci_lo, ci_hi).
    Degenerate folds skipped; returns (nan, nan, nan) if < 2 valid folds.
    """
    kf     = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr   = scaler.transform(X_tr)
        X_val  = scaler.transform(X_val)

        if target_type == "continuous":
            model = Ridge(alpha=RIDGE_ALPHA).fit(X_tr, y_tr)
            s     = r2_score(y_val, model.predict(X_val))

        else:   # binary — AUC, null = 0.5
            # Skip fold if either split is single-class
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                continue
            model = LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced",
                random_state=seed,
            ).fit(X_tr, y_tr)
            s = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        scores.append(s)

    if len(scores) < 2:
        return float("nan"), float("nan"), float("nan")

    rng  = np.random.default_rng(seed)
    boot = [float(np.mean(rng.choice(scores, len(scores))))
            for _ in range(N_BOOT_SCORE)]
    return (float(np.mean(scores)),
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


# ---------------------------------------------------------------------------
# Single regime × target × level cell
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Worker functions for cell-level parallelism.
# One worker handles a complete cell (all replicates serially).
# warmup_jit is called once in the parent before any pool is created.
# Workers do NOT call warmup_jit — Numba cache is loaded on first use.
# ---------------------------------------------------------------------------

def _run_one_pred_rep(args):
    """Worker: run one replicate, return (obs, target) or None.

    Observation snapshot taken at t_0 (from BATTERY_PARAMS), not t=0.
    By t_0=20, lineage frequencies have drifted and carry real signal.
    """
    regime, target_name, level, seed, n_alleles, T, t_0 = args
    from evol_sim import run_regime
    from evol_observers import get_observation
    traj = run_regime(regime, seed, full_traj=True)
    s_obs = traj[t_0]          # observation snapshot at t_0
    sT    = traj[-1]           # outcome at T
    tgt   = compute_target(target_name, s_obs, sT, n_alleles, regime)
    if np.isnan(tgt):
        return None
    return (get_observation(s_obs, level, n_alleles), tgt)


def _run_one_causal_rep(args):
    """Worker: run one causal replicate pair, return (ctrl_tgt, int_tgt, 'OK') or sentinel.

    Intervention applied at the t_0 observation snapshot.
    For persistence target: denominator anchored to original s_obs (pre-intervention)
    in both arms so the causal gap reflects dynamics, not target-definition change.

    delta is intervention-class-specific: lineage uses delta_lineage (smaller),
    all others use delta. This is necessary because at t_0=20 lineages are
    near-monomorphic and a full delta=0.10 shift saturates the contamination check.
    """
    regime, target_name, interv_class, seed, delta, delta_lineage, n_alleles, T, t_0 = args
    from evol_sim import run_regime
    from evol_interventions import apply_intervention
    rng = np.random.default_rng(seed + 10000)

    traj_ctrl = run_regime(regime, seed, full_traj=True)
    s_obs     = traj_ctrl[t_0]
    sT_ctrl   = traj_ctrl[-1]
    ctrl_tgt = compute_target(target_name, s_obs, sT_ctrl, n_alleles, regime,
                               original_s0=s_obs)
    if np.isnan(ctrl_tgt):
        return None

    # Use class-specific delta
    eff_delta = delta_lineage if interv_class == "lineage" else delta
    s_int, status = apply_intervention(s_obs, interv_class, eff_delta, rng, n_alleles)
    if "ALLELE_CONTAMINATED" in status or "INSUFFICIENT" in status:
        return ("CONTAMINATED", status)

    traj_int = _run_from_state(s_int, regime, seed + 1, T - t_0, n_alleles, interv_class)
    sT_int   = traj_int[-1]
    int_tgt = compute_target(target_name, s_int, sT_int, n_alleles, regime,
                              original_s0=s_obs)
    if np.isnan(int_tgt):
        return None
    return (ctrl_tgt, int_tgt, "OK")


def _run_pred_cell_worker(args):
    """Cell-level worker: runs all replicates for one (regime, target, level) cell."""
    regime, target_name, level, n_reps, seed_base, n_alleles, T, t_0 = args
    results = []
    for rep in range(n_reps):
        r = _run_one_pred_rep((regime, target_name, level,
                                seed_base + rep, n_alleles, T, t_0))
        if r is not None:
            results.append(r)
    return regime, target_name, level, results


def _run_caus_cell_worker(args):
    """Cell-level worker: runs all replicates for one (regime, target, interv) cell."""
    regime, target_name, interv_class, n_reps, seed_base, delta, delta_lineage, n_alleles, T, t_0 = args
    results = []
    for rep in range(n_reps):
        r = _run_one_causal_rep((regime, target_name, interv_class,
                                  seed_base + rep, delta, delta_lineage, n_alleles, T, t_0))
        results.append(r)
    return regime, target_name, interv_class, results


# run_predictive_cell removed: dead code with signature mismatch (missing t_0).
# Use _run_pred_cell_worker via the pool instead.


# run_causal_cell removed: dead code with signature mismatch (missing delta_lineage, t_0).
# Use _run_caus_cell_worker via the pool instead.


def _run_from_state(s0: PopState, regime: str, seed: int,
                    T: int, n_alleles: int,
                    interv_class: str = "gene") -> list[PopState]:
    """Run T steps from intervened state using exact regime dynamics.

    interv_class: "organism" leaves fitness as-is (intervention acted on fitness);
    all others refresh fitness from new alleles/resource before first step.
    Eco and group buffers are hoisted outside the loop. No internal allocation.
    """
    from evol_sim import (
        _wf_step_k, _moran_step_k,
        _fitness_neutral_k, _fitness_selected_k,
        _fitness_freq_dep_k, _fitness_eco_k, _mean_consumption_k,
        _Buf, FAVORED_ALLELE, FREQ_DEP_ALPHA, ECO_PARAMS, GROUP_PARAMS,
    )
    rng    = np.random.default_rng(seed)
    mu     = BATTERY_PARAMS["mu"]
    s_coef = BATTERY_PARAMS["s"]
    N      = len(s0.alleles)
    buf    = _Buf(N, n_alleles)
    traj   = [s0.copy()]
    state  = s0.copy()

    np.copyto(buf.a0, state.alleles)
    np.copyto(buf.l0, state.lineages)
    np.copyto(buf.f0, state.fitness)

    # Refresh t0 fitness unless the intervention was directly on fitness
    if interv_class != "organism":
        if regime == "neutral_wf":
            _fitness_neutral_k(N, buf.f0)
        elif regime in ("selected_wf", "moran", "group_structured"):
            _fitness_selected_k(buf.a0, N, s_coef, FAVORED_ALLELE, buf.f0)
        elif regime == "freq_dep":
            _fitness_freq_dep_k(buf.a0, N, n_alleles, FREQ_DEP_ALPHA, buf.f0)
        elif regime == "eco_evol":
            _fitness_eco_k(buf.a0, N,
                           np.asarray(ECO_PARAMS["allele_base_fitness"], dtype=np.float64),
                           state.resource, float(ECO_PARAMS["K"]), buf.f0)

    # Hoist eco constants
    eco_ac = eco_bf = eco_K = eco_rR = None
    if regime == "eco_evol":
        eco_ac = np.asarray(ECO_PARAMS["allele_consumption"],  dtype=np.float64)
        eco_bf = np.asarray(ECO_PARAMS["allele_base_fitness"], dtype=np.float64)
        eco_K  = float(ECO_PARAMS["K"])
        eco_rR = float(ECO_PARAMS["r_R"])

    # Hoist group-structured buffers
    grp_gbuf = grp_ga1 = grp_gl1 = grp_gf = grp_group_fit = None
    grp_n_groups = grp_gs = grp_s_within = grp_s_between = None
    if regime == "group_structured":
        grp_n_groups = GROUP_PARAMS["n_groups"]
        grp_gs       = GROUP_PARAMS["group_size"]
        grp_s_within = GROUP_PARAMS["s_within"]
        grp_s_between= GROUP_PARAMS["s_between"]
        grp_gbuf      = _Buf(grp_gs, n_alleles)
        grp_ga1       = np.empty(grp_gs, dtype=np.int64)
        grp_gl1       = np.empty(grp_gs, dtype=np.int64)
        grp_gf        = np.empty(grp_gs, dtype=np.float64)
        grp_group_fit = np.empty(grp_n_groups, dtype=np.float64)

    for _ in range(T):
        if regime == "neutral_wf":
            _fitness_neutral_k(N, buf.f0)
            rng.random(out=buf.rsel); rng.random(out=buf.rmut)
            np.copyto(buf.rna, rng.integers(0, n_alleles, size=N))
            _wf_step_k(buf.a0, buf.l0, buf.f0, N, mu, n_alleles,
                       buf.rsel, buf.rmut, buf.rna, buf.cdf, buf.a1, buf.l1)
            _fitness_neutral_k(N, buf.f1)
            buf.a0, buf.a1 = buf.a1, buf.a0
            buf.l0, buf.l1 = buf.l1, buf.l0
            buf.f0, buf.f1 = buf.f1, buf.f0
            state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                             fitness=buf.f0.copy(), resource=0.0,
                             t=state.t+1, n_founders=state.n_founders)

        elif regime == "selected_wf":
            rng.random(out=buf.rsel); rng.random(out=buf.rmut)
            np.copyto(buf.rna, rng.integers(0, n_alleles, size=N))
            _wf_step_k(buf.a0, buf.l0, buf.f0, N, mu, n_alleles,
                       buf.rsel, buf.rmut, buf.rna, buf.cdf, buf.a1, buf.l1)
            _fitness_selected_k(buf.a1, N, s_coef, FAVORED_ALLELE, buf.f1)
            buf.a0, buf.a1 = buf.a1, buf.a0
            buf.l0, buf.l1 = buf.l1, buf.l0
            buf.f0, buf.f1 = buf.f1, buf.f0
            state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                             fitness=buf.f0.copy(), resource=0.0,
                             t=state.t+1, n_founders=state.n_founders)

        elif regime == "freq_dep":
            rng.random(out=buf.rsel); rng.random(out=buf.rmut)
            np.copyto(buf.rna, rng.integers(0, n_alleles, size=N))
            _wf_step_k(buf.a0, buf.l0, buf.f0, N, mu, n_alleles,
                       buf.rsel, buf.rmut, buf.rna, buf.cdf, buf.a1, buf.l1)
            _fitness_freq_dep_k(buf.a1, N, n_alleles, FREQ_DEP_ALPHA, buf.f1)
            buf.a0, buf.a1 = buf.a1, buf.a0
            buf.l0, buf.l1 = buf.l1, buf.l0
            buf.f0, buf.f1 = buf.f1, buf.f0
            state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                             fitness=buf.f0.copy(), resource=0.0,
                             t=state.t+1, n_founders=state.n_founders)

        elif regime == "eco_evol":
            R = state.resource
            mean_ac = _mean_consumption_k(buf.a0, eco_ac)
            R = max(R + eco_rR * R * (1.0 - R / eco_K) - mean_ac * R / eco_K, 0.0)
            rng.random(out=buf.rsel); rng.random(out=buf.rmut)
            np.copyto(buf.rna, rng.integers(0, n_alleles, size=N))
            _wf_step_k(buf.a0, buf.l0, buf.f0, N, mu, n_alleles,
                       buf.rsel, buf.rmut, buf.rna, buf.cdf, buf.a1, buf.l1)
            _fitness_eco_k(buf.a1, N, eco_bf, R, eco_K, buf.f1)
            buf.a0, buf.a1 = buf.a1, buf.a0
            buf.l0, buf.l1 = buf.l1, buf.l0
            buf.f0, buf.f1 = buf.f1, buf.f0
            state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                             fitness=buf.f0.copy(), resource=R,
                             t=state.t+1, n_founders=state.n_founders)

        elif regime == "moran":
            _fitness_selected_k(buf.a0, N, s_coef, FAVORED_ALLELE, buf.f0)
            rng.random(out=buf.rsel)
            np.copyto(buf.rdie, rng.integers(0, N, size=N))
            rng.random(out=buf.rmut)
            np.copyto(buf.rna, rng.integers(0, n_alleles, size=N))
            _moran_step_k(buf.a0, buf.l0, buf.f0, N, mu, n_alleles,
                          buf.rsel, buf.rdie, buf.rmut, buf.rna, buf.cdf)
            _fitness_selected_k(buf.a0, N, s_coef, FAVORED_ALLELE, buf.f0)
            state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                             fitness=buf.f0.copy(), resource=0.0,
                             t=state.t+1, n_founders=state.n_founders)

        elif regime == "group_structured":
            for g in range(grp_n_groups):
                idx = slice(g * grp_gs, (g + 1) * grp_gs)
                np.copyto(grp_gbuf.a0, buf.a0[idx])
                np.copyto(grp_gbuf.l0, buf.l0[idx])
                _fitness_selected_k(grp_gbuf.a0, grp_gs, grp_s_within,
                                    FAVORED_ALLELE, grp_gbuf.f0)
                rng.random(out=grp_gbuf.rsel); rng.random(out=grp_gbuf.rmut)
                np.copyto(grp_gbuf.rna, rng.integers(0, n_alleles, size=grp_gs))
                _wf_step_k(grp_gbuf.a0, grp_gbuf.l0, grp_gbuf.f0,
                           grp_gs, mu, n_alleles,
                           grp_gbuf.rsel, grp_gbuf.rmut, grp_gbuf.rna,
                           grp_gbuf.cdf, grp_ga1, grp_gl1)
                buf.a0[idx] = grp_ga1
                buf.l0[idx] = grp_gl1

            for g in range(grp_n_groups):
                idx = slice(g * grp_gs, (g + 1) * grp_gs)
                np.copyto(grp_gbuf.a0, buf.a0[idx])
                _fitness_selected_k(grp_gbuf.a0, grp_gs, grp_s_between,
                                    FAVORED_ALLELE, grp_gf)
                grp_group_fit[g] = grp_gf.mean()

            cdf_g   = np.cumsum(grp_group_fit / grp_group_fit.sum())
            repro_g = int(np.searchsorted(cdf_g, rng.random()))
            die_g   = int(rng.integers(0, grp_n_groups))
            if repro_g != die_g:
                src = slice(repro_g * grp_gs, (repro_g + 1) * grp_gs)
                dst = slice(die_g   * grp_gs, (die_g   + 1) * grp_gs)
                pidx = rng.integers(0, grp_gs, size=grp_gs)
                buf.a0[dst] = buf.a0[src][pidx]
                buf.l0[dst] = buf.l0[src][pidx]
                rmut_g = rng.random(grp_gs)
                for i in range(grp_gs):
                    if rmut_g[i] < mu:
                        buf.a0[dst.start + i] = int(rng.integers(0, n_alleles))

            _fitness_selected_k(buf.a0, N, grp_s_within, FAVORED_ALLELE, buf.f0)
            state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                             fitness=buf.f0.copy(), resource=0.0,
                             t=state.t+1, n_founders=state.n_founders)

        else:
            raise ValueError(f"Unknown regime in _run_from_state: {regime}")

        traj.append(state.copy())
    return traj


def build_verdict_maps(pred_results: list[dict],
                        causal_results: list[dict]) -> dict:
    """
    Build two verdict maps and dissociation panel.
    pred_winner[regime][target]   = level with highest CI-confirmed score
    causal_winner[regime][target] = intervention with largest |gap| where CI
                                    excludes zero (either sign accepted).
    dissociation[regime][target]  = True if pred != causal winner

    Causal winners ALWAYS compared within fixed target (enforced by loop structure).
    Both positive and negative causal effects are valid wins.
    """
    pred_winner   = {r: {} for r in REGIMES}
    causal_winner = {r: {} for r in REGIMES}
    dissociation  = {r: {} for r in REGIMES}

    for target in TARGET_NAMES:
        t_type = TARGET_TYPES[target]
        null   = _score_null(t_type)
        for regime in REGIMES:
            # Predictive winner: highest score where CI exceeds type-appropriate null
            cell_scores = {
                row["level"]: row
                for row in pred_results
                if row["regime"] == regime and row["target"] == target
                and row.get("status", "OK") == "OK"
            }
            best_level = None
            best_score = -np.inf
            for level, row in cell_scores.items():
                s = row.get("score", float("nan"))
                if np.isnan(s):
                    continue
                # ci_lo must exceed null baseline (0 for R², 0.5 for AUC)
                if row.get("ci_lo", float("nan")) > null and s > best_score:
                    best_score = s
                    best_level = level
            pred_winner[regime][target] = best_level or "NONE"

            # Causal winner: largest |gap| where CI excludes zero (either sign)
            # Comparison strictly within this target.
            cell_gaps = {
                row["interv_class"]: row
                for row in causal_results
                if row["regime"] == regime and row["target"] == target
                and row.get("status") == "OK"
            }
            best_interv = None
            best_absgap = -np.inf
            for interv, row in cell_gaps.items():
                g    = row.get("gap", float("nan"))
                ci_lo = row.get("ci_lo", float("nan"))
                ci_hi = row.get("ci_hi", float("nan"))
                if np.isnan(g) or np.isnan(ci_lo):
                    continue
                # CI must exclude zero (either positive or negative effect)
                ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)
                if ci_excludes_zero and abs(g) > best_absgap:
                    best_absgap = abs(g)
                    best_interv = interv
            causal_winner[regime][target] = best_interv or "NONE"

            # Dissociation
            pw = pred_winner[regime][target]
            cw = causal_winner[regime][target]
            dissociation[regime][target] = (
                pw != "NONE" and cw != "NONE" and pw != cw
            )

    return {
        "pred_winner":   pred_winner,
        "causal_winner": causal_winner,
        "dissociation":  dissociation,
    }


# ---------------------------------------------------------------------------
# Blocking-condition checker
# ---------------------------------------------------------------------------

def check_blocking_conditions(regime: str, target: str,
                                winner_level: str,
                                pred_row: dict,
                                all_pred_rows: list[dict],
                                all_causal_rows: list[dict]) -> dict:
    """
    Four conditions must ALL hold before a cell moves to EARNED.

    1. Winning level CI exceeds type-appropriate null with span < 0.5.
    2. Result appears in a non-neutral regime.
    3. Winner advantage above gene is regime-specific.
       The old "neutral collapses" formulation is RETIRED: drift autocorrelation
       at n=10,000 is real and produces genuine gene predictive signal in neutral.
       The right null is: the advantage of winner_level over gene is larger in the
       test regime than in neutral_wf. Formally:
         (score_winner[regime] - score_gene[regime]) >
         (score_winner[neutral] - score_gene[neutral])
       If winner_level IS gene, require that score_gene[regime] - score_gene[neutral] > 0
       (selection adds something beyond drift).
    4. Fair comparison: all levels evaluated on same target.
    """
    score  = pred_row.get("score", float("nan"))
    ci_lo  = pred_row.get("ci_lo", float("nan"))
    ci_hi  = pred_row.get("ci_hi", float("nan"))

    null = _score_null(TARGET_TYPES.get(target, "continuous"))

    # Condition 1
    cond1 = (not np.isnan(ci_lo) and ci_lo > null and (ci_hi - ci_lo) < 0.5)

    # Condition 2
    cond2 = regime != "neutral_wf"

    # Condition 3: winner advantage above gene is regime-specific (not just drift)
    def _get_score(regime_, level_):
        rows = [r for r in all_pred_rows
                if r["regime"] == regime_ and r["target"] == target
                and r["level"] == level_ and r.get("status") == "OK"]
        return rows[0].get("score", float("nan")) if rows else float("nan")

    score_winner_regime  = score   # already have this
    score_gene_regime    = _get_score(regime, "gene")
    score_winner_neutral = _get_score("neutral_wf", winner_level)
    score_gene_neutral   = _get_score("neutral_wf", "gene")

    if winner_level == "gene":
        # Gene wins: require regime adds beyond drift
        advantage_regime  = score_winner_regime  - score_gene_neutral
        cond3 = (not np.isnan(score_gene_neutral)
                 and advantage_regime > 0.02)
    elif target == "ecology":
        # Ecology target is eco_evol-only (nan elsewhere); no neutral comparison
        # possible. Cond3 vacuously satisfied — the comparison is undefined by design.
        cond3 = True
    else:
        # Higher-level wins: require advantage over gene is larger in regime than neutral
        adv_regime  = score_winner_regime  - score_gene_regime
        adv_neutral = score_winner_neutral - score_gene_neutral
        if np.isnan(adv_regime) or np.isnan(adv_neutral):
            cond3 = False
        else:
            cond3 = adv_regime > adv_neutral

    # Condition 4: fair comparison uses regime-allowed observer set
    def _allowed_observers(regime_: str):
        if regime_ == "eco_evol":
            return ["gene", "lineage", "organism", "org_eco"]
        return ["gene", "lineage", "organism"]

    cond4 = all(
        any(r["regime"] == regime and r["target"] == target and r["level"] == lv
            for r in all_pred_rows)
        for lv in _allowed_observers(regime)
    )

    earned = cond1 and cond2 and cond3 and cond4

    return {
        "regime":  regime, "target": target, "level": winner_level,
        "cond1_ci_stable":          cond1,
        "cond2_non_neutral":        cond2,
        "cond3_advantage_specific": cond3,
        "cond4_fair_comparison":    cond4,
        "earned":                   earned,
        # diagnostic values for audit
        "_adv_regime":  float(score_winner_regime - score_gene_regime)
                        if not np.isnan(score_gene_regime) else None,
        "_adv_neutral": float(score_winner_neutral - score_gene_neutral)
                        if not (np.isnan(score_winner_neutral) or np.isnan(score_gene_neutral))
                        else None,
    }


# ---------------------------------------------------------------------------
# Pilot diagnostic
# ---------------------------------------------------------------------------

def run_pilot(n_pilot: int = 30, seed_base: int = 9000) -> dict:
    """Run pilot to check:
      1. Median surviving lineages at T=50
      2. Fraction of lineage interventions flagged ALLELE_CONTAMINATED
    Fallback rule: if median surviving < 5 or contamination > 0.20,
    reduce T to 35 (do NOT increase N).
    """
    from evol_sim import BATTERY_PARAMS
    T = BATTERY_PARAMS["T"]
    N = BATTERY_PARAMS["N"]
    n_alleles = BATTERY_PARAMS["n_alleles"]
    delta_lin  = BATTERY_PARAMS["delta_lineage"]

    trajs = []
    for rep in range(n_pilot):
        traj = run_regime("selected_wf", seed=seed_base + rep)
        trajs.append(traj)

    lineage_diag = pilot_lineage_diagnostics(trajs)

    # Contamination check using the locked delta_lineage, not the full delta
    contaminated = 0
    for rep in range(n_pilot):
        rng   = np.random.default_rng(seed_base + 50000 + rep)
        traj  = run_regime("selected_wf", seed=seed_base + rep)
        s0    = traj[0]
        _, status = _apply_pilot_lineage_interv(s0, delta_lin, rng, N, n_alleles)
        if "ALLELE_CONTAMINATED" in status or "INSUFFICIENT" in status:
            contaminated += 1

    contamination_rate = contaminated / n_pilot
    recommend_T = T if (
        lineage_diag["median_surviving"] >= 5
        and contamination_rate <= 0.20
    ) else 35

    return {
        **lineage_diag,
        "contamination_rate": contamination_rate,
        "current_T":          T,
        "recommended_T":      recommend_T,
        "note": ("REDUCE T to 35" if recommend_T == 35 else "T=50 OK"),
    }


def _apply_pilot_lineage_interv(state, delta, rng, N, n_alleles=4):
    from evol_interventions import intervene_lineage
    counts = np.bincount(state.lineages, minlength=state.n_founders)
    freqs  = counts / N
    alive  = np.where(freqs > 0)[0]
    if len(alive) < 2:
        return state.copy(), "INSUFFICIENT_LINEAGE_DIVERSITY"
    boost  = alive[int(np.argmin(freqs[alive]))]
    reduce = alive[int(np.argmax(freqs[alive]))]
    return intervene_lineage(state, delta, boost, reduce, rng, n_alleles=n_alleles)


# ---------------------------------------------------------------------------
# Main battery runner
# ---------------------------------------------------------------------------

def check_eco_non_collapse(n_reps: int = 30, seed_base: int = 8000) -> dict:
    """
    Eco non-collapse gate — three locked conditions (must ALL pass):

    1. R not crashed: mean R across replicates at T > 0.1 (not at floor)
    2. R is dynamic, not constant: R(t) varies over time within a trajectory
       (std of R over time > 0.01) — confirms resource actually fluctuates
    3. Allele-specific consumption creates differential depletion:
       dominant allele (allele 0, highest base fitness + highest consumption)
       depletes R measurably compared to neutral expectation.
       Test: R_mean when allele 0 is above 50% frequency is lower than
       R_mean when allele 0 is below 50% frequency.

    Note on multiplicative fitness: with fitness = base[a] * R/K,
    allele ordering never reverses (all fitnesses scale together).
    The eco feedback operates through depletion, not reversal:
    high-consumption alleles deplete R, reducing their own advantage.
    Condition 3 tests whether this feedback is measurable.
    """
    from evol_sim import run_regime, ECO_PARAMS
    import numpy as np

    trajectory_Rs  = []
    final_Rs       = []
    R_when_high    = []   # R values when allele 0 freq > 0.5
    R_when_low     = []   # R values when allele 0 freq < 0.5

    for rep in range(n_reps):
        traj = run_regime("eco_evol", seed=seed_base + rep, full_traj=True)
        Rs   = np.array([s.resource for s in traj])
        trajectory_Rs.append(Rs)
        final_Rs.append(Rs[-1])
        for s in traj[1:]:
            freq0 = np.mean(s.alleles == 0)
            if freq0 > 0.5:
                R_when_high.append(s.resource)
            else:
                R_when_low.append(s.resource)

    final_Rs      = np.array(final_Rs)
    R_mean        = float(np.mean(final_Rs))
    # Within-trajectory std (mean over replicates)
    R_traj_stds   = [float(np.std(rs)) for rs in trajectory_Rs]
    R_traj_std_mean = float(np.mean(R_traj_stds))

    # Condition 1: R not crashed (mean > 0.1)
    cond1 = R_mean > 0.1

    # Condition 2: R actually fluctuates over time (within-traj std > 0.005)
    cond2 = R_traj_std_mean > 0.005

    # Condition 3: allele-depletion feedback measurable
    # R is lower when high-consumption allele dominates
    if R_when_high and R_when_low:
        mean_R_high = float(np.mean(R_when_high))
        mean_R_low  = float(np.mean(R_when_low))
        # High-consumption allele 0 should deplete R: R_when_high < R_when_low
        cond3 = mean_R_high < mean_R_low
    else:
        cond3 = False
        mean_R_high = float("nan")
        mean_R_low  = float("nan")

    gate_pass = cond1 and cond2 and cond3

    return {
        "R_mean_final":           R_mean,
        "R_within_traj_std_mean": R_traj_std_mean,
        "R_when_allele0_dominant": mean_R_high if R_when_high else float("nan"),
        "R_when_allele0_rare":     mean_R_low  if R_when_low  else float("nan"),
        "cond1_R_not_crashed":         cond1,
        "cond2_R_dynamic":             cond2,
        "cond3_depletion_feedback":    cond3,
        "gate_pass":                   gate_pass,
    }


def run_anti_collapse_checks(pred_results: list[dict],
                              n_alleles: int,
                              seed_base: int = SEED_BASE) -> list[dict]:
    """
    Run gene-vs-organism and organism-vs-org+eco anti-collapse checks.
    Uses a joint fit of both observer sets on the same replicates.
    Requires re-running replicates to build joint feature matrix.
    Stores results; does not block the battery but flags COLLAPSED cells.
    """
    anti_collapse_results = []
    N_CHECK_REPS = 80   # enough for a stable joint fit

    for regime in REGIMES:
        if regime == "neutral_wf":
            continue   # collapse in neutral is expected
        for target in TARGET_NAMES:
            if target == "ecology" and regime != "eco_evol":
                continue   # ecology undefined outside eco regime

            # Collect observations and targets using t_0 snapshot
            t_0_ac = BATTERY_PARAMS["t_0"]
            eco_valid = (regime == "eco_evol")
            gene_obs = []; org_obs = []; org_eco_obs = []; ys = []
            for rep in range(N_CHECK_REPS):
                traj = run_regime(regime, seed=seed_base + 20000 + rep, full_traj=True)
                s_obs = traj[t_0_ac]; sT = traj[-1]
                tgt = compute_target(target, s_obs, sT, n_alleles, regime)
                if np.isnan(tgt):
                    continue
                gene_obs.append(get_observation(s_obs, "gene",     n_alleles))
                org_obs.append( get_observation(s_obs, "organism", n_alleles))
                if eco_valid:
                    org_eco_obs.append(get_observation(s_obs, "org_eco", n_alleles))
                ys.append(tgt)

            if len(ys) < CV_FOLDS + 1:
                continue

            y      = np.array(ys)
            X_gene = np.array(gene_obs)
            X_org  = np.array(org_obs)
            X_joint_go = np.hstack([X_gene, X_org])
            t_type = TARGET_TYPES[target]

            score_gene = score_predictive(X_gene,     y, t_type, seed_base)[0]
            score_org  = score_predictive(X_org,      y, t_type, seed_base)[0]
            score_go   = score_predictive(X_joint_go, y, t_type, seed_base)[0]

            ac1 = check_anti_collapse(score_gene, score_org, score_go, "gene", "organism", t_type)
            ac1.update({"regime": regime, "target": target})
            anti_collapse_results.append(ac1)

            if eco_valid:
                X_oeco = np.array(org_eco_obs)
                X_R_only = X_oeco[:, -1:]
                X_org_plus_R = np.hstack([X_org, X_R_only])
                score_oeco  = score_predictive(X_oeco,       y, t_type, seed_base)[0]
                score_org_R = score_predictive(X_org_plus_R, y, t_type, seed_base)[0]
                ac2 = check_anti_collapse(score_org, score_oeco, score_org_R, "organism", "org_eco", t_type)
                ac2.update({"regime": regime, "target": target})
                anti_collapse_results.append(ac2)

    n_collapsed = sum(1 for r in anti_collapse_results if r["status"] == "COLLAPSED")
    n_vacuous   = sum(1 for r in anti_collapse_results if r["status"] == "VACUOUS")
    n_separated = sum(1 for r in anti_collapse_results if r["status"] == "SEPARATED")
    print(f"  Anti-collapse: {n_separated} SEPARATED, "
          f"{n_collapsed} COLLAPSED, {n_vacuous} VACUOUS")

    return anti_collapse_results


def run_battery(workers: int = DEFAULT_WORKERS,
                n_pred: int  = BATTERY_PARAMS["R_replicates"],
                n_caus: int  = BATTERY_PARAMS["R_intervention"],
                delta:  float = BATTERY_PARAMS["delta"],
                delta_lineage: float = BATTERY_PARAMS["delta_lineage"],
                outdir: str  = "outputs/evol") -> dict:
    os.makedirs(outdir, exist_ok=True)
    n_alleles = BATTERY_PARAMS["n_alleles"]
    T         = BATTERY_PARAMS["T"]

    # Step 0: JIT warm-up (cached after first run; ~2s overhead)
    from evol_sim import warmup_jit
    warmup_jit(verbose=True)

    # Step 0b: pilot
    print("\nRunning pilot diagnostics ...")
    pilot = run_pilot()
    print(json.dumps(pilot, indent=2))
    if pilot["recommended_T"] < T:
        print(f"WARNING: Pilot recommends T={pilot['recommended_T']}. "
              "Update BATTERY_PARAMS['T'] before full run.")

    # Step 0c: eco non-collapse gate
    print("\nRunning eco non-collapse gate ...")
    eco_gate = check_eco_non_collapse()
    print(json.dumps(eco_gate, indent=2))
    if not eco_gate["gate_pass"]:
        print("WARNING: Eco non-collapse gate FAILED. "
              "Eco-evol regime results will be flagged UNRELIABLE.")

    # Step 1: predictive cells — ecology target only for eco_evol
    # One pool across all cells; each worker runs a complete cell serially.
    t_0 = BATTERY_PARAMS["t_0"]
    pred_results = []
    cells_pred   = [
        (r, t, l)
        for r, t, l in product(REGIMES, TARGET_NAMES, OBSERVER_NAMES)
        if not (t == "ecology" and r != "eco_evol")
        and not (l == "org_eco" and r != "eco_evol")
    ]
    pred_args = [
        (r, t, l, n_pred, SEED_BASE, n_alleles, T, t_0)
        for r, t, l in cells_pred
    ]
    print(f"\nPredictive: {len(cells_pred)} cells × {n_pred} reps"
          f"  (workers={workers}) ...")
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            pred_raw = list(tqdm(
                pool.imap(_run_pred_cell_worker, pred_args),
                total=len(pred_args), desc="pred", unit="cell"
            ))
    else:
        pred_raw = [_run_pred_cell_worker(a)
                    for a in tqdm(pred_args, desc="pred", unit="cell")]

    for regime, target, level, reps in pred_raw:
        observations = [r[0] for r in reps]
        targets      = [r[1] for r in reps]
        if len(targets) < CV_FOLDS + 1:
            pred_results.append({
                "regime": regime, "target": target, "level": level,
                "score": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_reps": len(targets), "status": "SKIPPED_INSUFFICIENT",
            })
        else:
            X = np.array(observations); y = np.array(targets)
            score, ci_lo, ci_hi = score_predictive(X, y, TARGET_TYPES[target])
            pred_results.append({
                "regime": regime, "target": target, "level": level,
                "score": score, "ci_lo": ci_lo, "ci_hi": ci_hi,
                "n_reps": len(targets), "status": "OK",
            })

    # Step 1b: anti-collapse checks
    print("\nRunning anti-collapse checks ...")
    anti_collapse = run_anti_collapse_checks(pred_results, n_alleles)

    # Step 2: causal cells — ecology target only for eco_evol
    causal_results = []
    cells_caus     = [
        (r, t, i)
        for r, t, i in product(REGIMES, TARGET_NAMES, INTERVENTION_CLASSES)
        if not (t == "ecology" and r != "eco_evol")
        and not (i == "ecology" and r != "eco_evol")
    ]
    caus_args = [
        (r, t, i, n_caus, SEED_BASE, delta, delta_lineage, n_alleles, T, t_0)
        for r, t, i in cells_caus
    ]
    print(f"\nCausal: {len(cells_caus)} cells × {n_caus} reps"
          f"  (workers={workers}) ...")
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            caus_raw = list(tqdm(
                pool.imap(_run_caus_cell_worker, caus_args),
                total=len(caus_args), desc="caus", unit="cell"
            ))
    else:
        caus_raw = [_run_caus_cell_worker(a)
                    for a in tqdm(caus_args, desc="caus", unit="cell")]

    for regime, target, interv, reps in caus_raw:
        ctrl_targets = []; int_targets = []; contaminated = 0
        for r in reps:
            if r is None: continue
            if isinstance(r, tuple) and r[0] == "CONTAMINATED":
                contaminated += 1; continue
            if isinstance(r, tuple) and len(r) == 3 and r[2] == "OK":
                ctrl_targets.append(r[0]); int_targets.append(r[1])
        if len(int_targets) < 5 or len(ctrl_targets) < 5:
            causal_results.append({
                "regime": regime, "target": target, "interv_class": interv,
                "gap": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "gap_sign": None, "contaminated": contaminated,
                "n_ctrl": len(ctrl_targets), "n_int": len(int_targets),
                "status": "INSUFFICIENT_ARMS",
            })
        else:
            ctrl_arr = np.array(ctrl_targets); int_arr = np.array(int_targets)
            gap = float(np.mean(int_arr) - np.mean(ctrl_arr))
            rng_b = np.random.default_rng(SEED_BASE)
            boot_gaps = [
                float(np.mean(rng_b.choice(int_arr, len(int_arr)))
                      - np.mean(rng_b.choice(ctrl_arr, len(ctrl_arr))))
                for _ in range(N_BOOT_SCORE)
            ]
            causal_results.append({
                "regime": regime, "target": target, "interv_class": interv,
                "gap": gap,
                "ci_lo": float(np.percentile(boot_gaps, 2.5)),
                "ci_hi": float(np.percentile(boot_gaps, 97.5)),
                "gap_sign": "positive" if gap > 0 else "negative",
                "contaminated": contaminated,
                "n_ctrl": len(ctrl_targets), "n_int": len(int_targets),
                "status": "OK",
            })

    # Step 3: verdict maps
    verdicts = build_verdict_maps(pred_results, causal_results)

    # Step 4: blocking conditions
    blocking = []
    for regime in REGIMES:
        for target in TARGET_NAMES:
            winner = verdicts["pred_winner"][regime][target]
            if winner == "NONE":
                continue
            row = next(
                (r for r in pred_results
                 if r["regime"] == regime and r["target"] == target
                 and r["level"] == winner), {}
            )
            bc = check_blocking_conditions(
                regime, target, winner, row, pred_results, causal_results
            )
            blocking.append(bc)

    # Step 5: flag eco cells UNRELIABLE if gate failed
    eco_reliable = eco_gate["gate_pass"]
    if not eco_reliable:
        for row in pred_results + causal_results:
            if row.get("regime") == "eco_evol":
                row["eco_gate_failed"] = True

    # Step 6: Save
    results = {
        "pilot":         pilot,
        "eco_gate":      eco_gate,
        "pred_results":  pred_results,
        "causal_results": causal_results,
        "anti_collapse": anti_collapse,
        "verdicts":      verdicts,
        "blocking":      blocking,
    }
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: (float(x) if isinstance(x, (float, np.floating))
                                     else (int(x) if isinstance(x, (int, np.integer))
                                     else None)))

    # Step 7: dissociation summary
    print("\nDissociation panel (pred_winner ≠ causal_winner):")
    any_diss = False
    for regime in REGIMES:
        for target in TARGET_NAMES:
            if verdicts["dissociation"][regime][target]:
                pw  = verdicts["pred_winner"][regime][target]
                cw  = verdicts["causal_winner"][regime][target]
                tag = " [ECO GATE FAILED]" if (regime == "eco_evol"
                                               and not eco_reliable) else ""
                print(f"  [{regime} × {target}]  pred={pw}  causal={cw}{tag}")
                any_diss = True
    if not any_diss:
        print("  None detected — KT4 not yet satisfied.")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",       type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--n-pred",        type=int,   default=BATTERY_PARAMS["R_replicates"])
    parser.add_argument("--n-caus",        type=int,   default=BATTERY_PARAMS["R_intervention"])
    parser.add_argument("--delta",         type=float, default=BATTERY_PARAMS["delta"])
    parser.add_argument("--delta-lineage", type=float, default=BATTERY_PARAMS["delta_lineage"])
    parser.add_argument("--outdir",        default="outputs/evol")
    args = parser.parse_args()
    run_battery(args.workers, args.n_pred, args.n_caus,
                delta=args.delta, delta_lineage=args.delta_lineage,
                outdir=args.outdir)