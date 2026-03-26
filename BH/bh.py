"""
bh.py
=====
One-file paper package for the Bose-Hubbard causal-handle paper (B1).

Generates all data, figures, and LaTeX-ready tables.
Exact Lindblad evolution in the fixed-particle-number sector.

System:
  1D Bose-Hubbard chain, open BC, half-filling, nmax=3.
  Lindblad dephasing with L_i = n_i.

Experiment:
  Targeted (high-Fi sites) vs random dephasing intervention.
  Target: local future occupation loss at intervention sites.

Outputs:
  outputs/data/       -- raw CSV data
  outputs/figures/    -- fig1_main_heatmap.pdf, fig2_robustness.pdf, fig3_mechanism.pdf
  outputs/tables/     -- table_main.tex, table_robust.tex (LaTeX-ready)
  outputs/logs/       -- adjudication note

Dependencies:
  numpy, scipy, pandas, matplotlib, seaborn, tqdm

Run:
  python bh.py
"""

from __future__ import annotations

import itertools
import json
import os
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm

# =============================================================================
# PATHS / STYLE
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


def savefig(fig, stem):
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(FIG_DIR, stem + ext), dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


# =============================================================================
# FOCK SPACE BASIS
# =============================================================================

def build_basis(L, N, nmax):
    """All Fock states |n1,...,nL> with sum=N, 0<=ni<=nmax."""
    states = []
    for combo in itertools.product(range(nmax + 1), repeat=L):
        if sum(combo) == N:
            states.append(combo)
    return sorted(states)


def basis_index(basis):
    return {state: idx for idx, state in enumerate(basis)}


# =============================================================================
# OPERATORS
# =============================================================================

def number_op(site, D, basis):
    op = np.zeros((D, D), dtype=np.float64)
    for idx, state in enumerate(basis):
        op[idx, idx] = state[site]
    return op


def build_hamiltonian(L, J, U, basis, idx_map):
    D = len(basis)
    H = np.zeros((D, D), dtype=np.float64)

    for state_idx, state in enumerate(basis):
        for site in range(L):
            ni = state[site]
            H[state_idx, state_idx] += 0.5 * U * ni * (ni - 1)

    for site in range(L - 1):
        for state_idx, state in enumerate(basis):
            ni, nj = state[site], state[site + 1]

            if nj > 0 and ni < 3:
                new = list(state)
                new[site] = ni + 1
                new[site + 1] = nj - 1
                t = tuple(new)
                if t in idx_map:
                    H[idx_map[t], state_idx] += -J * np.sqrt((ni + 1) * nj)

            if ni > 0 and nj < 3:
                new = list(state)
                new[site] = ni - 1
                new[site + 1] = nj + 1
                t = tuple(new)
                if t in idx_map:
                    H[idx_map[t], state_idx] += -J * np.sqrt(ni * (nj + 1))

    return H


# =============================================================================
# LINDBLAD SUPEROPERATOR
# =============================================================================

def build_liouvillian(H, L_ops, gammas):
    D = H.shape[0]
    I = np.eye(D)
    sup = -1j * (np.kron(H, I) - np.kron(I, H.T))

    for Lk, gk in zip(L_ops, gammas):
        LdL = Lk.conj().T @ Lk
        sup += gk * (
            np.kron(Lk, Lk.conj())
            - 0.5 * np.kron(LdL, I)
            - 0.5 * np.kron(I, LdL.T)
        )
    return sup


def evolve_rho(rho, liouvillian, tau):
    D = rho.shape[0]
    vec = rho.flatten()
    vec_t = expm_multiply(liouvillian * tau, vec)
    return vec_t.reshape(D, D)


def site_expectations(rho, n_ops):
    return np.array([np.real(np.trace(nop @ rho)) for nop in n_ops])


def site_variances(rho, n_ops, n2_ops):
    means = site_expectations(rho, n_ops)
    sq = np.array([np.real(np.trace(n2op @ rho)) for n2op in n2_ops])
    return sq - means**2


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_single_condition(L, N, nmax, J_over_U, gamma_base, gamma_extra,
                         tau_list, n_trials, burn_in_time, seed):

    U = 1.0
    J = J_over_U * U

    basis = build_basis(L, N, nmax)
    idx_map = basis_index(basis)
    D = len(basis)

    H = build_hamiltonian(L, J, U, basis, idx_map)
    n_ops = [number_op(i, D, basis) for i in range(L)]
    n2_ops = [nop @ nop for nop in n_ops]

    L_ops = [n_ops[i] for i in range(L)]
    gammas_base = [gamma_base] * L

    liouv_base = build_liouvillian(H, L_ops, gammas_base)

    # Precompute per-site dephasing super-ops
    I_D = np.eye(D)
    site_diss = []
    for i in range(L):
        Lk = n_ops[i]
        LdL = Lk.conj().T @ Lk
        Di = (np.kron(Lk, Lk.conj())
              - 0.5 * np.kron(LdL, I_D)
              - 0.5 * np.kron(I_D, LdL.T))
        site_diss.append(Di)

    # Ground state -> burn-in
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0 = eigvecs[:, 0]
    rho0 = np.outer(psi0, psi0.conj())
    rho_burn = evolve_rho(rho0, liouv_base, burn_in_time)

    Fi = site_variances(rho_burn, n_ops, n2_ops)
    k = max(1, int(np.ceil(L / 3)))
    selected = np.argsort(Fi)[-k:][::-1].tolist()

    # Targeted Liouvillian (fixed)
    liouv_tgt = liouv_base.copy()
    for s in selected:
        liouv_tgt += gamma_extra * site_diss[s]

    rng = np.random.default_rng(seed)
    occ_before = site_expectations(rho_burn, n_ops)

    results = []

    for tau in tau_list:
        rho_tgt = evolve_rho(rho_burn, liouv_tgt, tau)
        occ_tgt = site_expectations(rho_tgt, n_ops)
        loss_tgt = sum(max(0.0, occ_before[s] - occ_tgt[s]) for s in selected)
        delta_tgt = occ_tgt - occ_before

        diffs = []
        mech = []

        for _ in tqdm(range(n_trials),
                      desc=f"  L={L} J/U={J_over_U:.2f} tau={tau:.0f}",
                      leave=False):
            rsites = rng.choice(L, size=k, replace=False).tolist()

            liouv_rnd = liouv_base.copy()
            for s in rsites:
                liouv_rnd += gamma_extra * site_diss[s]

            rho_rnd = evolve_rho(rho_burn, liouv_rnd, tau)
            occ_rnd = site_expectations(rho_rnd, n_ops)
            loss_rnd = sum(max(0.0, occ_before[s] - occ_rnd[s]) for s in rsites)

            diffs.append(loss_tgt - loss_rnd)
            mech.append({
                "delta_tgt": delta_tgt.tolist(),
                "delta_rnd": (occ_rnd - occ_before).tolist(),
                "selected": selected,
                "random": rsites,
            })

        diffs = np.array(diffs)
        mean_diff = float(np.mean(diffs))

        boot = np.empty(1000)
        for b in range(1000):
            idx = rng.integers(0, len(diffs), size=len(diffs))
            boot[b] = np.mean(diffs[idx])
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))

        results.append({
            "L": L, "J_over_U": J_over_U, "tau": tau,
            "mean_diff": mean_diff, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_trials": n_trials, "selected": selected, "selected_k": k,
            "Fi": Fi.tolist(), "mechanism": mech,
            "trial_diffs": diffs.tolist(),
        })

    return {"L": L, "N": N, "J_over_U": J_over_U, "D": D, "k": k,
            "selected": selected, "Fi": Fi.tolist(), "results": results}


# =============================================================================
# MAIN RUN
# =============================================================================

def run_all():
    cfg = {
        "SEED": 20260325,
        "NMAX": 3,
        "GAMMA_BASE": 0.1,
        "GAMMA_EXTRA": 0.5,
        "BURN_IN_TIME": 5.0,
        "N_TRIALS": 100,
        "TAU_LIST": [1.0, 2.0, 3.0,],
        "J_OVER_U_LIST": [0.12, 0.20, 0.30, 0.40],
        "L_LIST": [6, 7],
    }
    save_json(cfg, os.path.join(DATA_DIR, "config.json"))

    all_res = []
    sc = cfg["SEED"]

    for L in cfg["L_LIST"]:
        N = L // 2
        for ju in cfg["J_OVER_U_LIST"]:
            print(f"\n--- L={L}, J/U={ju}, N={N}, D=? ---")
            sc += 1
            res = run_single_condition(
                L=L, N=N, nmax=cfg["NMAX"], J_over_U=ju,
                gamma_base=cfg["GAMMA_BASE"], gamma_extra=cfg["GAMMA_EXTRA"],
                tau_list=cfg["TAU_LIST"], n_trials=cfg["N_TRIALS"],
                burn_in_time=cfg["BURN_IN_TIME"], seed=sc)
            all_res.append(res)
            for r in res["results"]:
                tag = "PASS" if r["ci_lo"] > 0 else "FAIL"
                print(f"  tau={r['tau']:.0f}: diff={r['mean_diff']:.4f} "
                      f"CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] {tag}")

    return all_res, cfg


# =============================================================================
# TABLES (LaTeX-ready)
# =============================================================================

def make_tables(all_res, cfg):
    # --- Table 1: main results at L=6 ---
    rows6 = []
    for res in all_res:
        if res["L"] != 6:
            continue
        for r in res["results"]:
            rows6.append(r)

    df6 = pd.DataFrame(rows6)
    df6.to_csv(os.path.join(DATA_DIR, "results_L6.csv"), index=False)

    # Pivot for LaTeX
    lines = []
    lines.append(r"\begin{tabular}{l ccc}")
    lines.append(r"\toprule")
    lines.append(r"$J/U$ & $\tau=1$ & $\tau=2$ & $\tau=3$ \\")
    lines.append(r"\midrule")
    for ju in cfg["J_OVER_U_LIST"]:
        cells = []
        for tau in cfg["TAU_LIST"]:
            row = df6[(abs(df6["J_over_U"] - ju) < 0.001) &
                      (abs(df6["tau"] - tau) < 0.01)]
            if len(row) == 1:
                r = row.iloc[0]
                cells.append(f"${r['mean_diff']:.3f}$ $[{r['ci_lo']:.3f},\\,{r['ci_hi']:.3f}]$")
            else:
                cells.append("---")
        lines.append(f"{ju:.2f} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(os.path.join(TAB_DIR, "table_main.tex"), "w") as f:
        f.write("\n".join(lines))

    # --- Table 2: robustness across L at J/U=0.30 and 0.40 (with CI) ---
    rng_boot = np.random.default_rng(99)
    rob_lines = []
    rob_lines.append(r"\begin{tabular}{l cc}")
    rob_lines.append(r"\toprule")
    rob_lines.append(r"Condition & $L=6$ & $L=7$ \\")
    rob_lines.append(r"\midrule")

    for ju in [0.30, 0.40]:
        cells = []
        for Lval in [6, 7]:
            all_diffs = []
            for res in all_res:
                if res["L"] != Lval:
                    continue
                for r in res["results"]:
                    if abs(r["J_over_U"] - ju) < 0.001:
                        all_diffs.extend(r.get("trial_diffs", []))
            if all_diffs:
                arr = np.array(all_diffs)
                pooled = float(np.mean(arr))
                boot = np.empty(1000)
                for b in range(1000):
                    bi = rng_boot.integers(0, len(arr), size=len(arr))
                    boot[b] = np.mean(arr[bi])
                lo = float(np.percentile(boot, 2.5))
                hi = float(np.percentile(boot, 97.5))
                cells.append(f"${pooled:.3f}$ $[{lo:.3f},\\,{hi:.3f}]$")
            else:
                cells.append("---")
        rob_lines.append(f"$J/U={ju:.2f}$ & " + " & ".join(cells) + r" \\")

    rob_lines.append(r"\bottomrule")
    rob_lines.append(r"\end{tabular}")

    with open(os.path.join(TAB_DIR, "table_robust.tex"), "w") as f:
        f.write("\n".join(rob_lines))

    # CSV summary
    summary = []
    for res in all_res:
        for r in res["results"]:
            summary.append({
                "L": res["L"], "J_over_U": r["J_over_U"], "tau": r["tau"],
                "mean_diff": r["mean_diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
            })
    pd.DataFrame(summary).to_csv(os.path.join(TAB_DIR, "all_results.csv"), index=False)


# =============================================================================
# FIGURES
# =============================================================================

def make_figures(all_res, cfg):
    # --- Fig 1: heatmap ---
    L6 = [r for res in all_res if res["L"] == 6 for r in res["results"]]
    df = pd.DataFrame(L6)[["J_over_U", "tau", "mean_diff"]]
    pivot = df.pivot(index="J_over_U", columns="tau", values="mean_diff")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Local-loss diff (targeted − random)"})
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("$J/U$")
    ax.set_title("Local-loss difference across regimes ($L=6$)")
    savefig(fig, "fig1_main_heatmap")

    # --- Fig 2: robustness ---
    rob = []
    for res in all_res:
        for r in res["results"]:
            if r["J_over_U"] in [0.30, 0.40]:
                rob.append({"L": res["L"], "J_over_U": r["J_over_U"],
                            "tau": r["tau"], "mean_diff": r["mean_diff"],
                            "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    df_rob = pd.DataFrame(rob)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, ju in zip(axes, [0.30, 0.40]):
        sub = df_rob[df_rob["J_over_U"] == ju]
        for Lv, mk, cl in [(6, "o", "C0"), (7, "^", "C1")]:
            ss = sub[sub["L"] == Lv]
            ax.errorbar(ss["tau"], ss["mean_diff"],
                        yerr=[ss["mean_diff"] - ss["ci_lo"],
                              ss["ci_hi"] - ss["mean_diff"]],
                        fmt=mk, capsize=4, label=f"$L={Lv}$", color=cl,
                        markersize=7, linewidth=1.5)
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_xlabel(r"$\tau$")
        ax.set_title(f"$J/U = {ju}$")
        ax.legend()
    axes[0].set_ylabel("Local-loss diff")
    fig.tight_layout()
    savefig(fig, "fig2_robustness")

    # --- Fig 3: mechanism ---
    mech_res = None
    for res in all_res:
        if res["L"] == 6:
            for r in res["results"]:
                if abs(r["J_over_U"] - 0.30) < 0.01 and abs(r["tau"] - 2.0) < 0.1:
                    mech_res = r
                    break

    if mech_res is not None:
        L = 6
        sel = mech_res["selected"]
        nt = len(mech_res["mechanism"])

        d_tgt = np.zeros(L)
        d_rnd = np.zeros(L)
        for md in mech_res["mechanism"]:
            d_tgt += np.array(md["delta_tgt"])
            d_rnd += np.array(md["delta_rnd"])
        d_tgt /= nt
        d_rnd /= nt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(L)
        w = 0.35
        ax.bar(x - w/2, d_tgt, w, label="Targeted", color="C0", alpha=0.8)
        ax.bar(x + w/2, d_rnd, w, label="Random", color="C1", alpha=0.8)
        for s in sel:
            ax.axvspan(s - 0.5, s + 0.5, alpha=0.12, color="blue")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_xlabel("Site index")
        ax.set_ylabel(r"$\Delta\langle n_i \rangle$")
        ax.set_title(r"Site-resolved occupation change ($J/U=0.30$, $L=6$, $\tau=2$)")
        ax.set_xticks(x)
        ax.legend()
        savefig(fig, "fig3_mechanism")


# =============================================================================
# ADJUDICATION
# =============================================================================

def write_adjudication(all_res):
    lines = ["# B1 Adjudication Note\n\n"]
    all_pass = True
    for res in all_res:
        for r in res["results"]:
            ok = r["ci_lo"] > 0
            if not ok:
                all_pass = False
            lines.append(
                f"- L={res['L']}, J/U={r['J_over_U']:.2f}, tau={r['tau']:.0f}: "
                f"diff={r['mean_diff']:.4f} CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] "
                f"{'PASS' if ok else 'FAIL'}\n")
    lines.append(f"\nVerdict: {'ALL CIs EXCLUDE ZERO' if all_pass else 'SOME FAIL'}\n")
    with open(os.path.join(LOG_DIR, "B1_adjudication.md"), "w") as f:
        f.writelines(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("Running BH causal-handle paper package...\n")

    all_res, cfg = run_all()
    make_tables(all_res, cfg)
    make_figures(all_res, cfg)
    write_adjudication(all_res)

    print(f"\nDone. {time.time() - t0:.0f}s")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
