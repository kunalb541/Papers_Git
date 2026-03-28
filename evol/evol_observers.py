"""
evol_observers.py  —  Observer functions (locked definitions)
=============================================================
Locked observer feature vectors:

  Gene          allele frequency vector           shape (k,)   k=n_alleles
  Lineage       top-20 sorted freqs + 3 stats     shape (23,)
  Organism      20 quantiles + 2 moments + top-10 shape (32,)
  Org+Ecology   organism vector + R scalar        shape (33,)

All observers are permutation-invariant (do not depend on individual ordering).
Gene and lineage use separate representations:
  - predictive lineage = sorted frequency spectrum (permutation-invariant)
  - intervention/target lineage = exact founder IDs in state.lineages
"""
from __future__ import annotations
import numpy as np
from evol_sim import PopState

# Anti-collapse thresholds (provisional, locked at preregistration)
MIN_INCREMENTAL_R2 = 0.02

# Lineage top-k for predictive observer
LINEAGE_TOP_K = 20

# Organism observer parameters
N_QUANTILES    = 20
ORGANISM_TOP_K = 10


# ---------------------------------------------------------------------------
# Gene observer
# ---------------------------------------------------------------------------

def observe_gene(state: PopState, n_alleles: int) -> np.ndarray:
    """Allele frequency vector p(t_0). Shape (n_alleles,).
    Gene observer sees genotype frequencies only.
    No one-to-one genotype-phenotype assumption."""
    counts = np.bincount(state.alleles, minlength=n_alleles)
    return counts / len(state.alleles)


# ---------------------------------------------------------------------------
# Lineage observer (predictive — permutation-invariant)
# ---------------------------------------------------------------------------

def observe_lineage(state: PopState,
                    top_k: int = LINEAGE_TOP_K) -> np.ndarray:
    """Permutation-invariant lineage summary for predictive scoring.
    Uses sorted founder-frequency spectrum, NOT exact founder IDs.
    Exact founder IDs remain in state.lineages for interventions/targets.

    Features:
      - top_k sorted lineage frequencies (descending), zero-padded
      - number of surviving lineages (freq > 0)
      - entropy of lineage distribution
      - max lineage share (= sorted_freqs[0])
    Shape: (top_k + 3,) = (23,)
    """
    counts = np.bincount(state.lineages, minlength=state.n_founders)
    freqs  = counts / len(state.lineages)
    sorted_freqs = np.sort(freqs)[::-1]

    # top-k, zero-padded if fewer survive
    top_k_freqs = sorted_freqs[:top_k]
    if len(top_k_freqs) < top_k:
        top_k_freqs = np.pad(top_k_freqs, (0, top_k - len(top_k_freqs)))

    surviving  = float(np.sum(freqs > 0))
    p_nonzero  = freqs[freqs > 0]
    entropy    = float(-np.sum(p_nonzero * np.log(p_nonzero + 1e-12)))
    max_share  = float(sorted_freqs[0]) if len(sorted_freqs) > 0 else 0.0

    return np.concatenate([top_k_freqs, [surviving, entropy, max_share]])


# ---------------------------------------------------------------------------
# Organism observer (permutation-invariant)
# ---------------------------------------------------------------------------

def observe_organism(state: PopState,
                     n_quantiles: int = N_QUANTILES,
                     top_k: int = ORGANISM_TOP_K) -> np.ndarray:
    """Permutation-invariant realized fitness summary.
    Observer sees realized fitness values only.
    No genotype-phenotype assumption: two individuals with the same allele
    can have different fitness (e.g. in eco regime).

    Features:
      - n_quantiles evenly spaced quantiles [0%..100%]
      - mean, variance
      - top_k sorted fitness values (descending)
    Shape: (n_quantiles + 2 + top_k,) = (32,)
    """
    f = state.fitness
    quantiles = np.percentile(f, np.linspace(0, 100, n_quantiles))  # (20,)
    moments   = np.array([np.mean(f), np.var(f)])                    # (2,)
    top_vals  = np.sort(f)[::-1][:top_k]                            # (10,)
    return np.concatenate([quantiles, moments, top_vals])


# ---------------------------------------------------------------------------
# Organism + Ecology observer
# ---------------------------------------------------------------------------

def observe_org_eco(state: PopState,
                    n_quantiles: int = N_QUANTILES,
                    top_k: int = ORGANISM_TOP_K) -> np.ndarray:
    """Organism summary augmented with resource scalar R(t_0).
    Collapses to observe_organism when R is constant across replicates.

    Shape: (32 + 1,) = (33,)
    """
    org = observe_organism(state, n_quantiles, top_k)
    return np.append(org, state.resource)


# ---------------------------------------------------------------------------
# Observer dispatcher
# ---------------------------------------------------------------------------

OBSERVER_NAMES = ["gene", "lineage", "organism", "org_eco"]

def get_observation(state: PopState, level: str,
                    n_alleles: int = 4) -> np.ndarray:
    if level == "gene":
        return observe_gene(state, n_alleles)
    elif level == "lineage":
        return observe_lineage(state)
    elif level == "organism":
        return observe_organism(state)
    elif level == "org_eco":
        return observe_org_eco(state)
    else:
        raise ValueError(f"Unknown observer level: {level}")


# ---------------------------------------------------------------------------
# Anti-collapse diagnostic
# ---------------------------------------------------------------------------

def check_anti_collapse(score_A: float, score_B: float, score_AB: float,
                         label_A: str, label_B: str,
                         target_type: str = "continuous") -> dict:
    """
    Anti-collapse check between two observers A and B.

    target_type controls the null baseline:
      continuous (R²): null = 0.  Vacuous if max(score) < 0.05.
      binary (AUC):    null = 0.5. Vacuous if max(score - 0.5) < 0.05,
                       i.e. max(score) < 0.55.

    MIN_INCREMENTAL_R2 is a practical threshold on the score scale relative
    to null — not a theoretically scale-invariant bound.
    """
    null = 0.5 if target_type == "binary" else 0.0
    # Shift scores to be relative to null for fair comparison
    sA_shifted  = score_A  - null
    sB_shifted  = score_B  - null
    sAB_shifted = score_AB - null

    increment = sAB_shifted - max(sA_shifted, sB_shifted)

    if max(sA_shifted, sB_shifted) < 0.05:
        status = "VACUOUS"
    elif increment >= MIN_INCREMENTAL_R2:
        status = "SEPARATED"
    else:
        status = "COLLAPSED"
    return {
        "label_A":        label_A,
        "label_B":        label_B,
        "score_A":        score_A,
        "score_B":        score_B,
        "score_combined": score_AB,
        "null_baseline":  null,
        "increment":      increment,
        "status":         status,
        "threshold":      MIN_INCREMENTAL_R2,
        "target_type":    target_type,
    }


# ---------------------------------------------------------------------------
# Pilot diagnostics (run before full battery)
# ---------------------------------------------------------------------------

def pilot_lineage_diagnostics(trajectories: list) -> dict:
    """
    Run on a set of trajectories (each a list of PopState).
    Reports median surviving lineages at T and fraction extinct.
    Input: list of trajectories, each trajectory = list of PopState (t=0..T)
    """
    final_states = [traj[-1] for traj in trajectories]
    surviving_counts = []
    for s in final_states:
        counts = np.bincount(s.lineages, minlength=s.n_founders)
        surviving_counts.append(np.sum(counts > 0))
    arr = np.array(surviving_counts)
    return {
        "median_surviving":   float(np.median(arr)),
        "mean_surviving":     float(np.mean(arr)),
        "p10_surviving":      float(np.percentile(arr, 10)),
        "p90_surviving":      float(np.percentile(arr, 90)),
        "frac_one_lineage":   float(np.mean(arr <= 1)),
    }
