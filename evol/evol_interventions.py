"""
evol_interventions.py  —  Intervention operators (locked)
=========================================================
All interventions:
  - Keep N fixed
  - Change composition, not population size
  - Return (new_state, status_string)

Lineage intervention includes allele-contamination check.
Causal winners compared within fixed target only (never across targets).
"""
from __future__ import annotations
import numpy as np
from evol_sim import PopState
from evol_observers import observe_gene

# Allele-contamination tolerance (locked)
# Allele-contamination tolerance for lineage interventions.
# At t_0=20, lineages have drifted to near-monomorphism, so any lineage
# redistribution necessarily moves some allele mass. 0.05 = one sampling
# standard deviation at N=200 (sigma = sqrt(p(1-p)/N) ≈ 0.025 at p=0.25).
# Arms with shift > 0.05 are flagged ALLELE_CONTAMINATED and excluded.
ALLELE_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# Intervention 1: Gene — allele-frequency perturbation
# ---------------------------------------------------------------------------

def intervene_gene(state: PopState, delta: float,
                   target_allele: int, source_allele: int,
                   rng: np.random.Generator) -> tuple[PopState, str]:
    """
    Shift delta mass from source_allele to target_allele.
    N fixed: resample floor(delta*N) individuals that carry source_allele
    and change their allele to target_allele.
    Lineage tags unaffected.
    """
    N = len(state.alleles)
    n_shift = max(1, int(np.floor(delta * N)))

    new_alleles  = state.alleles.copy()
    new_lineages = state.lineages.copy()
    new_fitness  = state.fitness.copy()

    source_idx = np.where(new_alleles == source_allele)[0]
    if len(source_idx) == 0:
        return state.copy(), "NO_SOURCE_ALLELE"

    n_actual = min(n_shift, len(source_idx))
    chosen   = rng.choice(source_idx, size=n_actual, replace=False)
    new_alleles[chosen] = target_allele
    # Fitness recomputed externally by caller after step — not recomputed here
    # to keep intervention operators pure state manipulations.

    new_state = PopState(
        alleles=new_alleles, lineages=new_lineages,
        fitness=new_fitness, resource=state.resource,
        t=state.t, n_founders=state.n_founders,
    )
    return new_state, "OK"


# ---------------------------------------------------------------------------
# Intervention 2: Lineage — redistribution with allele-contamination check
# ---------------------------------------------------------------------------

def intervene_lineage(state: PopState, delta: float,
                      boost_lineage: int, reduce_lineage: int,
                      rng: np.random.Generator,
                      n_alleles: int = 4,
                      allele_tolerance: float = ALLELE_TOLERANCE
                      ) -> tuple[PopState, str]:
    """
    Lineage redistribution: boost boost_lineage by delta, reduce
    reduce_lineage by delta. N fixed.

    n_alleles must be passed explicitly to avoid inferring from max(alleles)+1,
    which can shrink when rare alleles are absent from the current state.

    Allele-contamination check: if max shift on any allele > allele_tolerance,
    flag ALLELE_CONTAMINATED. Caller must exclude those arms from causal scoring.
    """
    N = len(state.alleles)
    n_shift = max(1, int(np.floor(delta * N)))

    p_before = observe_gene(state, n_alleles)

    new_alleles  = state.alleles.copy()
    new_lineages = state.lineages.copy()
    new_fitness  = state.fitness.copy()

    reduce_idx = np.where(new_lineages == reduce_lineage)[0]
    boost_idx  = np.where(new_lineages == boost_lineage)[0]

    if len(reduce_idx) == 0:
        return state.copy(), "NO_REDUCE_LINEAGE"
    if len(boost_idx) == 0:
        return state.copy(), "NO_BOOST_LINEAGE"

    n_actual = min(n_shift, len(reduce_idx))
    remove   = rng.choice(reduce_idx, size=n_actual, replace=False)
    sources  = rng.choice(boost_idx,  size=n_actual, replace=True)

    new_alleles[remove]  = new_alleles[sources]
    new_lineages[remove] = boost_lineage

    new_state = PopState(
        alleles=new_alleles, lineages=new_lineages,
        fitness=new_fitness, resource=state.resource,
        t=state.t, n_founders=state.n_founders,
    )
    p_after   = observe_gene(new_state, n_alleles)
    max_shift = float(np.max(np.abs(p_after - p_before)))

    if max_shift > allele_tolerance:
        return new_state, f"ALLELE_CONTAMINATED(shift={max_shift:.4f})"
    return new_state, "OK"


# ---------------------------------------------------------------------------
# Intervention 3: Organism — fitness shift
# ---------------------------------------------------------------------------

def intervene_organism(state: PopState, delta_f: float,
                        target_fraction: float,
                        rng: np.random.Generator) -> tuple[PopState, str]:
    """
    Shift fitness of the lowest-fitness target_fraction of individuals by delta_f.
    Tie-breaking is randomized via lexsort to avoid label-order artifacts.
    rng is used for tie-breaking only. Clips to non-negative. N fixed.
    """
    N = len(state.alleles)
    n_target = max(1, int(np.floor(target_fraction * N)))
    # lexsort: primary key = fitness, secondary key = random (breaks ties)
    order    = np.lexsort((rng.random(N), state.fitness))
    chosen   = order[:n_target]

    new_fitness = state.fitness.copy()
    new_fitness[chosen] = np.maximum(0.0, new_fitness[chosen] + delta_f)

    new_state = PopState(
        alleles=state.alleles.copy(), lineages=state.lineages.copy(),
        fitness=new_fitness, resource=state.resource,
        t=state.t, n_founders=state.n_founders,
    )
    return new_state, "OK"


# ---------------------------------------------------------------------------
# Intervention 4: Ecology — resource perturbation
# ---------------------------------------------------------------------------

def intervene_ecology(state: PopState, delta_R: float) -> tuple[PopState, str]:
    """
    Perturb resource level: R(t_0) += delta_R.
    phi(t_0) unchanged. Fitness updated on next step by regime.
    Clips R to [0, 2*K] using K from locked ECO_PARAMS.
    """
    from evol_sim import ECO_PARAMS
    K     = ECO_PARAMS["K"]
    new_R = max(0.0, min(state.resource + delta_R, 2.0 * K))
    new_state = PopState(
        alleles=state.alleles.copy(), lineages=state.lineages.copy(),
        fitness=state.fitness.copy(), resource=new_R,
        t=state.t, n_founders=state.n_founders,
    )
    return new_state, "OK"


# ---------------------------------------------------------------------------
# Intervention dispatcher
# ---------------------------------------------------------------------------

INTERVENTION_CLASSES = ["gene", "lineage", "organism", "ecology"]

def apply_intervention(state: PopState, interv_class: str,
                        delta: float, rng: np.random.Generator,
                        n_alleles: int = 4) -> tuple[PopState, str]:
    """
    Apply a level-targeted intervention.
    Uses pre-set intervention targets:
      gene:      shift allele 1 → allele 0 (toward favored)
      lineage:   boost lineage with lowest frequency, reduce lineage with highest
      organism:  increase fitness of bottom 10% by delta
      ecology:   increase resource by delta (positive perturbation)
    """
    if interv_class == "gene":
        counts = np.bincount(state.alleles, minlength=n_alleles)
        source = int(np.argmax(counts[1:]) + 1)   # most common non-favored
        return intervene_gene(state, delta, target_allele=0,
                               source_allele=source, rng=rng)

    elif interv_class == "lineage":
        counts = np.bincount(state.lineages, minlength=state.n_founders)
        freqs  = counts / len(state.lineages)
        alive  = np.where(freqs > 0)[0]
        if len(alive) < 2:
            return state.copy(), "INSUFFICIENT_LINEAGE_DIVERSITY"
        boost  = alive[int(np.argmin(freqs[alive]))]   # rarest
        reduce = alive[int(np.argmax(freqs[alive]))]   # most common
        return intervene_lineage(state, delta, boost, reduce, rng,
                                  n_alleles=n_alleles)

    elif interv_class == "organism":
        return intervene_organism(state, delta_f=delta,
                                   target_fraction=0.10, rng=rng)

    elif interv_class == "ecology":
        return intervene_ecology(state, delta_R=delta)

    else:
        raise ValueError(f"Unknown intervention class: {interv_class}")
