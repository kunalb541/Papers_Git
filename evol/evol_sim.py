"""
evol_sim.py  —  Population state and regime simulators (Numba-optimised)
=========================================================================
Architecture:
  - Inner step kernels: @njit(cache=True) — no Python overhead in hot loops
  - Preallocated work buffers: a0/a1, l0/l1, f0/f1 swapped, never reallocated
  - No fastmath: float64 IEEE behaviour preserved throughout
  - Outer parallelism: multiprocessing across replicate batches (evol_battery)
  - RNG: NumPy Generator at Python level; random arrays passed into njit kernels

Locked state representation (unchanged):
  alleles  : (N,) int64  — allele ID, changed by mutation
  lineages : (N,) int64  — founder tag from t=0, NEVER changed by mutation
  fitness  : (N,) float64 — realized fitness per individual
  resource : float64
  t        : int
  n_founders: int
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Locked parameters
# ---------------------------------------------------------------------------
BATTERY_PARAMS = {
    "N": 200,
    "n_alleles": 4,
    "mu": 0.005,
    "s": 0.05,
    "T": 50,
    "t_0": 20,              # observation horizon: run to t_0 before taking snapshot
    "R_replicates": 100,
    "R_intervention": 100,
    "delta": 0.10,
    # Lineage intervention uses a smaller delta: at t_0=20 lineages are near-
    # monomorphic, so floor(delta*N) individuals moved. With delta=0.02 and
    # N=200 that moves 4 individuals — the max allele shift is 4/200 = 0.02,
    # which stays within ALLELE_TOLERANCE=0.05. delta=0.10 causes allele shifts
    # equal to delta itself (all 20 movers carry the same allele), saturating
    # the contamination check. Locking delta_lineage=0.02 is the right scale.
    "delta_lineage": 0.02,
    "n_boot": 1000,
    "threshold_verdict": 0.05,
}

GROUP_PARAMS = {
    "n_groups": 10,
    "group_size": 20,
    "s_within": 0.05,
    "s_between": 0.08,
}

ECO_PARAMS = {
    "K": 1.0,
    "r_R": 0.3,
    "R0": 0.8,
    "allele_base_fitness": np.array([1.16, 1.10, 1.04, 0.98]),
    "allele_consumption":  np.array([0.20, 0.15, 0.10, 0.05]),
}

FAVORED_ALLELE = 0
FREQ_DEP_ALPHA = 0.5

REGIMES = [
    "neutral_wf",
    "selected_wf",
    "moran",
    "freq_dep",
    "eco_evol",
    "group_structured",
]


# ---------------------------------------------------------------------------
# PopState
# ---------------------------------------------------------------------------
@dataclass
class PopState:
    alleles:    np.ndarray
    lineages:   np.ndarray
    fitness:    np.ndarray
    resource:   float
    t:          int
    n_founders: int

    def copy(self) -> "PopState":
        return PopState(
            alleles   = self.alleles.copy(),
            lineages  = self.lineages.copy(),
            fitness   = self.fitness.copy(),
            resource  = self.resource,
            t         = self.t,
            n_founders= self.n_founders,
        )


# ---------------------------------------------------------------------------
# njit fitness kernels  (operate on raw arrays)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fitness_neutral_k(N: int, out: np.ndarray) -> None:
    for i in range(N):
        out[i] = 1.0


@njit(cache=True)
def _fitness_selected_k(alleles: np.ndarray, N: int,
                         s: float, favored: int,
                         out: np.ndarray) -> None:
    for i in range(N):
        out[i] = 1.0 + s if alleles[i] == favored else 1.0


@njit(cache=True)
def _fitness_freq_dep_k(alleles: np.ndarray, N: int, n_alleles: int,
                         alpha: float, out: np.ndarray) -> None:
    counts = np.zeros(n_alleles, dtype=np.int64)
    for i in range(N):
        counts[alleles[i]] += 1
    for i in range(N):
        f = 1.0 - alpha * counts[alleles[i]] / N
        out[i] = f if f > 1e-6 else 1e-6


@njit(cache=True)
def _fitness_eco_k(alleles: np.ndarray, N: int,
                   base_fitness: np.ndarray, R: float, K: float,
                   out: np.ndarray) -> None:
    ratio = (R / K) if R > 1e-4 else (1e-4 / K)
    for i in range(N):
        f = base_fitness[alleles[i]] * ratio
        out[i] = f if f > 1e-6 else 1e-6


# ---------------------------------------------------------------------------
# njit WF reproduction kernel (preallocated in/out arrays)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _wf_step_k(alleles: np.ndarray, lineages: np.ndarray,
               fitness: np.ndarray, N: int, mu: float, n_alleles: int,
               rng_sel:  np.ndarray,   # (N,) float64 uniform for parent selection
               rng_mut:  np.ndarray,   # (N,) float64 for mutation trigger
               rng_na:   np.ndarray,   # (N,) int64 for new allele on mutation
               cdf:      np.ndarray,   # (N,) float64 preallocated CDF buffer
               out_a:    np.ndarray,   # (N,) int64 output alleles
               out_l:    np.ndarray,   # (N,) int64 output lineages
               ) -> None:
    """One WF generation. No internal allocation — CDF written into preallocated buf."""
    total = 0.0
    for i in range(N):
        total += fitness[i]

    acc = 0.0
    for i in range(N):
        acc += fitness[i] / total
        cdf[i] = acc

    for i in range(N):
        u = rng_sel[i]
        lo, hi = 0, N - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if cdf[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        out_a[i] = alleles[lo]
        out_l[i] = lineages[lo]

    for i in range(N):
        if rng_mut[i] < mu:
            out_a[i] = rng_na[i] % n_alleles


# ---------------------------------------------------------------------------
# njit Moran kernel (N steps in-place per generation)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _moran_step_k(alleles: np.ndarray, lineages: np.ndarray,
                  fitness: np.ndarray, N: int, mu: float, n_alleles: int,
                  rng_parent: np.ndarray,  # (N,) float64
                  rng_die:    np.ndarray,  # (N,) int64
                  rng_mut:    np.ndarray,  # (N,) float64
                  rng_na:     np.ndarray,  # (N,) int64
                  cdf:        np.ndarray,  # (N,) float64 preallocated
                  ) -> None:
    """N Moran birth-death steps in-place. No internal allocation."""
    total = 0.0
    for i in range(N):
        total += fitness[i]
    acc = 0.0
    for i in range(N):
        acc += fitness[i] / total
        cdf[i] = acc

    for step in range(N):
        u = rng_parent[step]
        lo, hi = 0, N - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if cdf[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        parent = lo
        die    = rng_die[step] % N
        alleles[die]  = alleles[parent]
        lineages[die] = lineages[parent]
        if rng_mut[step] < mu:
            alleles[die] = rng_na[step] % n_alleles


@njit(cache=True)
def _mean_consumption_k(alleles: np.ndarray, ac: np.ndarray) -> float:
    """Mean resource consumption for current allele composition. njit-safe."""
    s = 0.0
    N = len(alleles)
    for i in range(N):
        s += ac[alleles[i]]
    return s / N


# ---------------------------------------------------------------------------
# Preallocated buffer class (one instance per trajectory run, reused)
# ---------------------------------------------------------------------------

class _Buf:
    __slots__ = ("a0","a1","l0","l1","f0","f1",
                 "rsel","rmut","rna","rdie","cdf")
    def __init__(self, N: int, n_alleles: int):
        self.a0   = np.empty(N, dtype=np.int64)
        self.a1   = np.empty(N, dtype=np.int64)
        self.l0   = np.empty(N, dtype=np.int64)
        self.l1   = np.empty(N, dtype=np.int64)
        self.f0   = np.empty(N, dtype=np.float64)
        self.f1   = np.empty(N, dtype=np.float64)
        self.rsel = np.empty(N, dtype=np.float64)
        self.rmut = np.empty(N, dtype=np.float64)
        self.rna  = np.empty(N, dtype=np.int64)
        self.rdie = np.empty(N, dtype=np.int64)
        self.cdf  = np.empty(N, dtype=np.float64)  # preallocated CDF buffer


# ---------------------------------------------------------------------------
# Initial state factory
# ---------------------------------------------------------------------------

def make_initial_state(N: int, n_alleles: int, rng: np.random.Generator,
                       resource: float = 0.0,
                       fitness_fn=None) -> PopState:
    alleles  = rng.integers(0, n_alleles, size=N).astype(np.int64)
    lineages = np.arange(N, dtype=np.int64)
    fitness  = (fitness_fn(alleles, resource) if fitness_fn
                else np.ones(N, dtype=np.float64))
    return PopState(alleles=alleles, lineages=lineages, fitness=fitness,
                    resource=resource, t=0, n_founders=N)


# ---------------------------------------------------------------------------
# Regime 1: Neutral WF
# ---------------------------------------------------------------------------

def run_neutral_wf(N: int, n_alleles: int, mu: float, T: int,
                   seed: int, full_traj: bool = False) -> list[PopState]:
    rng   = np.random.default_rng(seed)
    buf   = _Buf(N, n_alleles)
    state = make_initial_state(N, n_alleles, rng)

    np.copyto(buf.a0, state.alleles)
    np.copyto(buf.l0, state.lineages)
    _fitness_neutral_k(N, buf.f0)

    traj = [state.copy()]
    for _ in range(T):
        rng.random(out=buf.rsel)
        rng.random(out=buf.rmut)
        np.copyto(buf.rna, rng.integers(0, n_alleles, size=N))
        _wf_step_k(buf.a0, buf.l0, buf.f0, N, mu, n_alleles,
                   buf.rsel, buf.rmut, buf.rna, buf.cdf, buf.a1, buf.l1)
        _fitness_neutral_k(N, buf.f1)
        buf.a0, buf.a1 = buf.a1, buf.a0
        buf.l0, buf.l1 = buf.l1, buf.l0
        buf.f0, buf.f1 = buf.f1, buf.f0
        state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                         fitness=buf.f0.copy(), resource=0.0,
                         t=state.t+1, n_founders=N)
        if full_traj:
            traj.append(state.copy())
    if not full_traj:
        traj.append(state.copy())
    return traj


# ---------------------------------------------------------------------------
# Regime 2: Selected WF
# ---------------------------------------------------------------------------

def run_selected_wf(N: int, n_alleles: int, mu: float, T: int,
                    s: float, favored: int, seed: int,
                    full_traj: bool = False) -> list[PopState]:
    rng   = np.random.default_rng(seed)
    buf   = _Buf(N, n_alleles)
    state = make_initial_state(N, n_alleles, rng)

    np.copyto(buf.a0, state.alleles)
    np.copyto(buf.l0, state.lineages)
    _fitness_selected_k(buf.a0, N, s, favored, buf.f0)

    # Rebuild state with correct realized fitness before saving t0 snapshot
    state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                     fitness=buf.f0.copy(), resource=0.0, t=0, n_founders=N)
    traj = [state.copy()]
    for _ in range(T):
        rng.random(out=buf.rsel)
        rng.random(out=buf.rmut)
        np.copyto(buf.rna, rng.integers(0, n_alleles, size=N))
        _wf_step_k(buf.a0, buf.l0, buf.f0, N, mu, n_alleles,
                   buf.rsel, buf.rmut, buf.rna, buf.cdf, buf.a1, buf.l1)
        _fitness_selected_k(buf.a1, N, s, favored, buf.f1)
        buf.a0, buf.a1 = buf.a1, buf.a0
        buf.l0, buf.l1 = buf.l1, buf.l0
        buf.f0, buf.f1 = buf.f1, buf.f0
        state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                         fitness=buf.f0.copy(), resource=0.0,
                         t=state.t+1, n_founders=N)
        if full_traj:
            traj.append(state.copy())
    if not full_traj:
        traj.append(state.copy())
    return traj


# ---------------------------------------------------------------------------
# Regime 3: Moran
# ---------------------------------------------------------------------------

def run_moran(N: int, n_alleles: int, mu: float, T: int,
              s: float, favored: int, seed: int,
              full_traj: bool = False) -> list[PopState]:
    """1 generation = N Moran steps. Exact per-step dynamics."""
    rng   = np.random.default_rng(seed)
    buf   = _Buf(N, n_alleles)
    state = make_initial_state(N, n_alleles, rng)

    alleles  = state.alleles.copy()
    lineages = state.lineages.copy()
    # Fix: compute correct t=0 fitness before saving snapshot
    _fitness_selected_k(alleles, N, s, favored, buf.f0)
    state = PopState(alleles=alleles.copy(), lineages=lineages.copy(),
                     fitness=buf.f0.copy(), resource=0.0, t=0, n_founders=N)
    traj = [state.copy()]

    for gen in range(T):
        _fitness_selected_k(alleles, N, s, favored, buf.f0)
        rng.random(out=buf.rsel)
        np.copyto(buf.rdie, rng.integers(0, N, size=N))
        rng.random(out=buf.rmut)
        np.copyto(buf.rna, rng.integers(0, n_alleles, size=N))
        _moran_step_k(alleles, lineages, buf.f0, N, mu, n_alleles,
                      buf.rsel, buf.rdie, buf.rmut, buf.rna, buf.cdf)
        _fitness_selected_k(alleles, N, s, favored, buf.f0)
        state = PopState(alleles=alleles.copy(), lineages=lineages.copy(),
                         fitness=buf.f0.copy(), resource=0.0,
                         t=gen+1, n_founders=N)
        if full_traj:
            traj.append(state.copy())
    if not full_traj:
        traj.append(state.copy())
    return traj


# ---------------------------------------------------------------------------
# Regime 4: Frequency-dependent selection
# ---------------------------------------------------------------------------

def run_freq_dep(N: int, n_alleles: int, mu: float, T: int,
                 alpha: float, seed: int,
                 full_traj: bool = False) -> list[PopState]:
    rng   = np.random.default_rng(seed)
    buf   = _Buf(N, n_alleles)
    state = make_initial_state(N, n_alleles, rng)

    np.copyto(buf.a0, state.alleles)
    np.copyto(buf.l0, state.lineages)
    _fitness_freq_dep_k(buf.a0, N, n_alleles, alpha, buf.f0)

    state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                     fitness=buf.f0.copy(), resource=0.0, t=0, n_founders=N)
    traj = [state.copy()]
    for _ in range(T):
        rng.random(out=buf.rsel)
        rng.random(out=buf.rmut)
        np.copyto(buf.rna, rng.integers(0, n_alleles, size=N))
        _wf_step_k(buf.a0, buf.l0, buf.f0, N, mu, n_alleles,
                   buf.rsel, buf.rmut, buf.rna, buf.cdf, buf.a1, buf.l1)
        _fitness_freq_dep_k(buf.a1, N, n_alleles, alpha, buf.f1)
        buf.a0, buf.a1 = buf.a1, buf.a0
        buf.l0, buf.l1 = buf.l1, buf.l0
        buf.f0, buf.f1 = buf.f1, buf.f0
        state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                         fitness=buf.f0.copy(), resource=0.0,
                         t=state.t+1, n_founders=N)
        if full_traj:
            traj.append(state.copy())
    if not full_traj:
        traj.append(state.copy())
    return traj


# ---------------------------------------------------------------------------
# Regime 5: Eco-evolutionary coupled
# ---------------------------------------------------------------------------

def run_eco_evol(N: int, n_alleles: int, mu: float, T: int,
                 eco: dict, seed: int,
                 full_traj: bool = False) -> list[PopState]:
    """Density-dependent consumption (Holling type I). R* ≈ 0.58."""
    rng = np.random.default_rng(seed)
    buf = _Buf(N, n_alleles)
    K   = float(eco["K"]); r_R = float(eco["r_R"])
    bf  = np.asarray(eco["allele_base_fitness"], dtype=np.float64)
    ac  = np.asarray(eco["allele_consumption"],  dtype=np.float64)
    R   = float(eco["R0"])

    state = make_initial_state(N, n_alleles, rng, resource=R)
    np.copyto(buf.a0, state.alleles)
    np.copyto(buf.l0, state.lineages)
    _fitness_eco_k(buf.a0, N, bf, R, K, buf.f0)

    state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                     fitness=buf.f0.copy(), resource=R, t=0, n_founders=N)
    traj = [state.copy()]
    for _ in range(T):
        # Resource update via njit kernel
        mean_ac = _mean_consumption_k(buf.a0, ac)
        R = R + r_R * R * (1.0 - R / K) - mean_ac * R / K
        if R < 0.0:
            R = 0.0

        rng.random(out=buf.rsel)
        rng.random(out=buf.rmut)
        np.copyto(buf.rna, rng.integers(0, n_alleles, size=N))
        _wf_step_k(buf.a0, buf.l0, buf.f0, N, mu, n_alleles,
                   buf.rsel, buf.rmut, buf.rna, buf.cdf, buf.a1, buf.l1)
        _fitness_eco_k(buf.a1, N, bf, R, K, buf.f1)
        buf.a0, buf.a1 = buf.a1, buf.a0
        buf.l0, buf.l1 = buf.l1, buf.l0
        buf.f0, buf.f1 = buf.f1, buf.f0
        state = PopState(alleles=buf.a0.copy(), lineages=buf.l0.copy(),
                         fitness=buf.f0.copy(), resource=R,
                         t=state.t+1, n_founders=N)
        if full_traj:
            traj.append(state.copy())
    if not full_traj:
        traj.append(state.copy())
    return traj


# ---------------------------------------------------------------------------
# Regime 6: Group-structured / multilevel selection
# ---------------------------------------------------------------------------

def run_group_structured(N: int, n_alleles: int, mu: float, T: int,
                          s: float, group: dict, seed: int,
                          full_traj: bool = False) -> list[PopState]:
    """
    Each generation:
      1. Within-group WF for all groups.
      2. One between-group event: reproducing group replaces dying group.
    Lineage tags propagate through group bottlenecks explicitly.
    """
    rng      = np.random.default_rng(seed)
    n_groups = group["n_groups"]
    gs       = group["group_size"]
    s_within = group["s_within"]
    s_between= group["s_between"]
    assert N == n_groups * gs

    alleles  = rng.integers(0, n_alleles, N).astype(np.int64)
    lineages = np.arange(N, dtype=np.int64)
    f_all    = np.empty(N, dtype=np.float64)
    _fitness_selected_k(alleles, N, s_within, FAVORED_ALLELE, f_all)

    state = PopState(alleles=alleles.copy(), lineages=lineages.copy(),
                     fitness=f_all.copy(), resource=0.0, t=0, n_founders=N)
    traj  = [state.copy()]

    # Per-group buffers (reused every generation)
    ga   = np.empty(gs, dtype=np.int64)
    gcdf = np.empty(gs, dtype=np.float64)  # CDF buffer for per-group WF step
    gl   = np.empty(gs, dtype=np.int64)
    gf   = np.empty(gs, dtype=np.float64)
    ga1  = np.empty(gs, dtype=np.int64)
    gl1  = np.empty(gs, dtype=np.int64)
    rsel = np.empty(gs, dtype=np.float64)
    rmut = np.empty(gs, dtype=np.float64)
    rna  = np.empty(gs, dtype=np.int64)
    group_fit = np.empty(n_groups, dtype=np.float64)

    for gen in range(T):
        # Step 1: within-group WF
        for g in range(n_groups):
            idx = slice(g * gs, (g + 1) * gs)
            np.copyto(ga, alleles[idx])
            np.copyto(gl, lineages[idx])
            _fitness_selected_k(ga, gs, s_within, FAVORED_ALLELE, gf)
            rng.random(out=rsel)
            rng.random(out=rmut)
            np.copyto(rna, rng.integers(0, n_alleles, size=gs))
            _wf_step_k(ga, gl, gf, gs, mu, n_alleles,
                       rsel, rmut, rna, gcdf, ga1, gl1)
            alleles[idx]  = ga1
            lineages[idx] = gl1

        # Step 2: one between-group event
        for g in range(n_groups):
            idx = slice(g * gs, (g + 1) * gs)
            _fitness_selected_k(alleles[idx], gs, s_between, FAVORED_ALLELE, gf)
            group_fit[g] = gf.mean()

        cdf_g   = np.cumsum(group_fit / group_fit.sum())
        repro_g = int(np.searchsorted(cdf_g, rng.random()))
        die_g   = int(rng.integers(0, n_groups))

        if repro_g != die_g:
            src = slice(repro_g * gs, (repro_g + 1) * gs)
            dst = slice(die_g   * gs, (die_g   + 1) * gs)
            pidx = rng.integers(0, gs, size=gs)
            alleles[dst]  = alleles[src][pidx]
            lineages[dst] = lineages[src][pidx]
            n_mut = 0
            rmut_g = rng.random(gs)
            for i in range(gs):
                if rmut_g[i] < mu:
                    alleles[dst.start + i] = int(rng.integers(0, n_alleles))
                    n_mut += 1

        _fitness_selected_k(alleles, N, s_within, FAVORED_ALLELE, f_all)
        state = PopState(alleles=alleles.copy(), lineages=lineages.copy(),
                         fitness=f_all.copy(), resource=0.0,
                         t=gen+1, n_founders=N)
        if full_traj:
            traj.append(state.copy())
    if not full_traj:
        traj.append(state.copy())
    return traj


# ---------------------------------------------------------------------------
# Regime dispatcher
# full_traj=False returns [t0, tT] — default for battery (saves memory).
# full_traj=True  returns all T+1 states — for pilot diagnostics only.
# ---------------------------------------------------------------------------

def run_regime(regime: str, seed: int,
               params: dict = BATTERY_PARAMS,
               group:  dict = GROUP_PARAMS,
               eco:    dict = ECO_PARAMS,
               full_traj: bool = False) -> list[PopState]:
    N  = params["N"];  k  = params["n_alleles"]
    mu = params["mu"]; T  = params["T"]; s = params["s"]
    if regime == "neutral_wf":
        return run_neutral_wf(N, k, mu, T, seed, full_traj)
    elif regime == "selected_wf":
        return run_selected_wf(N, k, mu, T, s, FAVORED_ALLELE, seed, full_traj)
    elif regime == "moran":
        return run_moran(N, k, mu, T, s, FAVORED_ALLELE, seed, full_traj)
    elif regime == "freq_dep":
        return run_freq_dep(N, k, mu, T, FREQ_DEP_ALPHA, seed, full_traj)
    elif regime == "eco_evol":
        return run_eco_evol(N, k, mu, T, eco, seed, full_traj)
    elif regime == "group_structured":
        return run_group_structured(N, k, mu, T, s, group, seed, full_traj)
    else:
        raise ValueError(f"Unknown regime: {regime}")


# ---------------------------------------------------------------------------
# Warm-up: trigger JIT compilation before any timed run.
# Call once at battery startup (takes ~2s; cached after first run).
# ---------------------------------------------------------------------------

def warmup_jit(verbose: bool = True) -> None:
    """Compile all 7 njit kernels with N=10. Cached on disk after first call."""
    if verbose:
        print("Warming up Numba JIT kernels (cached after first run)...")
    N, k = 10, 4
    a  = np.zeros(N, dtype=np.int64)
    l  = np.arange(N, dtype=np.int64)
    f  = np.ones(N,  dtype=np.float64)
    of = np.empty(N, dtype=np.float64)
    oa = np.empty(N, dtype=np.int64)
    ol = np.empty(N, dtype=np.int64)
    cdf = np.empty(N, dtype=np.float64)
    rng = np.random.default_rng(0)
    rs  = rng.random(N); rm = rng.random(N)
    rna = rng.integers(0, k, N); rd = rng.integers(0, N, N)
    bf  = ECO_PARAMS["allele_base_fitness"]

    _fitness_neutral_k(N, of)
    _fitness_selected_k(a, N, 0.05, 0, of)
    _fitness_freq_dep_k(a, N, k, 0.5, of)
    _fitness_eco_k(a, N, bf, 0.8, 1.0, of)
    _mean_consumption_k(a, np.asarray(ECO_PARAMS["allele_consumption"]))
    _wf_step_k(a, l, f, N, 0.005, k, rs, rm, rna, cdf, oa, ol)
    _moran_step_k(a, l, f, N, 0.005, k, rs, rd, rm, rna, cdf)
    if verbose:
        print("  7 kernels compiled.")
