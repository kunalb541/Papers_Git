#!/usr/bin/env python3
"""
nbody_stress.py — rigorous stress-test battery for the N-body ODD result
=========================================================================

An astronomer should not believe a single run at one N, one softening,
one grid, one k, with no confidence intervals, no velocity information,
and no N-scaling.  This battery tests all of those.

Stress dimensions
  1. N scaling          : 256, 512, 1024, 2048
  2. Softening length   : eps ∈ {0.02, 0.05, 0.10}
  3. Coarse grid size   : g ∈ {4, 8, 16}  (tests if result is grid-artifact)
  4. kNN neighbour k    : k ∈ {8, 16, 32} swept via --k-fine (tests null is k-independent)
  5. Prediction horizon : early (h=100), mid (h=300), late (h=600/1000 steps)
  6. Prediction targets : d_coarse (3 grid sizes, 3 horizons), d_hmr
  7. Fine observables   :
       fine_knn_all       — kNN density over all particles (not top-k)
       fine_pk_small      — small-scale Fourier power
       fine_close_pairs   — close-pair fraction at 4*eps
       fine_vel_disp      — local velocity dispersion (phase-space, k nearest neighbours)
       fof_ngroups        — friends-of-friends group count (b=0.2*mean_sep)
  8. Bootstrap 95% CI   : 1000 resamples on every Pearson r
  9. Partial r          : controlling for initial virial ratio (removes IC quality effect)
 10. IC families        : plummer3d, cold_clumpy3d, hernquist3d (new), bimodal3d (new)

Output
  outputs/stress/results_full.csv
  outputs/stress/summary_table.txt   ← the publishable table
  outputs/stress/ci_table.json       ← all CIs
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ── re-use core physics from nbody_3d ─────────────────────────────────────────
from nbody_3d import (_HAS_NUMBA, Array, SimConfig, _cic_deposit3,
                      _numba_direct_acc, _worker_init, acceleration,
                      half_mass_radius, initial_conditions, kinetic_energy,
                      min_image, potential_energy_direct, virial_ratio)

# ── Stress config ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StressConfig:
    """One cell in the stress-test grid."""
    model:       str
    init:        str
    seed:        int
    n:           int
    steps:       int
    dt:          float   = 0.005
    G:           float   = 1.0
    eps:         float   = 0.05
    box_size:    float   = 2.0
    pm_grid:     int     = 32
    coarse_grid: int     = 8
    k_fine:      int     = 16
    h_early:     int     = 100
    h_mid:       int     = 300
    plummer_a:   float   = 0.20
    vel_scale:   float   = 0.35
    cold_scale:  float   = 0.05
    clump_count: int     = 5
    fof_b:       float   = 0.20  # FoF linking length in units of mean separation


# ── Additional IC families ─────────────────────────────────────────────────────

def _sphere_directions(rng, n):
    cth = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    sth = np.sqrt(1.0 - cth * cth)
    return np.column_stack([sth * np.cos(phi), sth * np.sin(phi), cth])


def sample_hernquist3d(rng, cfg: StressConfig) -> Tuple[Array, Array]:
    """Hernquist (1990) profile: ρ ∝ 1/(r(r+a)^3).
    CDF: M(<r)/M = r^2/(r+a)^2  →  sqrt(u) = r/(r+a)  →  r = a*sqrt(u)/(1-sqrt(u))
    """
    a = cfg.plummer_a
    u = rng.uniform(1e-6, 1.0 - 1e-6, cfg.n)
    sq = np.sqrt(u)
    r = a * sq / (1.0 - sq)                          # fixed: was a*u/(1-sqrt(u))
    r = np.clip(r, 0.0, 5.0 * cfg.box_size)
    center = np.full(3, cfg.box_size / 2.0)
    pos = center + r[:, None] * _sphere_directions(rng, cfg.n)
    sigma = math.sqrt(cfg.G / (6.0 * a))
    vel = rng.normal(0.0, sigma * 0.8, (cfg.n, 3))
    vel -= np.mean(vel, axis=0)
    return pos, vel


def sample_bimodal3d(rng, cfg: StressConfig) -> Tuple[Array, Array]:
    """Two equal-mass Plummer spheres on a collision course.
    Tests whether the result holds when large-scale structure is bimodal.
    """
    a = cfg.plummer_a * 0.5
    half = cfg.n // 2
    offset = cfg.box_size * 0.25

    def _plummer_half(n, center):
        u = rng.uniform(1e-6, 1.0 - 1e-6, n)
        r = a / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
        return np.array(center) + r[:, None] * _sphere_directions(rng, n)

    c1 = [cfg.box_size / 2.0 - offset, cfg.box_size / 2.0, cfg.box_size / 2.0]
    c2 = [cfg.box_size / 2.0 + offset, cfg.box_size / 2.0, cfg.box_size / 2.0]
    p1 = _plummer_half(half, c1)
    p2 = _plummer_half(cfg.n - half, c2)
    pos = np.vstack([p1, p2])

    sigma = math.sqrt(cfg.G / (6.0 * a))
    vel = rng.normal(0.0, sigma, (cfg.n, 3))
    # Give each group a small infall velocity toward the midpoint
    vel[:half,  0] += +0.1
    vel[half:,  0] += -0.1
    vel -= np.mean(vel, axis=0)
    return pos, vel


def get_initial_conditions(cfg: StressConfig) -> Tuple[Array, Array]:
    rng = np.random.default_rng(cfg.seed)
    sc = SimConfig(
        model=cfg.model, integrator="leapfrog_kdk", init=cfg.init,
        seed=cfg.seed, n=cfg.n, steps=cfg.steps, dt=cfg.dt,
        G=cfg.G, eps=cfg.eps, box_size=cfg.box_size, pm_grid=cfg.pm_grid,
        coarse_grid=cfg.coarse_grid, top_k=cfg.k_fine,
        plummer_a=cfg.plummer_a, vel_scale=cfg.vel_scale,
        cold_scale=cfg.cold_scale, clump_count=cfg.clump_count,
    )
    if cfg.init == "hernquist3d":
        return sample_hernquist3d(rng, cfg)
    if cfg.init == "bimodal3d":
        return sample_bimodal3d(rng, cfg)
    return initial_conditions(sc)


def get_simconfig(cfg: StressConfig) -> SimConfig:
    return SimConfig(
        model=cfg.model, integrator="leapfrog_kdk", init=cfg.init,
        seed=cfg.seed, n=cfg.n, steps=cfg.steps, dt=cfg.dt,
        G=cfg.G, eps=cfg.eps, box_size=cfg.box_size, pm_grid=cfg.pm_grid,
        coarse_grid=cfg.coarse_grid, top_k=cfg.k_fine,
        plummer_a=cfg.plummer_a, vel_scale=cfg.vel_scale,
        cold_scale=cfg.cold_scale, clump_count=cfg.clump_count,
    )


# ── Fine observables ───────────────────────────────────────────────────────────

def _knn_r2(pos: Array, k: int, periodic: bool, box_size: float) -> Array:
    """Return k-th neighbour squared distances for all particles."""
    if len(pos) < 2 or k < 1:
        return np.full(len(pos), np.nan)
    dx = pos[:, None, :] - pos[None, :, :]
    if periodic:
        dx = min_image(dx, box_size)
    r2 = np.sum(dx * dx, axis=-1)
    np.fill_diagonal(r2, np.inf)
    return np.partition(r2, kth=k - 1, axis=1)[:, k - 1]


def _filter_box(pos: Array, vel: Optional[Array],
                periodic: bool, box_size: float):
    """For isolated runs: restrict to particles inside [0, box_size)^3.

    Keeps coarse, Fourier, and pairwise fine observables on the same
    boundary.  Periodic models are unaffected (all particles in box).
    Returns (pos_in, vel_in) — vel_in is None if vel is None.
    """
    if periodic:
        return pos, vel
    mask = np.all((pos >= 0.0) & (pos < box_size), axis=1)
    return pos[mask], (vel[mask] if vel is not None else None)


def obs_fine_knn_all(pos: Array, k: int, eps: float,
                     periodic: bool, box_size: float) -> float:
    """Mean kNN local density over ALL in-box particles (3-D sphere)."""
    pos, _ = _filter_box(pos, None, periodic, box_size)
    if len(pos) < 2:
        return float("nan")
    k_eff  = min(k, len(pos) - 1)
    kth_r2 = _knn_r2(pos, k_eff, periodic, box_size)
    rk     = np.sqrt(np.maximum(kth_r2, eps ** 2))
    return float(np.mean(k_eff / (4.0 / 3.0 * math.pi * rk ** 3)))


def obs_fine_pk_small(pos: Array, periodic: bool, box_size: float,
                      pm_grid: int, mass_pp: float) -> float:
    """Power in high-k (small-scale) density modes.
    Isolated models use discard-not-wrap CIC to avoid aliasing tail particles.
    """
    g = pm_grid
    L = box_size
    cell_vol = (L / g) ** 3
    rho   = _cic_deposit3(pos, mass_pp, L, g, periodic=periodic) / cell_vol
    delta = rho - np.mean(rho)
    dk    = np.fft.rfftn(delta)
    pk    = np.abs(dk) ** 2
    freqs  = np.fft.fftfreq(g) * g
    rfreqs = np.fft.rfftfreq(g) * g
    FX, FY, FZ = np.meshgrid(freqs, freqs, rfreqs, indexing="ij")
    k_abs = np.sqrt(FX ** 2 + FY ** 2 + FZ ** 2)
    return float(np.sum(pk[k_abs > g / 4.0]))


def obs_fine_close_pairs(pos: Array, eps: float,
                         periodic: bool, box_size: float) -> float:
    """Fraction of in-box pairs within 4*eps."""
    pos, _ = _filter_box(pos, None, periodic, box_size)
    if len(pos) < 2:
        return float("nan")
    n       = len(pos)
    thresh2 = (4.0 * eps) ** 2
    dx      = pos[:, None, :] - pos[None, :, :]
    if periodic:
        dx = min_image(dx, box_size)
    r2 = np.sum(dx * dx, axis=-1)
    np.fill_diagonal(r2, np.inf)
    return float(np.sum(r2 < thresh2)) / 2.0 / (n * (n - 1) / 2.0)


def obs_fine_local_vel_disp(pos: Array, vel: Array, k: int,
                             periodic: bool, box_size: float) -> float:
    """Mean LOCAL velocity dispersion over in-box particles."""
    pos, vel = _filter_box(pos, vel, periodic, box_size)
    k_eff = min(k, len(pos) - 1)
    if k_eff < 1:
        return float("nan")
    dx = pos[:, None, :] - pos[None, :, :]
    if periodic:
        dx = min_image(dx, box_size)
    r2 = np.sum(dx * dx, axis=-1)
    np.fill_diagonal(r2, np.inf)
    nn_idx    = np.argpartition(r2, kth=k_eff - 1, axis=1)[:, :k_eff]
    v_nn      = vel[nn_idx]
    v_mean    = np.mean(v_nn, axis=1, keepdims=True)
    local_std = np.sqrt(np.mean(np.sum((v_nn - v_mean) ** 2, axis=-1), axis=1))
    return float(np.mean(local_std))


def obs_fof_groups(pos: Array, periodic: bool, box_size: float,
                   fof_b: float) -> int:
    """Friends-of-friends group count over in-box particles."""
    pos, _ = _filter_box(pos, None, periodic, box_size)
    n = len(pos)
    if n < 2:
        return 0
    if periodic:
        mean_sep = box_size / n ** (1.0 / 3.0)
    else:
        span = np.max(pos, axis=0) - np.min(pos, axis=0)
        vol  = max(np.prod(span), 1e-10)
        mean_sep = (vol / n) ** (1.0 / 3.0)

    link2  = (fof_b * mean_sep) ** 2
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    dx = pos[:, None, :] - pos[None, :, :]
    if periodic:
        dx = min_image(dx, box_size)
    r2 = np.sum(dx * dx, axis=-1)
    np.fill_diagonal(r2, np.inf)
    ii, jj = np.where((r2 < link2) & (r2 > 0))
    for a, b in zip(ii, jj):
        union(int(a), int(b))
    return len({find(i) for i in range(n)})


def obs_coarse_var(pos: Array, cfg: StressConfig,
                   grid: int, periodic: bool) -> float:
    """Density variance on a fixed g^3 grid.

    Both isolated coarse and isolated Fourier use discard-outside so both
    observables share the same boundary convention.
    """
    L = cfg.box_size
    if periodic:
        p = np.mod(pos, L) / L
    else:
        inside = np.all((pos >= 0.0) & (pos < L), axis=1)
        if not np.any(inside):
            return 0.0
        p = pos[inside] / L
    idx = np.floor(p * grid).astype(int)
    idx = np.clip(idx, 0, grid - 1)
    hist = np.zeros((grid, grid, grid))
    np.add.at(hist, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    return float(np.var(hist))


# ── Single stress run ──────────────────────────────────────────────────────────

@dataclass
class StressResult:
    # identity
    model: str; init: str; seed: int; n: int
    eps: float; coarse_grid: int; k_fine: int; steps: int
    # conservation
    energy_rel_drift: Optional[float]
    virial_0: float; virial_f: float
    # coarse obs (three grid sizes at t=0)
    coarse_g4_0:  float; coarse_g8_0:  float; coarse_g16_0: float
    # fine obs at t=0
    fine_knn_all_0:   float
    fine_pk_0:        float
    fine_close_0:     float
    fine_vel_disp_0:  float
    fine_fof_0:       int
    # targets (three horizons, three coarse grids)
    d_coarse_g8_early:  float; d_coarse_g8_mid:  float; d_coarse_g8_late: float
    d_coarse_g4_early:  float; d_coarse_g4_late:  float
    d_coarse_g16_early: float; d_coarse_g16_late: float
    d_hmr_early:        float; d_hmr_late:         float
    # status
    status: str; message: str

def apply_pbc_local(pos: Array, box_size: float, periodic: bool) -> Array:
    if periodic:
        return np.mod(pos, box_size)
    return pos


def integrate_leapfrog_local(pos0: Array, vel0: Array, mass: float,
                             cfg: SimConfig, snap_steps: List[int],
                             use_numba: bool) -> Dict[int, Tuple[Array, Array]]:
    """Local KDK leapfrog using THIS file's acceleration/boundary rules."""
    snaps: Dict[int, Tuple[Array, Array]] = {}
    pos = pos0.copy()
    vel = vel0.copy()
    periodic = cfg.model in ("direct_periodic", "pm_periodic")
    dt = cfg.dt

    acc = acceleration(pos, mass, cfg, use_numba)

    if 0 in snap_steps:
        snaps[0] = (pos.copy(), vel.copy())

    for step in range(1, cfg.steps + 1):
        vel = vel + 0.5 * dt * acc
        pos = apply_pbc_local(pos + dt * vel, cfg.box_size, periodic)
        acc = acceleration(pos, mass, cfg, use_numba)
        vel = vel + 0.5 * dt * acc

        if step in snap_steps:
            snaps[step] = (pos.copy(), vel.copy())

    return snaps


def run_stress(cfg: StressConfig, use_numba: bool = False) -> Dict:
    try:
        sc       = get_simconfig(cfg)
        periodic = cfg.model in ("direct_periodic", "pm_periodic")
        mass     = 1.0 / cfg.n

        pos0, vel0 = get_initial_conditions(cfg)

        # Energy at t=0
        ke0 = kinetic_energy(vel0, mass)
        pe0 = potential_energy_direct(pos0, mass, sc)
        e0  = ke0 + pe0

        # Initial observables — compute once
        def _all_obs(pos, vel):
            return {
                "cg4":      obs_coarse_var(pos, cfg, 4,  periodic),
                "cg8":      obs_coarse_var(pos, cfg, 8,  periodic),
                "cg16":     obs_coarse_var(pos, cfg, 16, periodic),
                "knn_all":  obs_fine_knn_all(pos, cfg.k_fine, cfg.eps,
                                             periodic, cfg.box_size),
                "pk":       obs_fine_pk_small(pos, periodic, cfg.box_size,
                                              cfg.pm_grid, mass),
                "close":    obs_fine_close_pairs(pos, cfg.eps,
                                                 periodic, cfg.box_size),
                "vel_disp": obs_fine_local_vel_disp(pos, vel, cfg.k_fine,
                                                    periodic, cfg.box_size),
                "fof":      obs_fof_groups(pos, periodic, cfg.box_size, cfg.fof_b),
                "hmr":      half_mass_radius(pos, periodic),
                "ke":       kinetic_energy(vel, mass),
                "pe":       potential_energy_direct(pos, mass, sc),
            }

        snap_steps = sorted({0,
                             min(cfg.h_early, cfg.steps),
                             min(cfg.h_mid,   cfg.steps),
                             cfg.steps})
        snaps = integrate_leapfrog_local(pos0, vel0, mass, sc,
                                 snap_steps, use_numba)

        h_e = min(cfg.h_early, cfg.steps)
        h_m = min(cfg.h_mid,   cfg.steps)

        o0  = _all_obs(*snaps[0])
        oe  = _all_obs(*snaps[h_e])
        om  = _all_obs(*snaps[h_m])
        of_ = _all_obs(*snaps[cfg.steps])

        # Energy drift
        ef = of_["ke"] + of_["pe"]
        e_drift = None
        if cfg.model != "pm_periodic":
            e_drift = float(abs(ef - e0) / max(abs(e0), 1e-30))

        # Virial ratio: only meaningful for direct models where PE is exact.
        # For pm_periodic, direct isolated PE is a different Hamiltonian than
        # the periodic PM field — using it would contaminate partial-r analysis.
        if cfg.model == "pm_periodic":
            virial_0 = float("nan")
            virial_f = float("nan")
        else:
            virial_0 = virial_ratio(o0["ke"], o0["pe"])
            virial_f = virial_ratio(of_["ke"], of_["pe"])

        return asdict(StressResult(
            model=cfg.model, init=cfg.init, seed=cfg.seed, n=cfg.n,
            eps=cfg.eps, coarse_grid=cfg.coarse_grid,
            k_fine=cfg.k_fine, steps=cfg.steps,
            energy_rel_drift=e_drift,
            virial_0=virial_0, virial_f=virial_f,
            coarse_g4_0=o0["cg4"],  coarse_g8_0=o0["cg8"],
            coarse_g16_0=o0["cg16"],
            fine_knn_all_0=o0["knn_all"],
            fine_pk_0=o0["pk"],
            fine_close_0=o0["close"],
            fine_vel_disp_0=o0["vel_disp"],
            fine_fof_0=o0["fof"],
            d_coarse_g8_early  = oe["cg8"]  - o0["cg8"],
            d_coarse_g8_mid    = om["cg8"]  - o0["cg8"],
            d_coarse_g8_late   = of_["cg8"] - o0["cg8"],
            d_coarse_g4_early  = oe["cg4"]  - o0["cg4"],
            d_coarse_g4_late   = of_["cg4"] - o0["cg4"],
            d_coarse_g16_early = oe["cg16"] - o0["cg16"],
            d_coarse_g16_late  = of_["cg16"] - o0["cg16"],
            d_hmr_early        = (oe["hmr"]  - o0["hmr"])
                                  if math.isfinite(o0["hmr"]) else float("nan"),
            d_hmr_late         = (of_["hmr"] - o0["hmr"])
                                  if math.isfinite(o0["hmr"]) else float("nan"),
            status="ok", message="",
        ))

    except Exception as exc:
        nan = float("nan")
        return asdict(StressResult(
            model=cfg.model, init=cfg.init, seed=cfg.seed, n=cfg.n,
            eps=cfg.eps, coarse_grid=cfg.coarse_grid,
            k_fine=cfg.k_fine, steps=cfg.steps,
            energy_rel_drift=None, virial_0=nan, virial_f=nan,
            coarse_g4_0=nan, coarse_g8_0=nan, coarse_g16_0=nan,
            fine_knn_all_0=nan, fine_pk_0=nan, fine_close_0=nan,
            fine_vel_disp_0=nan, fine_fof_0=0,
            d_coarse_g8_early=nan, d_coarse_g8_mid=nan, d_coarse_g8_late=nan,
            d_coarse_g4_early=nan, d_coarse_g4_late=nan,
            d_coarse_g16_early=nan, d_coarse_g16_late=nan,
            d_hmr_early=nan, d_hmr_late=nan,
            status="error", message=str(exc),
        ))


# ── Statistics ─────────────────────────────────────────────────────────────────

def _clean(x, y):
    """Return aligned finite pairs."""
    a, b = [], []
    for xi, yi in zip(x, y):
        if (xi is not None and yi is not None
                and np.isfinite(xi) and np.isfinite(yi)):
            a.append(float(xi)); b.append(float(yi))
    return np.array(a), np.array(b)


def pearson_with_ci(x, y, n_boot=1000, seed=0):
    """Pearson r + 95% bootstrap CI.  Returns (r, lo, hi, n) or (nan,nan,nan,0)."""
    a, b = _clean(x, y)
    if len(a) < 5:
        return float("nan"), float("nan"), float("nan"), len(a)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None, None, None, len(a)  # degenerate
    r = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    boot_r = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        ai, bi = a[idx], b[idx]
        if np.std(ai) < 1e-12 or np.std(bi) < 1e-12:
            continue
        boot_r.append(float(np.corrcoef(ai, bi)[0, 1]))
    if len(boot_r) < 10:
        return r, float("nan"), float("nan"), len(a)
    lo, hi = float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))
    return r, lo, hi, len(a)


def partial_r(x, y, z):
    """Partial correlation r(x,y|z): residualise both on z, triple-aligned."""
    triples = [
        (xi, yi, zi)
        for xi, yi, zi in zip(x, y, z)
        if all(v is not None and np.isfinite(float(v)) for v in [xi, yi, zi])
    ]
    if len(triples) < 5:
        return float("nan")
    xa = np.array([t[0] for t in triples], dtype=float)
    ya = np.array([t[1] for t in triples], dtype=float)
    za = np.array([t[2] for t in triples], dtype=float)

    def _resid(v, u):
        if np.std(u) < 1e-12:
            return v - np.mean(v)
        slope = np.cov(v, u, ddof=0)[0, 1] / np.var(u)
        intercept = np.mean(v) - slope * np.mean(u)
        return v - (slope * u + intercept)

    rx = _resid(xa, za)
    ry = _resid(ya, za)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ── Analysis ───────────────────────────────────────────────────────────────────

FINE_OBS = [
    ("fine_knn_all_0",  "kNN-all"),
    ("fine_pk_0",       "Pk-small"),
    ("fine_close_0",    "ClosePairs"),
    ("fine_vel_disp_0", "VelDisp"),
    ("fine_fof_0",      "FoF-groups"),
]

COARSE_OBS = [
    ("coarse_g4_0",  "CoarseG4"),
    ("coarse_g8_0",  "CoarseG8"),
    ("coarse_g16_0", "CoarseG16"),
]

TARGETS = [
    ("d_coarse_g8_early",  "ΔC8-early"),
    ("d_coarse_g8_mid",    "ΔC8-mid"),
    ("d_coarse_g8_late",   "ΔC8-late"),
    ("d_coarse_g4_early",  "ΔC4-early"),
    ("d_coarse_g16_early", "ΔC16-early"),
    ("d_hmr_early",        "ΔHMR-early"),
    ("d_hmr_late",         "ΔHMR-late"),
]


def analyse(rows: List[Dict], n_boot: int = 1000) -> Dict:
    """For each (model, init, N, eps) cell compute the full correlation table."""
    ok = [r for r in rows if r["status"] == "ok"]

    # Group by the four stress dimensions
    groups: Dict[str, List[Dict]] = {}
    for r in ok:
        key = (f"N={r['n']}|eps={r['eps']}|k={r['k_fine']}"
               f"|model={r['model']}|init={r['init']}")
        groups.setdefault(key, []).append(r)

    out = {}
    for key, rs in groups.items():
        cell: Dict = {"n_reps": len(rs), "key": key}

        # For every (predictor, target) pair: r + CI + partial r on virial
        for pred_col, pred_name in FINE_OBS + COARSE_OBS:
            for tgt_col, tgt_name in TARGETS:
                px = [r.get(pred_col) for r in rs]
                ty = [r.get(tgt_col)  for r in rs]
                vr = [r.get("virial_0") for r in rs]
                r_val, lo, hi, n = pearson_with_ci(px, ty, n_boot=n_boot)
                pr = partial_r(px, ty, vr)
                cell[f"r_{pred_name}_{tgt_name}"] = r_val
                cell[f"ci_lo_{pred_name}_{tgt_name}"] = lo
                cell[f"ci_hi_{pred_name}_{tgt_name}"] = hi
                cell[f"pr_{pred_name}_{tgt_name}"] = pr

        # Verdict: for primary target ΔC8-early, does any fine obs beat best coarse?
        fine_rs  = [abs(cell.get(f"r_{pn}_ΔC8-early") or 0.0)
                    for _, pn in FINE_OBS
                    if cell.get(f"r_{pn}_ΔC8-early") is not None
                    and np.isfinite(cell.get(f"r_{pn}_ΔC8-early") or float("nan"))]
        coarse_rs = [abs(cell.get(f"r_{pn}_ΔC8-early") or 0.0)
                     for _, pn in COARSE_OBS
                     if cell.get(f"r_{pn}_ΔC8-early") is not None
                     and np.isfinite(cell.get(f"r_{pn}_ΔC8-early") or float("nan"))]

        best_fine   = max(fine_rs)   if fine_rs   else 0.0
        best_coarse = max(coarse_rs) if coarse_rs else 0.0
        cell["best_fine_r"]   = best_fine
        cell["best_coarse_r"] = best_coarse
        cell["fine_adv"]      = best_fine - best_coarse
        cell["verdict"]       = (
            "FINE"   if best_fine - best_coarse >  0.05 else
            "COARSE" if best_fine - best_coarse < -0.05 else
            "TIE"
        )
        out[key] = cell
    return out


# ── Formatted output ───────────────────────────────────────────────────────────

def print_summary(analysis: Dict) -> str:
    """Print the publishable summary table."""
    lines = []
    lines.append("=" * 120)
    lines.append("STRESS-TEST SUMMARY — fine vs coarse initial structure → future clustering (ΔC8-early)")
    lines.append("Each r value: point estimate [95% bootstrap CI]  |  partial-r controlling for initial virial ratio")
    lines.append("=" * 120)

    hdr = (f"{'Group':<42} {'n':>4}  "
           f"{'CoarseG8':>22}  "
           f"{'kNN-all':>22}  "
           f"{'VelDisp':>22}  "
           f"{'FoF':>22}  "
           f"{'verdict':>8}")
    lines.append(hdr)
    lines.append("─" * 120)

    def _fmt(r, lo, hi, pr):
        if r is None:
            return f"{'n/a':>22}"
        if not np.isfinite(r):
            return f"{'n/a':>22}"
        ci = (f"[{lo:+.2f},{hi:+.2f}]"
              if (lo is not None and np.isfinite(lo)) else "[  n/a  ]")
        return f"{r:+.3f} {ci} p{pr:+.2f}"

    for key in sorted(analysis.keys()):
        c = analysis[key]
        n = c["n_reps"]

        def _get(pred, tgt="ΔC8-early"):
            return (
                c.get(f"r_{pred}_{tgt}"),
                c.get(f"ci_lo_{pred}_{tgt}"),
                c.get(f"ci_hi_{pred}_{tgt}"),
                c.get(f"pr_{pred}_{tgt}"),
            )

        row = (f"  {key:<42} {n:>4}  "
               f"{_fmt(*_get('CoarseG8')):>22}  "
               f"{_fmt(*_get('kNN-all')):>22}  "
               f"{_fmt(*_get('VelDisp')):>22}  "
               f"{_fmt(*_get('FoF-groups')):>22}  "
               f"[{c['verdict']:>6}]")
        lines.append(row)

    lines.append("")
    lines.append("Columns: r = Pearson correlation with future ΔCoarse8 (early horizon)")
    lines.append("         [lo, hi] = 95% bootstrap CI (1000 resamples)")
    lines.append("         p = partial r controlling for initial virial ratio")
    lines.append("         COARSE: best fine r < best coarse r by >0.05 across all fine obs")
    lines.append("         FINE:   some fine obs beats all coarse by >0.05")
    lines.append("         TIE:    difference < 0.05")
    lines.append("")

    # N-scaling summary
    lines.append("N-SCALING: does coarse dominance persist as N grows?")
    lines.append(f"  {'N':>6}  {'k':>4}  {'model':>18}  {'init':>14}  "
                 f"{'r_CoarseG8':>12}  {'best_fine_r':>12}  {'verdict':>8}")
    lines.append("  " + "─" * 86)
    for key in sorted(analysis.keys()):
        c = analysis[key]
        parts = dict(p.split("=") for p in key.split("|") if "=" in p)
        n_val = parts.get("N", "?")
        k_val = parts.get("k", "?")
        model = parts.get("model", "?")
        init  = parts.get("init",  "?")
        rc  = c.get("r_CoarseG8_ΔC8-early")
        rc_s = f"{rc:+.3f}" if rc is not None and np.isfinite(rc) else "  n/a"
        bf   = c.get("best_fine_r", float("nan"))
        bf_s = f"{bf:+.3f}" if np.isfinite(bf) else "  n/a"
        lines.append(f"  {n_val:>6}  {k_val:>4}  {model:>18}  {init:>14}  "
                     f"{rc_s:>12}  {bf_s:>12}  [{c['verdict']:>6}]")

    text = "\n".join(lines)
    print(text)
    return text


# ── I/O ────────────────────────────────────────────────────────────────────────

def write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="N-body ODD stress-test battery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outdir",      default="outputs/stress")
    parser.add_argument("--n",           type=int, nargs="+",
                        default=[256, 512, 1024],
                        help="Particle counts to sweep")
    parser.add_argument("--eps",         type=float, nargs="+",
                        default=[0.05],
                        help="Softening lengths to sweep")
    parser.add_argument("--steps",       type=int,   default=600)
    parser.add_argument("--replicates",  type=int,   default=30)
    parser.add_argument("--workers",     type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--models",      nargs="+",
                        default=["direct_isolated", "pm_periodic"],
                        choices=["direct_isolated", "direct_periodic", "pm_periodic"])
    parser.add_argument("--inits",       nargs="+",
                        default=["plummer3d", "cold_clumpy3d",
                                 "hernquist3d", "bimodal3d"])
    parser.add_argument("--k-fine",      type=int, nargs="+",
                        default=[8, 16, 32],
                        help="kNN neighbour counts to sweep")
    parser.add_argument("--n-boot",      type=int, default=1000,
                        help="Bootstrap resamples for CIs")
    parser.add_argument("--use-numba",   dest="use_numba",
                        action="store_true", default=True)
    parser.add_argument("--no-numba",    dest="use_numba", action="store_false")
    args = parser.parse_args()

    if args.use_numba and not _HAS_NUMBA:
        print("numba not found — falling back to numpy")
        args.use_numba = False

    os.makedirs(args.outdir, exist_ok=True)
    seeds = [2000 + i for i in range(args.replicates)]

    configs: List[StressConfig] = []
    for n_val, eps_val, k_val, model, init, seed in product(
            args.n, args.eps, args.k_fine, args.models, args.inits, seeds):
        configs.append(StressConfig(
            model=model, init=init, seed=seed,
            n=n_val, steps=args.steps, eps=eps_val, k_fine=k_val,
        ))

    total   = len(configs)
    n_cells = (len(args.n) * len(args.eps) * len(args.k_fine)
               * len(args.models) * len(args.inits))
    print(
        f"Stress-test battery: {total} runs\n"
        f"  N            : {args.n}\n"
        f"  eps          : {args.eps}\n"
        f"  k_fine       : {args.k_fine}\n"
        f"  models       : {args.models}\n"
        f"  inits        : {args.inits}\n"
        f"  replicates   : {args.replicates} per cell  ({n_cells} cells)\n"
        f"  steps/dt     : {args.steps}/0.005\n"
        f"  workers      : {args.workers}\n"
        f"  bootstrap n  : {args.n_boot}\n"
        f"  numba        : {args.use_numba}\n"
    )

    rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init,
                             initargs=(args.use_numba,)) as ex:
        futs = {ex.submit(run_stress, cfg, args.use_numba): cfg
                for cfg in configs}
        with tqdm(total=total, unit="run", ncols=100) as pbar:
            for fut in as_completed(futs):
                res = fut.result()
                rows.append(res)
                tag = "✓" if res["status"] == "ok" else "✗"
                pbar.set_postfix_str(
                    f"{tag} N={res['n']:4d} {res['model'][:10]} "
                    f"{res['init'][:10]} s={res['seed']}"
                )
                pbar.update(1)
                if res["status"] == "error":
                    tqdm.write(f"  ✗ {res['model']} N={res['n']} "
                               f"seed={res['seed']}: {res['message']}")

    rows.sort(key=lambda r: (r["n"], r["eps"], r["k_fine"], r["model"], r["init"], r["seed"]))
    write_csv(os.path.join(args.outdir, "results_full.csv"), rows)
    print(f"\nRunning analysis with {args.n_boot} bootstrap resamples...")
    analysis = analyse(rows, n_boot=args.n_boot)

    summary_text = print_summary(analysis)
    with open(os.path.join(args.outdir, "summary_table.txt"), "w") as f:
        f.write(summary_text)
    with open(os.path.join(args.outdir, "ci_table.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=lambda x: None
                  if x is None else (float(x) if np.isfinite(x) else None))

    print(f"\nAll outputs → {args.outdir}/")


if __name__ == "__main__":
    main()
