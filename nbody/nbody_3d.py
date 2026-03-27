#!/usr/bin/env python3
"""
3-D standalone N-body hard-test battery — ODD program
======================================================

Models
  direct_isolated   softened direct summation, open 3-D space
  direct_periodic   softened direct summation, minimum-image PBC
  pm_periodic       particle-mesh FFT Poisson solver, periodic

Integrators
  leapfrog_kdk      Kick-Drift-Kick symplectic (energy-conserving by design)
  rk4               4th-order Runge-Kutta (higher-order, not symplectic)

Initial conditions
  plummer3d         3-D Plummer sphere (physically motivated)
  uniform3d         uniform random cube
  cold_clumpy3d     multi-clump cold initial condition

Observables (for ODD battery)
  coarse_var        variance of density on 8^3 coarse grid
  fine_knn          mean k-NN density of top-k densest particles (3-D sphere volume)
  hmr               half-mass radius R_50
  energy / momentum / angular momentum conservation diagnostics
  virial ratio      2*KE / |PE|

ODD correlation summary
  Across replicates, computes Pearson r between:
    fine_knn_0    vs  delta_coarse_early / delta_hmr_early
    coarse_var_0  vs  delta_coarse_early / delta_hmr_early
  Direct empirical test of whether fine or coarse initial structure
  better predicts future dynamical evolution.

Output
  outputs/data/results.jsonl
  outputs/data/results.csv
  outputs/data/results_summary.json

Usage
  python nbody_3d.py --n 256 --steps 400 --replicates 8 --workers 6
  python nbody_3d.py --n 512 --use-numba --models direct_isolated pm_periodic
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

# ── Optional Numba JIT (significant speedup for N > 256) ──────────────────────
try:
    from numba import njit
    _HAS_NUMBA = True

    # NOTE: parallel=True + ProcessPoolExecutor(fork) = OpenMP crash.
    # We get process-level parallelism from ProcessPoolExecutor; thread-level
    # parallelism inside each worker is unnecessary and causes fork() to abort.
    # cache=True means compilation happens once per machine and is reused.
    @njit(cache=True, fastmath=True)
    def _numba_direct_acc(pos, mass, G, eps2, periodic, box_size):
        """O(N^2) direct gravitational acceleration, Numba-JIT compiled."""
        n = pos.shape[0]
        acc = np.zeros((n, 3))
        for i in range(n):
            ax = ay = az = 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dz = pos[i, 2] - pos[j, 2]
                if periodic:
                    dx -= box_size * round(dx / box_size)
                    dy -= box_size * round(dy / box_size)
                    dz -= box_size * round(dz / box_size)
                r2 = dx * dx + dy * dy + dz * dz + eps2
                fac = G * mass / (r2 * math.sqrt(r2))
                ax -= fac * dx
                ay -= fac * dy
                az -= fac * dz
            acc[i, 0] = ax
            acc[i, 1] = ay
            acc[i, 2] = az
        return acc

except ImportError:
    _HAS_NUMBA = False

    def _numba_direct_acc(pos, mass, G, eps2, periodic, box_size):  # numpy stub
        """Numpy fallback — only called when numba is unavailable."""
        n = pos.shape[0]
        acc = np.zeros_like(pos)
        for i in range(n):
            dx = pos[i] - pos
            if periodic:
                dx = dx - box_size * np.round(dx / box_size)
            r2 = np.sum(dx * dx, axis=1) + eps2
            r2[i] = np.inf
            fac = G * mass / (r2 * np.sqrt(r2))
            acc[i] = -np.sum(fac[:, None] * dx, axis=0)
        return acc

Array = np.ndarray


# ── Configuration ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SimConfig:
    model:       str    # "direct_isolated" | "direct_periodic" | "pm_periodic"
    integrator:  str    # "leapfrog_kdk" | "rk4"
    init:        str    # "plummer3d" | "uniform3d" | "cold_clumpy3d"
    seed:        int
    n:           int   = 256      # particle count
    steps:       int   = 400      # total integration steps
    dt:          float = 0.005    # time step
    G:           float = 1.0      # gravitational constant
    eps:         float = 0.05     # softening length (3-D)
    box_size:    float = 2.0      # periodic box side; also IC placement scale
    pm_grid:     int   = 32       # PM cells per dimension (32^3 = 32 768)
    coarse_grid: int   = 8        # coarse density grid per dim (8^3 = 512)
    top_k:       int   = 16       # k for kNN density estimator
    h_early:     int   = 80       # step index for early-stage snapshot
    h_mid:       int   = 200      # step index for mid-stage snapshot
    plummer_a:   float = 0.20     # Plummer scale radius
    vel_scale:   float = 0.35     # velocity dispersion
    cold_scale:  float = 0.05     # within-clump dispersion
    clump_count: int   = 5        # number of clumps in cold_clumpy3d


@dataclass
class RunResult:
    # identity
    model: str; integrator: str; init: str; seed: int
    n: int; steps: int; dt: float; eps: float
    # conservation diagnostics
    energy_rel_drift:    Optional[float]   # None for PM (no exact potential)
    momentum_drift_rel:  float
    ang_mom_drift_rel:   float
    # coarse observable at four stages (0=initial, e=early, m=mid, f=final)
    coarse_var_0: float; coarse_var_e: float
    coarse_var_m: float; coarse_var_f: float
    # fine observable at initial and final
    fine_knn_0: float; fine_knn_f: float
    # kill-test fine observables (initial only — used for ODD correlation)
    fine_knn_all_0:     float   # kNN over ALL particles, not top-k
    fine_pk_small_0:    float   # small-scale Fourier power
    fine_close_pairs_0: float   # fraction of pairs within 4*eps
    # half-mass radius at four stages
    hmr_0: float; hmr_e: float; hmr_m: float; hmr_f: float
    # deltas (ODD primary test quantities)
    d_coarse_early: float; d_coarse_late: float
    d_hmr_early:    float; d_hmr_late:    float
    # virialization
    virial_0: float; virial_f: float
    # status
    status: str; message: str


# ── Utilities ──────────────────────────────────────────────────────────────────

def ensure_dirs(base: str) -> None:
    os.makedirs(os.path.join(base, "data"), exist_ok=True)


def min_image(dx: Array, L: float) -> Array:
    """Minimum-image convention. Works for any array shape."""
    return dx - L * np.round(dx / L)


def apply_pbc(pos: Array, cfg: SimConfig) -> Array:
    if cfg.model in ("direct_periodic", "pm_periodic"):
        return np.mod(pos, cfg.box_size)
    return pos


# ── Initial conditions ─────────────────────────────────────────────────────────

def _sphere_directions(rng: np.random.Generator, n: int) -> Array:
    """n unit vectors uniformly on S^2."""
    cth = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    sth = np.sqrt(1.0 - cth * cth)
    return np.column_stack([sth * np.cos(phi), sth * np.sin(phi), cth])


def sample_plummer3d(rng: np.random.Generator, cfg: SimConfig) -> Tuple[Array, Array]:
    """3-D Plummer sphere.
    Radial CDF: M(<r)/M = r^3 / (r^2 + a^2)^{3/2}  →  r = a / sqrt(u^{-2/3} - 1)
    Velocities: isotropic Gaussian scaled to approximate virial equilibrium.
    """
    a = cfg.plummer_a
    u = rng.uniform(1e-6, 1.0 - 1e-6, cfg.n)
    r = a / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    center = np.full(3, cfg.box_size / 2.0)
    pos = center + r[:, None] * _sphere_directions(rng, cfg.n)
    # σ ~ sqrt(G*M / (6*a)), approximate virial equipartition per component
    sigma = math.sqrt(cfg.G / (6.0 * a))
    vel = rng.normal(0.0, sigma, (cfg.n, 3))
    vel -= np.mean(vel, axis=0)
    return pos, vel


def sample_uniform3d(rng: np.random.Generator, cfg: SimConfig) -> Tuple[Array, Array]:
    pos = rng.uniform(0.0, cfg.box_size, (cfg.n, 3))
    vel = rng.normal(0.0, cfg.vel_scale, (cfg.n, 3))
    vel -= np.mean(vel, axis=0)
    return pos, vel


def sample_cold_clumpy3d(rng: np.random.Generator, cfg: SimConfig) -> Tuple[Array, Array]:
    lo, hi = 0.15 * cfg.box_size, 0.85 * cfg.box_size
    centers = rng.uniform(lo, hi, (cfg.clump_count, 3))
    labels  = rng.integers(0, cfg.clump_count, cfg.n)
    raw_pos = centers[labels] + rng.normal(0.0, cfg.cold_scale, (cfg.n, 3))
    # Only wrap for periodic models; isolated particles should be able to escape
    if cfg.model in ("direct_periodic", "pm_periodic"):
        pos = np.mod(raw_pos, cfg.box_size)
    else:
        pos = raw_pos
    vel = rng.normal(0.0, cfg.vel_scale * 0.15, (cfg.n, 3))
    vel -= np.mean(vel, axis=0)
    return pos, vel


def initial_conditions(cfg: SimConfig) -> Tuple[Array, Array]:
    rng = np.random.default_rng(cfg.seed)
    if cfg.init == "plummer3d":
        return sample_plummer3d(rng, cfg)
    if cfg.init == "uniform3d":
        return sample_uniform3d(rng, cfg)
    if cfg.init == "cold_clumpy3d":
        return sample_cold_clumpy3d(rng, cfg)
    raise ValueError(f"Unknown init: {cfg.init}")


# ── Force models ───────────────────────────────────────────────────────────────

def _numpy_direct_acc(pos: Array, mass: float, cfg: SimConfig, periodic: bool) -> Array:
    """Fully vectorized O(N^2) direct gravitational acceleration.

    Correct self-exclusion: fill_diagonal sets r2[i,i] = inf → inv_r3[i,i] = 0.
    Sign: a_i = -G*m * Σ_j (r_i - r_j) / |r_i - r_j|_soft^3   (attractive).
    """
    dx = pos[:, None, :] - pos[None, :, :]      # (N, N, 3)
    if periodic:
        dx = min_image(dx, cfg.box_size)
    r2 = np.sum(dx * dx, axis=-1) + cfg.eps ** 2  # (N, N)
    np.fill_diagonal(r2, np.inf)                   # exclude self; inf → 0 after **-1.5
    inv_r3 = r2 ** (-1.5)
    return -cfg.G * mass * np.einsum("ij,ijk->ik", inv_r3, dx)


def direct_acc(pos: Array, mass: float, cfg: SimConfig,
               periodic: bool, use_numba: bool) -> Array:
    if use_numba and _HAS_NUMBA:
        return _numba_direct_acc(pos, mass, cfg.G, cfg.eps ** 2, periodic, cfg.box_size)
    return _numpy_direct_acc(pos, mass, cfg, periodic)


# ── 3-D particle-mesh ─────────────────────────────────────────────────────────

def _cic_deposit3(pos: Array, mass_pp: float, L: float, g: int,
                  periodic: bool = True) -> Array:
    """Cloud-in-Cell mass assignment to a g^3 grid.

    periodic=True  : wrap particles with mod (standard PM periodic).
    periodic=False : deposit only particles inside [0,L)^3; discard tail
                     particles rather than wrapping or clipping them.
    """
    rho = np.zeros((g, g, g))
    if periodic:
        p = np.mod(pos, L) / L * g
    else:
        # Only deposit particles inside [0, L); discard rather than wrap
        inside = np.all((pos >= 0.0) & (pos < L), axis=1)
        p = pos[inside] / L * g
    i0 = np.floor(p).astype(int)
    f  = p - i0
    for bx in range(2):
        wx = (1.0 - f[:, 0]) if bx == 0 else f[:, 0]
        for by in range(2):
            wy = (1.0 - f[:, 1]) if by == 0 else f[:, 1]
            for bz in range(2):
                wz = (1.0 - f[:, 2]) if bz == 0 else f[:, 2]
                w  = wx * wy * wz
                if periodic:
                    ix = (i0[:, 0] + bx) % g
                    iy = (i0[:, 1] + by) % g
                    iz = (i0[:, 2] + bz) % g
                else:
                    ix = np.clip(i0[:, 0] + bx, 0, g - 1)
                    iy = np.clip(i0[:, 1] + by, 0, g - 1)
                    iz = np.clip(i0[:, 2] + bz, 0, g - 1)
                np.add.at(rho, (ix, iy, iz), mass_pp * w)
    return rho


def _cic_interp3(field: Array, pos: Array, L: float, g: int) -> Array:
    """CIC back-interpolation from g^3 grid to particle positions."""
    p   = np.mod(pos, L) / L * g
    i0  = np.floor(p).astype(int)
    f   = p - i0
    result = np.zeros(len(pos))
    for bx in range(2):
        wx = (1.0 - f[:, 0]) if bx == 0 else f[:, 0]
        for by in range(2):
            wy = (1.0 - f[:, 1]) if by == 0 else f[:, 1]
            for bz in range(2):
                wz = (1.0 - f[:, 2]) if bz == 0 else f[:, 2]
                w  = wx * wy * wz
                ix = (i0[:, 0] + bx) % g
                iy = (i0[:, 1] + by) % g
                iz = (i0[:, 2] + bz) % g
                result += w * field[ix, iy, iz]
    return result


def pm_acc_3d(pos: Array, mass_pp: float, cfg: SimConfig) -> Array:
    """3-D particle-mesh Poisson solver via FFT.

    Physics: ∇²φ = 4πGρ  →  φ_k = -4πG ρ_k / k²
    Force:   a = -∇φ     →  a_k = -ik φ_k = 4πGik ρ_k / k²

    rho must be mass *density* (mass per volume), not mass per cell.
    Dividing by cell_vol = (L/g)³ makes force amplitude grid-independent.
    """
    g, L, G = cfg.pm_grid, cfg.box_size, cfg.G
    cell_vol = (L / g) ** 3
    rho   = _cic_deposit3(pos, mass_pp, L, g, periodic=True) / cell_vol  # PM always periodic
    delta = rho - np.mean(rho)
    dk    = np.fft.fftn(delta)

    kf = 2.0 * math.pi / L                              # fundamental wavenumber
    freqs = np.fft.fftfreq(g) * g                       # integer frequencies
    kx = kf * freqs; ky = kf * freqs; kz = kf * freqs
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    k2[0, 0, 0] = np.inf                                # avoid DC division

    phi_k           = -4.0 * math.pi * G * dk / k2
    phi_k[0, 0, 0] = 0.0                                # zero mean potential

    ax_g = np.real(np.fft.ifftn(-1j * KX * phi_k))
    ay_g = np.real(np.fft.ifftn(-1j * KY * phi_k))
    az_g = np.real(np.fft.ifftn(-1j * KZ * phi_k))

    ax = _cic_interp3(ax_g, pos, L, g)
    ay = _cic_interp3(ay_g, pos, L, g)
    az = _cic_interp3(az_g, pos, L, g)
    return np.column_stack([ax, ay, az])


def acceleration(pos: Array, mass: float, cfg: SimConfig, use_numba: bool) -> Array:
    periodic = cfg.model in ("direct_periodic", "pm_periodic")
    if cfg.model == "pm_periodic":
        return pm_acc_3d(pos, mass, cfg)
    return direct_acc(pos, mass, cfg, periodic=periodic, use_numba=use_numba)


# ── Physical diagnostics ───────────────────────────────────────────────────────

def kinetic_energy(vel: Array, mass: float) -> float:
    return 0.5 * mass * float(np.sum(vel * vel))


def potential_energy_direct(pos: Array, mass: float, cfg: SimConfig) -> float:
    """O(N^2) direct pairwise potential. Returns nan for PM model."""
    if cfg.model == "pm_periodic":
        return float("nan")
    periodic = cfg.model == "direct_periodic"
    dx = pos[:, None, :] - pos[None, :, :]
    if periodic:
        dx = min_image(dx, cfg.box_size)
    r2  = np.sum(dx * dx, axis=-1) + cfg.eps ** 2
    iu  = np.triu_indices(len(pos), k=1)
    return float(-cfg.G * mass * mass * np.sum(1.0 / np.sqrt(r2[iu])))


def total_momentum(vel: Array, mass: float) -> Array:
    return mass * np.sum(vel, axis=0)


def angular_momentum_3d(pos: Array, vel: Array, mass: float) -> Array:
    """L = m * Σ_i (r_i - r_com) × (v_i - v_com).

    COM-relative to avoid origin-dependence: if total momentum is not
    exactly zero numerically, box-origin L picks up a translational term.
    """
    com = np.mean(pos, axis=0)
    vcm = np.mean(vel, axis=0)
    return mass * np.sum(np.cross(pos - com, vel - vcm), axis=0)


def conservation_drift(v0: Array, vf: Array, scale: float) -> float:
    """Absolute or relative drift, robustly normalised.

    When the initial conserved quantity is constructed to be near zero
    (e.g. total momentum after mean-velocity subtraction, ~1e-14), a pure
    relative measure blows up.  Use absolute drift normalised by a physical
    scale (sum of individual magnitudes) instead.  Fall back to relative only
    when the initial norm is safely large.
    """
    n0 = float(np.linalg.norm(v0))
    if n0 > 1e-8 * scale:
        return float(np.linalg.norm(vf - v0) / n0)
    return float(np.linalg.norm(vf - v0) / max(scale, 1e-30))


def virial_ratio(ke: float, pe: float) -> float:
    if not math.isfinite(pe) or abs(pe) < 1e-30:
        return float("nan")
    return 2.0 * ke / abs(pe)


def half_mass_radius(pos: Array, periodic: bool) -> float:
    """R_50: median distance from COM containing 50% of particles.

    Not meaningful on a torus (wrapped coordinates make COM and distances
    frame-dependent).  Returns nan for periodic models.
    """
    if periodic:
        return float("nan")
    com = np.mean(pos, axis=0)
    r   = np.linalg.norm(pos - com, axis=1)
    return float(np.sort(r)[len(r) // 2])


def coarse_density_var(pos: Array, cfg: SimConfig) -> float:
    """Variance of particle counts on a fixed g^3 grid.

    Periodic: fold into the periodic box.
    Isolated: count only particles inside [0, box_size)^3 — discard escapers.

    Both coarse and fine (fine_pk_small) use the same discard-outside rule
    for isolated models, so the two observables live on the same boundary.
    Using clip instead would create wall-pileup artifacts and an inconsistency
    with the Fourier observable.
    """
    g = cfg.coarse_grid
    L = cfg.box_size
    if cfg.model in ("direct_periodic", "pm_periodic"):
        p = np.mod(pos, L) / L
    else:
        inside = np.all((pos >= 0.0) & (pos < L), axis=1)
        if not np.any(inside):
            return 0.0
        p = pos[inside] / L
    idx = np.floor(p * g).astype(int)
    idx = np.clip(idx, 0, g - 1)
    hist = np.zeros((g, g, g))
    np.add.at(hist, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    return float(np.var(hist))


def fine_knn_density(pos: Array, k: int, eps: float, cfg: SimConfig) -> float:
    """Mean k-NN local density (3-D sphere) of the top-k densest particles."""
    if len(pos) < 2:
        return float("nan")
    n     = len(pos)
    k_eff = min(k, n - 1)
    dx    = pos[:, None, :] - pos[None, :, :]   # (N, N, 3)
    if cfg.model in ("direct_periodic", "pm_periodic"):
        dx = min_image(dx, cfg.box_size)
    r2 = np.sum(dx * dx, axis=-1)
    np.fill_diagonal(r2, np.inf)
    kth_r2  = np.partition(r2, kth=k_eff - 1, axis=1)[:, k_eff - 1]
    rk      = np.sqrt(np.maximum(kth_r2, eps ** 2))
    density = k_eff / (4.0 / 3.0 * math.pi * rk ** 3)
    top_idx = np.argsort(density)[-k_eff:]
    return float(np.mean(density[top_idx]))


def fine_knn_all(pos: Array, k: int, eps: float, cfg: SimConfig) -> float:
    """Mean k-NN local density averaged over ALL particles (not just top-k)."""
    if len(pos) < 2:
        return float("nan")
    n     = len(pos)
    k_eff = min(k, n - 1)
    dx    = pos[:, None, :] - pos[None, :, :]
    if cfg.model in ("direct_periodic", "pm_periodic"):
        dx = min_image(dx, cfg.box_size)
    r2 = np.sum(dx * dx, axis=-1)
    np.fill_diagonal(r2, np.inf)
    kth_r2  = np.partition(r2, kth=k_eff - 1, axis=1)[:, k_eff - 1]
    rk      = np.sqrt(np.maximum(kth_r2, eps ** 2))
    density = k_eff / (4.0 / 3.0 * math.pi * rk ** 3)
    return float(np.mean(density))   # mean over ALL particles


def fine_pk_small(pos: Array, cfg: SimConfig) -> float:
    """Power in small-scale density fluctuations (high-k Fourier modes).

    Kill-test observable: fine structure in Fourier space.  Isolated models
    use discard-not-wrap CIC so tail particles outside the box are excluded
    rather than aliased back in to spuriously inflate small-scale power.
    """
    g        = cfg.pm_grid
    L        = cfg.box_size
    mass_pp  = 1.0 / cfg.n
    cell_vol = (L / g) ** 3
    periodic = cfg.model in ("direct_periodic", "pm_periodic")
    rho   = _cic_deposit3(pos, mass_pp, L, g, periodic=periodic) / cell_vol
    delta = rho - np.mean(rho)
    dk    = np.fft.rfftn(delta)
    pk    = np.abs(dk) ** 2
    freqs  = np.fft.fftfreq(g) * g
    rfreqs = np.fft.rfftfreq(g) * g
    FX, FY, FZ = np.meshgrid(freqs, freqs, rfreqs, indexing="ij")
    k_abs = np.sqrt(FX**2 + FY**2 + FZ**2)
    return float(np.sum(pk[k_abs > g / 4.0]))


def fine_close_pairs(pos: Array, cfg: SimConfig) -> float:
    """Fraction of particle pairs separated by less than 4×softening.

    Kill-test observable 3: the most direct measure of sub-softening
    clumping.  If this is also uncorrelated with future clustering, fine
    structure at the particle scale carries no predictive power at all.
    Uses minimum-image for periodic models.
    """
    n        = len(pos)
    thresh2  = (4.0 * cfg.eps) ** 2
    dx       = pos[:, None, :] - pos[None, :, :]
    if cfg.model in ("direct_periodic", "pm_periodic"):
        dx = min_image(dx, cfg.box_size)
    r2 = np.sum(dx * dx, axis=-1)
    np.fill_diagonal(r2, np.inf)
    n_pairs  = n * (n - 1) / 2.0
    n_close  = float(np.sum(r2 < thresh2)) / 2.0   # upper triangle only
    return n_close / max(n_pairs, 1.0)


# ── Integrators ────────────────────────────────────────────────────────────────

SnapDict = Dict[int, Tuple[Array, Array]]


def integrate_leapfrog(pos0: Array, vel0: Array, mass: float,
                        cfg: SimConfig, snap_steps: List[int],
                        use_numba: bool) -> SnapDict:
    """KDK leapfrog. Symplectic; conserves a modified energy exactly."""
    snaps: SnapDict = {}
    pos, vel = pos0.copy(), vel0.copy()
    acc = acceleration(pos, mass, cfg, use_numba)
    dt  = cfg.dt
    if 0 in snap_steps:
        snaps[0] = (pos.copy(), vel.copy())
    for step in range(1, cfg.steps + 1):
        vel  = vel + 0.5 * dt * acc
        pos  = apply_pbc(pos + dt * vel, cfg)
        acc  = acceleration(pos, mass, cfg, use_numba)
        vel  = vel + 0.5 * dt * acc
        if step in snap_steps:
            snaps[step] = (pos.copy(), vel.copy())
    return snaps


def integrate_rk4(pos0: Array, vel0: Array, mass: float,
                   cfg: SimConfig, snap_steps: List[int],
                   use_numba: bool) -> SnapDict:
    """4th-order Runge-Kutta.  PBC applied ONLY after each complete step,
    not inside k-stage computations (fixes the original code's bug)."""
    snaps: SnapDict = {}
    pos, vel = pos0.copy(), vel0.copy()
    dt = cfg.dt
    if 0 in snap_steps:
        snaps[0] = (pos.copy(), vel.copy())

    def acc_at(p: Array) -> Array:
        # Force routines handle PBC internally; no extra wrapping here.
        return acceleration(apply_pbc(p, cfg), mass, cfg, use_numba)

    for step in range(1, cfg.steps + 1):
        # Stage 1
        k1v = acc_at(pos)
        k1x = vel
        # Stage 2
        k2v = acc_at(pos + 0.5 * dt * k1x)
        k2x = vel + 0.5 * dt * k1v
        # Stage 3
        k3v = acc_at(pos + 0.5 * dt * k2x)
        k3x = vel + 0.5 * dt * k2v
        # Stage 4
        k4v = acc_at(pos + dt * k3x)
        k4x = vel + dt * k3v
        # Full step — wrap only here
        pos = apply_pbc(
            pos + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x), cfg
        )
        vel = vel + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        if step in snap_steps:
            snaps[step] = (pos.copy(), vel.copy())
    return snaps


def integrate(pos0: Array, vel0: Array, mass: float,
              cfg: SimConfig, snap_steps: List[int],
              use_numba: bool) -> SnapDict:
    if cfg.integrator == "leapfrog_kdk":
        return integrate_leapfrog(pos0, vel0, mass, cfg, snap_steps, use_numba)
    if cfg.integrator == "rk4":
        return integrate_rk4(pos0, vel0, mass, cfg, snap_steps, use_numba)
    raise ValueError(f"Unknown integrator: {cfg.integrator}")


# ── Single-run battery item ────────────────────────────────────────────────────

def _stage_obs(pos: Array, vel: Array, mass: float, cfg: SimConfig,
               periodic: bool) -> dict:
    ke = kinetic_energy(vel, mass)
    pe = potential_energy_direct(pos, mass, cfg)
    return {
        "coarse_var":   coarse_density_var(pos, cfg),
        "fine_knn":     fine_knn_density(pos, cfg.top_k, cfg.eps, cfg),
        "fine_knn_all": fine_knn_all(pos, cfg.top_k, cfg.eps, cfg),
        "fine_pk":      fine_pk_small(pos, cfg),
        "fine_close":   fine_close_pairs(pos, cfg),
        "hmr":          half_mass_radius(pos, periodic),
        "virial":       virial_ratio(ke, pe),
        "ke":           ke,
        "pe":           pe,
    }


def run_one(cfg: SimConfig, use_numba: bool = False) -> Dict:
    try:
        mass = 1.0 / cfg.n
        periodic = cfg.model in ("direct_periodic", "pm_periodic")
        pos0, vel0 = initial_conditions(cfg)

        # Physical scales for robust drift normalisation (Bug 1 fix)
        # Use sum of individual magnitudes — never near zero even when total is.
        p_scale = float(mass * np.sum(np.linalg.norm(vel0, axis=1)))
        L_scale = float(mass * np.sum(
            np.linalg.norm(np.cross(pos0, vel0), axis=1)))

        p0  = total_momentum(vel0, mass)
        L0  = angular_momentum_3d(pos0, vel0, mass)
        ke0 = kinetic_energy(vel0, mass)
        pe0 = potential_energy_direct(pos0, mass, cfg)
        e0  = ke0 + pe0   # nan for PM

        # Clamp horizons so they never exceed cfg.steps.
        # Without this, snaps.get(h, snaps[0]) silently returns t=0 state,
        # making all early/mid deltas zero — a silent wrong result.
        h_e = min(cfg.h_early, cfg.steps)
        h_m = min(cfg.h_mid,   cfg.steps)

        snap_steps = sorted({0, h_e, h_m, cfg.steps})
        snaps = integrate(pos0, vel0, mass, cfg, snap_steps, use_numba)

        pos_0, vel_0 = snaps[0]
        pos_e, vel_e = snaps[h_e]
        pos_m, vel_m = snaps[h_m]
        pos_f, vel_f = snaps[cfg.steps]

        pf = total_momentum(vel_f, mass)
        Lf = angular_momentum_3d(pos_f, vel_f, mass)
        kef = kinetic_energy(vel_f, mass)
        pef = potential_energy_direct(pos_f, mass, cfg)
        ef  = kef + pef

        # Bug 1 fix: use physical-scale normalised drift
        mom_drift = conservation_drift(p0, pf, p_scale)

        # Bug 2 fix: angular momentum not conserved / not meaningful on torus
        if periodic:
            amom_drift = float("nan")
        else:
            amom_drift = conservation_drift(L0, Lf, L_scale)

        # Stage observables — must be computed before virial block uses o0/of
        o0 = _stage_obs(pos_0, vel_0, mass, cfg, periodic)
        oe = _stage_obs(pos_e, vel_e, mass, cfg, periodic)
        om = _stage_obs(pos_m, vel_m, mass, cfg, periodic)
        of = _stage_obs(pos_f, vel_f, mass, cfg, periodic)

        # Energy drift:
        #   direct_isolated — exact softened pair potential, true conservation test
        #   direct_periodic — minimum-image pair sum ≠ Ewald; proxy diagnostic only
        #   pm_periodic     — no direct PE available
        e_drift: Optional[float] = None
        if cfg.model in ("direct_isolated", "direct_periodic"):
            e_drift = float(abs(ef - e0) / max(abs(e0), 1e-30))

        # Virial: pm_periodic PE unavailable → nan; direct_periodic is proxy only
        if cfg.model == "pm_periodic":
            virial_0_val = float("nan")
            virial_f_val = float("nan")
        else:
            virial_0_val = virial_ratio(o0["ke"], o0["pe"])
            virial_f_val = virial_ratio(of["ke"], of["pe"])

        return asdict(RunResult(
            model=cfg.model, integrator=cfg.integrator,
            init=cfg.init, seed=cfg.seed, n=cfg.n,
            steps=cfg.steps, dt=cfg.dt, eps=cfg.eps,
            energy_rel_drift=e_drift,
            momentum_drift_rel=mom_drift,
            ang_mom_drift_rel=amom_drift,
            coarse_var_0=o0["coarse_var"], coarse_var_e=oe["coarse_var"],
            coarse_var_m=om["coarse_var"], coarse_var_f=of["coarse_var"],
            fine_knn_0=o0["fine_knn"],     fine_knn_f=of["fine_knn"],
            fine_knn_all_0=o0["fine_knn_all"],
            fine_pk_small_0=o0["fine_pk"],
            fine_close_pairs_0=o0["fine_close"],
            hmr_0=o0["hmr"], hmr_e=oe["hmr"],
            hmr_m=om["hmr"], hmr_f=of["hmr"],
            d_coarse_early=oe["coarse_var"] - o0["coarse_var"],
            d_coarse_late =of["coarse_var"] - o0["coarse_var"],
            d_hmr_early   =oe["hmr"] - o0["hmr"],
            d_hmr_late    =of["hmr"] - o0["hmr"],
            virial_0=virial_0_val, virial_f=virial_f_val,
            status="ok", message="",
        ))

    except Exception as exc:
        nan = float("nan")
        return asdict(RunResult(
            model=cfg.model, integrator=cfg.integrator,
            init=cfg.init, seed=cfg.seed, n=cfg.n,
            steps=cfg.steps, dt=cfg.dt, eps=cfg.eps,
            energy_rel_drift=None,
            momentum_drift_rel=nan, ang_mom_drift_rel=nan,
            coarse_var_0=nan, coarse_var_e=nan,
            coarse_var_m=nan, coarse_var_f=nan,
            fine_knn_0=nan,   fine_knn_f=nan,
            fine_knn_all_0=nan, fine_pk_small_0=nan, fine_close_pairs_0=nan,
            hmr_0=nan, hmr_e=nan, hmr_m=nan, hmr_f=nan,
            d_coarse_early=nan, d_coarse_late=nan,
            d_hmr_early=nan,    d_hmr_late=nan,
            virial_0=nan, virial_f=nan,
            status="error", message=str(exc),
        ))


# ── ODD correlation analysis ───────────────────────────────────────────────────

def _pearson(x: list, y: list) -> Optional[float]:
    a = np.array([v for v, w in zip(x, y)
                  if v is not None and w is not None
                  and np.isfinite(v) and np.isfinite(w)], float)
    b = np.array([w for v, w in zip(x, y)
                  if v is not None and w is not None
                  and np.isfinite(v) and np.isfinite(w)], float)
    if len(a) < 3:
        return None
    # Zero variance in either variable → r is undefined, not zero
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None   # degenerate: observable constant across replicates
    return float(np.corrcoef(a, b)[0, 1])


def odd_correlations(rows: List[Dict]) -> Dict:
    """For each model×integrator×init group, test whether fine or coarse
    initial structure better predicts future dynamical evolution.

    Primary ODD test:
      r(coarse_var_0,    d_coarse_early)  — original coarse predictor
    Kill-test fine observables (all must fail to confirm coarse dominance):
      r(fine_knn_0,      d_coarse_early)  — top-k kNN density (original)
      r(fine_knn_all_0,  d_coarse_early)  — full-distribution kNN density
      r(fine_pk_small_0, d_coarse_early)  — small-scale Fourier power
      r(fine_close_0,    d_coarse_early)  — fraction of close pairs
    """
    groups: Dict[str, list] = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        key = f"N={r['n']}|eps={r['eps']}|{r['model']}|{r['integrator']}|{r['init']}"
        groups.setdefault(key, []).append(r)

    out = {}
    for key, rs in groups.items():
        coarse0      = [r["coarse_var_0"]        for r in rs]
        fine_knn     = [r["fine_knn_0"]           for r in rs]
        fine_knn_all = [r["fine_knn_all_0"]       for r in rs]
        fine_pk      = [r["fine_pk_small_0"]      for r in rs]
        fine_close   = [r["fine_close_pairs_0"]   for r in rs]
        dc_early     = [r["d_coarse_early"]        for r in rs]
        dh_early     = [r["d_hmr_early"]           for r in rs]
        dc_late      = [r["d_coarse_late"]         for r in rs]
        dh_late      = [r["d_hmr_late"]            for r in rs]

        rc     = _pearson(coarse0,      dc_early)
        rf_top = _pearson(fine_knn,     dc_early)
        rf_all = _pearson(fine_knn_all, dc_early)
        rf_pk  = _pearson(fine_pk,      dc_early)
        rf_cl  = _pearson(fine_close,   dc_early)

        # best fine competitor: skip None (degenerate/constant observable) and nan
        fine_abs = [abs(r) for r in [rf_top, rf_all, rf_pk, rf_cl]
                    if r is not None and np.isfinite(r)]
        best_fine_r = max(fine_abs) if fine_abs else 0.0
        rc_abs      = abs(rc) if (rc is not None and np.isfinite(rc)) else 0.0
        fine_adv    = best_fine_r - rc_abs

        out[key] = {
            "n": len(rs),
            # coarse predictor
            "r_coarse0_vs_d_coarse_early":    rc,
            "r_coarse0_vs_d_hmr_early":       _pearson(coarse0, dh_early),
            "r_coarse0_vs_d_coarse_late":     _pearson(coarse0, dc_late),
            "r_coarse0_vs_d_hmr_late":        _pearson(coarse0, dh_late),
            # fine predictors — kill-test battery
            "r_fine_topk_vs_d_coarse_early":  rf_top,
            "r_fine_all_vs_d_coarse_early":   rf_all,
            "r_fine_pk_vs_d_coarse_early":    rf_pk,
            "r_fine_close_vs_d_coarse_early": rf_cl,
            # summary
            "best_fine_r_abs":      best_fine_r,
            "fine_advantage_early": fine_adv,   # positive = some fine obs beats coarse
        }
    return out


# ── Summary ────────────────────────────────────────────────────────────────────

def _smean(vals: list) -> Optional[float]:
    arr = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    return float(np.mean(arr)) if arr.size > 0 else None


def summarize(rows: List[Dict]) -> Dict:
    ok = [r for r in rows if r["status"] == "ok"]
    groups: Dict[str, list] = {}
    for r in ok:
        key = f"N={r['n']}|eps={r['eps']}|{r['model']}|{r['integrator']}|{r['init']}"
        groups.setdefault(key, []).append(r)

    group_sum = {}
    for key, rs in groups.items():
        group_sum[key] = {
            "count":                   len(rs),
            "mean_energy_rel_drift":   _smean([r["energy_rel_drift"]   for r in rs]),
            "mean_momentum_drift_rel": _smean([r["momentum_drift_rel"] for r in rs]),
            "mean_ang_mom_drift_rel":  _smean([r["ang_mom_drift_rel"]  for r in rs]),
            "mean_virial_final":       _smean([r["virial_f"]           for r in rs]),
            "mean_d_coarse_early":     _smean([r["d_coarse_early"]     for r in rs]),
            "mean_d_coarse_late":      _smean([r["d_coarse_late"]      for r in rs]),
            "mean_d_hmr_early":        _smean([r["d_hmr_early"]        for r in rs]),
            "mean_d_hmr_late":         _smean([r["d_hmr_late"]         for r in rs]),
        }

    return {
        "n_total":          len(rows),
        "n_ok":             len(ok),
        "n_error":          len(rows) - len(ok),
        "groups":           group_sum,
        "odd_correlations": odd_correlations(rows),
    }


# ── I/O ────────────────────────────────────────────────────────────────────────

def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def _worker_init(use_numba: bool) -> None:
    """Run inside each worker process before any jobs are dispatched.

    Triggers Numba JIT compilation once per worker (Numba caches the compiled
    kernel to disk so this is fast on subsequent runs).  Must happen inside the
    worker — running it in the main process and then fork()ing causes an OpenMP
    crash because Numba's runtime has already initialised its thread pool.
    """
    if use_numba and _HAS_NUMBA:
        pos = np.random.default_rng(0).uniform(0, 1, (32, 3)).astype(np.float64)
        _numba_direct_acc(pos, 1.0 / 32, 1.0, 0.05 ** 2, False, 1.0)
        _numba_direct_acc(pos, 1.0 / 32, 1.0, 0.05 ** 2, True,  1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-D N-body hard-test battery — ODD program (publishable run)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outdir",      default="outputs")
    parser.add_argument("--n",           type=int, nargs="+", default=[1024],
                        help="Particle count(s). Multiple values run a size sweep, "
                             "e.g. --n 512 1024 2048")
    parser.add_argument("--steps",       type=int,   default=600)
    parser.add_argument("--dt",          type=float, default=0.005)
    parser.add_argument("--eps",         type=float, default=0.05)
    parser.add_argument("--replicates",  type=int,   default=30,
                        help="Seeds per (N, model, integrator, IC) cell. "
                             "30 gives stable Pearson r estimates.")
    parser.add_argument("--workers",     type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--use-numba",   dest="use_numba",
                        action="store_true", default=True,
                        help="Use Numba JIT (strongly recommended for N>=512)")
    parser.add_argument("--no-numba",    dest="use_numba", action="store_false")
    parser.add_argument("--models",      nargs="+",
                        default=["direct_isolated", "pm_periodic"],
                        choices=["direct_isolated", "direct_periodic", "pm_periodic"],
                        help="direct_isolated = primary ODD hard test. "
                             "pm_periodic = fast large-N periodic model.")
    parser.add_argument("--integrators", nargs="+",
                        default=["leapfrog_kdk"],
                        choices=["leapfrog_kdk", "rk4"],
                        help="leapfrog_kdk is symplectic and preferred for long runs.")
    parser.add_argument("--inits",       nargs="+",
                        default=["plummer3d", "cold_clumpy3d"],
                        choices=["plummer3d", "uniform3d", "cold_clumpy3d"])
    args = parser.parse_args()

    if args.use_numba and not _HAS_NUMBA:
        print("WARNING: numba not installed. Install with:  pip install numba\n"
              "Falling back to numpy (will be much slower for N>=512).")
        args.use_numba = False

    ensure_dirs(args.outdir)
    data_dir = os.path.join(args.outdir, "data")

    if args.use_numba:
        print("Numba JIT enabled — workers will compile on first call "
              "(cached to disk, fast on repeat runs).\n")

    # ── Build config grid — sweeping over N if multiple values given ──────────
    seeds = [1000 + i for i in range(args.replicates)]
    configs: List[SimConfig] = []
    for n_val, model, integrator, init, seed in product(
            args.n, args.models, args.integrators, args.inits, seeds):
        configs.append(SimConfig(
            model=model, integrator=integrator, init=init,
            seed=seed, n=n_val, steps=args.steps,
            dt=args.dt, eps=args.eps,
        ))

    n_cells = len(args.n) * len(args.models) * len(args.integrators) * len(args.inits)
    total   = len(configs)
    print(
        f"Battery: {total} runs total\n"
        f"  N            : {args.n}\n"
        f"  models       : {args.models}\n"
        f"  integrators  : {args.integrators}\n"
        f"  inits        : {args.inits}\n"
        f"  replicates   : {args.replicates} per cell  ({n_cells} cells)\n"
        f"  steps / dt   : {args.steps} × {args.dt}\n"
        f"  workers      : {args.workers}\n"
        f"  numba        : {args.use_numba}\n"
    )

    # ── Parallel execution with tqdm progress bar ─────────────────────────────
    rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init,
                             initargs=(args.use_numba,)) as ex:
        futs = {ex.submit(run_one, cfg, args.use_numba): cfg for cfg in configs}
        with tqdm(total=total, unit="run", ncols=100,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                              "[{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for fut in as_completed(futs):
                res = fut.result()
                rows.append(res)
                tag  = "✓" if res["status"] == "ok" else "✗"
                e    = res["energy_rel_drift"]
                e_s  = f"ΔE={e:.1e}" if e is not None else "ΔE=n/a "
                vr   = res["virial_f"]
                vr_s = f"vir={vr:.2f}" if math.isfinite(vr) else "vir=nan"
                pbar.set_postfix_str(
                    f"{tag} N={res['n']:5d} {res['model'][:12]} "
                    f"{res['init'][:9]} s={res['seed']}  {e_s}  {vr_s}"
                )
                pbar.update(1)
                if res["status"] == "error":
                    tqdm.write(
                        f"  ✗ ERROR  N={res['n']}  {res['model']}  "
                        f"seed={res['seed']}: {res['message']}"
                    )

    rows.sort(key=lambda r: (r["n"], r["model"], r["integrator"], r["init"], r["seed"]))
    summary = summarize(rows)

    write_jsonl(os.path.join(data_dir, "results.jsonl"), rows)
    write_csv(  os.path.join(data_dir, "results.csv"),   rows)
    with open(os.path.join(data_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.  {summary['n_ok']}/{summary['n_total']} ok.")
    print(f"Results written to  {data_dir}/\n")

    # ── ODD verdict table ─────────────────────────────────────────────────────
    corr = summary["odd_correlations"]
    print("ODD kill-test battery — all fine observables vs d_coarse_early:")
    print(f"  {'Group':<48} {'n':>4} {'r_coarse':>9} "
          f"{'r_knn_top':>10} {'r_knn_all':>10} {'r_pk_sm':>8} {'r_close':>8}  verdict")
    print("  " + "─" * 112)
    for key in sorted(corr.keys()):
        c      = corr[key]
        rc     = c["r_coarse0_vs_d_coarse_early"]
        rf_top = c["r_fine_topk_vs_d_coarse_early"]
        rf_all = c["r_fine_all_vs_d_coarse_early"]
        rf_pk  = c["r_fine_pk_vs_d_coarse_early"]
        rf_cl  = c["r_fine_close_vs_d_coarse_early"]
        adv    = c["fine_advantage_early"]

        def _fs(v: Optional[float], width: int = 0) -> str:
            # None here means degenerate (zero-variance observable), not missing data
            if v is None:   return "degen".rjust(width) if width else "degen"
            return f"{v:+.3f}".rjust(width) if width else f"{v:+.3f}"

        adv_safe = adv if adv is not None else 0.0
        verdict = (
            "FINE  " if adv_safe >  0.05 else
            "COARSE" if adv_safe < -0.05 else
            "TIE   "
        )
        print(f"  {key:<48} {c['n']:>4} {_fs(rc):>9} "
              f"{_fs(rf_top):>10} {_fs(rf_all):>10} {_fs(rf_pk):>8} {_fs(rf_cl):>8}  [{verdict}]")

    print(
        "\nColumns: r_coarse = coarse density variance (the predictor that works)\n"
        "         r_knn_top = fine kNN top-k (original)  r_knn_all = kNN over all particles\n"
        "         r_pk_sm = small-scale Fourier power    r_close = close-pair fraction\n"
        "         'degen' = observable has zero variance for this IC (excluded from verdict)\n"
        "fine_advantage = max(|r_fine| over non-degenerate observables) − |r_coarse|\n"
        "  FINE   > +0.05: at least one fine observable beats coarse\n"
        "  COARSE < −0.05: coarse wins; ALL non-degenerate fine observables fail\n"
        "  TIE    |adv|<0.05: inconclusive\n"
    )


if __name__ == "__main__":
    main()

