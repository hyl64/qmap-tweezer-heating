"""Trap potentials and pluggable phase→motion coupling models.

The strict quantum mapping (路线三 A2) pipeline:

    SLM phase frames φ_SLM --(propagation)--> focal field U(x, t)
        --(extraction)--> x0(t) trap trajectory, ω(t) trap frequency,
                          φ(t) optical phase at the trap centre, I_rel(t)
        --(PhaseCoupling)--> effective trap parameters seen by the atom
        --(solver)--> exact quantum state → n̄(t), P0(t), P(n)

Physics boundary (honest, see COHERENT-TRANSPORT-REPORT §六):
for a *single-beam* intensity trap U ∝ −|E|² the carrier phase φ(t) does
NOT enter the potential.  Phase matters through concrete physical channels
only, each represented here by a CouplingModel:

  * NullCoupling            — phase-independent baseline (|E|² only).
  * InterferencePosition    — tweezer formed by interference of the SLM
                              beam with a reference (crossed beams / lattice
                              geometry): φ(t) transduces directly into trap
                              position δx = −φ/(2 k_eff).  The cleanest
                              phase→n̄ channel; k_eff is a geometry knob.
  * ParametricIntensity     — stray/ghost-beam interference turns phase
                              noise into intensity ripple → ω(t) modulation
                              → parametric (2ω-resonant) heating.
  * PhaseGradientKick       — transverse phase gradient at the trap gives a
                              momentum kick per frame (honest cap k0=2π/λ).

A collaborator's analytic derivation plugs in here: subclass PhaseCoupling
and implement trap_params(t, phi, I_rel) -> TrapParams.  Every model is
validated exactly in qmap/verify.py (see QMAP-REPORT.md §verification).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .constants import HBAR, M_RB87, a_ho, k_light


# ─────────────────────────────────────────────────────────────────────
# Trap-parameter dataclass (the atom's view of the world)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TrapParams:
    """Instantaneous effective trap parameters.

    x0       (Nt,)  trap centre position            [m]
    omega    (Nt,)  trap frequency                  [rad/s]
    t        (Nt,)  time axis [s]; None => np.arange(Nt) (index units)
    U0       (Nt,)  trap depth (positive)           [J]  (optional)
    sigma    (Nt,)  Gaussian waist                  [m]  (optional)
    dphi_dt  (Nt,)  unwrapped phase slope           [rad/s] (optional)

    The time axis matters: potentials interpolate x0/omega/U0/sigma onto
    the solver's clock with np.interp(t, tp.t, tp.x0).  If t is None the
    arrays are indexed 0..Nt-1 (use only for unit-style tests).
    """
    x0: np.ndarray
    omega: np.ndarray
    t: np.ndarray | None = None
    U0: np.ndarray | None = None
    sigma: np.ndarray | None = None
    dphi_dt: np.ndarray | None = None
    step: bool = False

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)
        self.omega = np.asarray(self.omega, dtype=float)
        n = len(self.x0)
        if self.t is None:
            self.t = np.arange(n, dtype=float)
        else:
            self.t = np.asarray(self.t, dtype=float)
            if len(self.t) != n:
                raise ValueError("TrapParams.t must match x0/omega length")
        if self.U0 is not None:
            self.U0 = np.asarray(self.U0, dtype=float)
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=float)
        if self.dphi_dt is not None:
            self.dphi_dt = np.asarray(self.dphi_dt, dtype=float)

    def x0_at(self, t):
        if self.step:
            idx = np.searchsorted(self.t, np.asarray(t), side="right") - 1
            return self.x0[np.clip(idx, 0, len(self.x0) - 1)]
        return np.interp(t, self.t, self.x0)

    def omega_at(self, t):
        if self.step:
            idx = np.searchsorted(self.t, np.asarray(t), side="right") - 1
            return self.omega[np.clip(idx, 0, len(self.omega) - 1)]
        return np.interp(t, self.t, self.omega)

    def U0_at(self, t):
        if self.U0 is None:
            raise ValueError("U0 not set")
        if self.step:
            idx = np.searchsorted(self.t, np.asarray(t), side="right") - 1
            return self.U0[np.clip(idx, 0, len(self.U0) - 1)]
        return np.interp(t, self.t, self.U0)

    def sigma_at(self, t):
        if self.sigma is None:
            raise ValueError("sigma not set")
        if self.step:
            idx = np.searchsorted(self.t, np.asarray(t), side="right") - 1
            return self.sigma[np.clip(idx, 0, len(self.sigma) - 1)]
        return np.interp(t, self.t, self.sigma)


class PhaseCoupling:
    """Base class: phase trajectory → effective trap parameters.

    THIS is the interface for the physics derivation
    (docs/COUPLING-INTERFACE.md).  Subclasses implement `trap_params`;
    the solver then does the exact quantum evolution and the verification
    suite checks the analytic predictions.
    """

    name = "base"

    def trap_params(self, t, phi, I_rel, x0_cmd, omega_ref, mass=M_RB87):
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    @staticmethod
    def _omega_from_I(I_rel, omega_ref):
        """ω(t) = ω_ref √I_rel  (trap depth ∝ intensity, red-detuned)."""
        return omega_ref * np.sqrt(np.clip(I_rel, 1e-9, None))

    @staticmethod
    def _depth_from_omega(omega, sigma, mass=M_RB87):
        """Gaussian-tweezer depth U0 = ½ m ω² σ² (curvature match)."""
        return 0.5 * mass * omega ** 2 * sigma ** 2


# ─────────────────────────────────────────────────────────────────────
# Model 0 — phase-independent baseline (|E|² only)
# ─────────────────────────────────────────────────────────────────────

class NullCoupling(PhaseCoupling):
    """The honest baseline: U ∝ −|E|², carrier phase does not enter.

    Trap centre = commanded trajectory; ω(t) = ω_ref √I_rel (intensity
    flicker → trap-depth modulation only).  This model contains NO
    phase→n̄ channel by construction.
    """

    name = "null"

    def trap_params(self, t, phi, I_rel, x0_cmd, omega_ref, mass=M_RB87):
        omega = self._omega_from_I(I_rel, omega_ref)
        return TrapParams(t=np.asarray(t, dtype=float), x0=x0_cmd, omega=omega,
                          dphi_dt=np.gradient(np.unwrap(phi), t))


# ─────────────────────────────────────────────────────────────────────
# Model A — interference tweezer: phase → position transduction
# ─────────────────────────────────────────────────────────────────────

class InterferencePosition(PhaseCoupling):
    """Phase transduces directly into trap position.

    Geometry: tweezer formed by interference of the SLM beam with a
    reference (crossed-beam / lattice geometry).  The interference term
    ∝ cos(2 k_eff x + φ(t)) moves the lattice/trap minimum:
        δx(t) = −φ(t)/(2 k_eff).

    k_eff = π/λ_p with λ_p the interference period (λ_p = λ/(2 sin(θ/2))
    for beams crossing at angle θ).  Counter-propagating (θ = π):
    λ_p = λ/2, k_eff = 2π/λ = k0.

    Phase jumps δφ between frames ⇒ sudden trap displacement
    δx = δφ/(2k_eff) ⇒ n̄ per jump = (δφ/2k_eff)²/(2a0²)
    — verified exactly in verify.py.
    """

    name = "interference-position"

    def __init__(self, k_eff=None, lam=0.80e-6):
        self.k_eff = k_eff if k_eff is not None else k_light(lam)

    def trap_params(self, t, phi, I_rel, x0_cmd, omega_ref, mass=M_RB87):
        phi_u = np.unwrap(phi)
        dx = -phi_u / (2.0 * self.k_eff)
        omega = self._omega_from_I(I_rel, omega_ref)
        return TrapParams(t=np.asarray(t, dtype=float), x0=x0_cmd + dx,
                          omega=omega, dphi_dt=np.gradient(phi_u, t))

    @staticmethod
    def nbar_per_jump(dphi, k_eff, omega, mass=M_RB87):
        """Exact n̄ from one sudden phase jump δφ (≡ sudden displacement)."""
        a0 = a_ho(omega, mass)
        dx = abs(dphi) / (2.0 * k_eff)
        return 0.5 * (dx / a0) ** 2


# ─────────────────────────────────────────────────────────────────────
# Model B — parametric intensity (stray interference → ω(t) ripple)
# ─────────────────────────────────────────────────────────────────────

class ParametricIntensity(PhaseCoupling):
    """Phase noise → intensity ripple → ω(t) modulation → parametric heating.

    A stray/ghost beam of relative amplitude ε interfering with the main
    beam produces I_rel(t) = 1 + 2ε cos(φ(t) − φ_ghost).  The trap frequency
    then oscillates: ω(t) = ω_ref √I_rel(t).  A φ component at 2ω is
    parametrically resonant; the exact quantum growth rate is verified
    against the analytic short-time rate κ = ε_ω ω_ref / 4 in verify.py.
    """

    name = "parametric-intensity"

    def __init__(self, eps=0.01, phi_ghost=0.0):
        self.eps = eps
        self.phi_ghost = phi_ghost

    def trap_params(self, t, phi, I_rel, x0_cmd, omega_ref, mass=M_RB87):
        phi_u = np.unwrap(phi)
        if np.allclose(I_rel, 1.0, atol=1e-9):
            I = 1.0 + 2.0 * self.eps * np.cos(phi_u - self.phi_ghost)
            I = np.clip(I, 1e-6, None)
        else:
            I = np.clip(I_rel, 1e-6, None)
        omega = self._omega_from_I(I, omega_ref)
        return TrapParams(t=np.asarray(t, dtype=float), x0=x0_cmd, omega=omega,
                          dphi_dt=np.gradient(phi_u, t))


# ─────────────────────────────────────────────────────────────────────
# Model C — phase-gradient momentum kicks (honest wavevector cap)
# ─────────────────────────────────────────────────────────────────────

class PhaseGradientKick(PhaseCoupling):
    """Per-frame focal-phase jumps → transverse momentum kicks.

    Honest version of atom_dynamics.nbar_phase_kick: the spatial phase
    gradient across the moving trap, |∇φ| ≈ |δφ|/step, transfers momentum
    dp = ħ·min(|δφ|/step, k0) (cap: a phase folded mod 2π cannot exceed
    one optical wavelength per radian of transport).  The quantum solver
    applies each kick exactly; for a single kick from the ground state
    n̄ = dp²/(2mħω), also exact.

    Note: for a pure intensity trap this channel is NOT physical — kept
    as the classical pipeline's legacy estimator upgraded to exact quantum
    dynamics; labelled 'illustrative' in the report.
    """

    name = "phase-gradient-kick"

    def __init__(self, lam=0.80e-6, step_m=0.5e-6):
        self.k0 = k_light(lam)
        self.step_m = step_m

    def kicks(self, phi, t, omega, mass=M_RB87):
        """Momentum kicks at frame boundaries from phase jumps.

        Returns (dp (F-1,), nbar_per_kick (F-1,)).  dp = ħ·min(|δφ|/step, k0).
        """
        phi_u = np.unwrap(phi)
        dphi = np.diff(phi_u)
        step = max(self.step_m, 1e-15)
        k_eff = np.minimum(np.abs(dphi) / step, self.k0)
        dp = np.sign(dphi) * HBAR * k_eff
        n = dp ** 2 / (2.0 * mass * HBAR * omega)
        return dp, n

    def trap_params(self, t, phi, I_rel, x0_cmd, omega_ref, mass=M_RB87):
        omega = self._omega_from_I(I_rel, omega_ref)
        return TrapParams(t=np.asarray(t, dtype=float), x0=x0_cmd, omega=omega,
                          dphi_dt=np.gradient(np.unwrap(phi), t))


# ─────────────────────────────────────────────────────────────────────
# Potentials (for the grid solver)
# ─────────────────────────────────────────────────────────────────────

class MovingPotential:
    """Base: V(x, t); implement __call__(x, t) -> (Nx,) or (Nx, Nt).

    `prepare(x)` precomputes the potential on the grid for every frame
    (frame-constant traps); __call__ then only indexes the cached array.
    This is ~20x faster than re-interpolating trap parameters per call
    (soft_solve evaluates V ~10 times per time step).
    """

    _cache = None

    def __call__(self, x, t):
        raise NotImplementedError

    def prepare(self, x):
        """Precompute V_f(x) for all frames; returns (Nf, Nx)."""
        t = np.asarray(self.tp.t, dtype=float)
        Vf = np.zeros((len(t), len(x)))
        for i, ti in enumerate(t):
            Vf[i] = np.asarray(self(x, ti)).ravel()
        self._cache = Vf
        return Vf

    def describe(self):
        return self.__class__.__name__


class MovingHarmonic(MovingPotential):
    """V(x,t) = ½ m ω(t)² (x − x0(t))²  — the strict-mapping workhorse."""

    def __init__(self, tp, mass=M_RB87):
        self.tp = tp
        self.mass = mass

    def __call__(self, x, t):
        if self._cache is not None:
            idx = np.searchsorted(self.tp.t, np.asarray(t), side="right") - 1
            idx = np.clip(idx, 0, len(self.tp.t) - 1)
            return self._cache[idx]
        t = np.atleast_1d(np.asarray(t, dtype=float))
        x0 = self.tp.x0_at(t)
        w = self.tp.omega_at(t)
        xx = x[:, None] - x0[None, :]
        return 0.5 * self.mass * w[None, :] ** 2 * xx ** 2


class GaussianTweezer(MovingPotential):
    """V(x,t) = −U0(t) exp(−(x − x0(t))²/(2σ(t)²)) — real (anharmonic) tweezer."""

    def __init__(self, tp, mass=M_RB87):
        self.tp = tp
        self.mass = mass
        if tp.U0 is None or tp.sigma is None:
            raise ValueError("GaussianTweezer needs U0 and sigma in TrapParams")

    def __call__(self, x, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        x0 = self.tp.x0_at(t)
        U = self.tp.U0_at(t)
        s = self.tp.sigma_at(t)
        xx = x[:, None] - x0[None, :]
        return -U[None, :] * np.exp(-xx ** 2 / (2.0 * s[None, :] ** 2))


def make_potential(tp, kind="harmonic", mass=M_RB87):
    if kind == "harmonic":
        return MovingHarmonic(tp, mass)
    if kind == "gaussian":
        return GaussianTweezer(tp, mass)
    raise ValueError(kind)

