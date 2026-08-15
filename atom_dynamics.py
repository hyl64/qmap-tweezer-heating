"""1-D moving harmonic oscillator: phonon heating from a tweezer trajectory.

Two estimators, both starting from the trap ground state at t=0.

1. Constant-ω analytic (exact for a driven HO)
       n̄ = (m / 2ħω) |∫ ẍ₀(t) e^{iωt} dt|²
   Valid when trap frequency is constant. Uses either the commanded
   continuous x₀(t) or a piecewise-linear interpolation of discrete frames.

2. Classical energy in a time-varying trap
       ẍ = −ω(t)² [x − x₀(t)],   ω(t) = ω₀ √(I(t)/I_ref)
   Final energy in the instantaneous oscillator:
       E = p²/2m + ½ m ω_f² (x − x₀,f)²
       n̄_cl = max(0, E / ħω_f − ½)

Physical defaults: ⁸⁷Rb, radial ω/2π = 80 kHz, 1 ms / SLM frame.
Pixel → metre conversion is explicit so the same code can run on
commanded pixel trajectories or measured focal-plane centroids.
"""
from __future__ import annotations

import numpy as np


# ── physical constants ──────────────────────────────────────────────
HBAR = 1.054_571_817e-34          # J s
M_RB87 = 1.443_160_60e-25         # kg
OMEGA0_DEFAULT = 2.0 * np.pi * 80e3   # rad/s
PX_M_DEFAULT = 0.5e-6             # 0.5 µm / pixel (demo calibration)


def a_ho(omega: float = OMEGA0_DEFAULT, mass: float = M_RB87) -> float:
    """Harmonic-oscillator length √(ħ / mω). ~38 nm at 80 kHz for ⁸⁷Rb."""
    return float(np.sqrt(HBAR / (mass * omega)))


def nbar_sudden(dx_m: float, omega: float = OMEGA0_DEFAULT,
                mass: float = M_RB87) -> float:
    """Phonon number after a sudden trap displacement dx (ground → coherent)."""
    return 0.5 * (dx_m / a_ho(omega, mass)) ** 2


# ── commanded continuous x0(t) ──────────────────────────────────────

def x0_of_t(start_m: np.ndarray, end_m: np.ndarray, t: np.ndarray,
            T: float, kind: str = "minjerk") -> np.ndarray:
    """x0(t) along a 1-D path. start/end (K,), t (Nt,), returns (Nt, K)."""
    u = np.clip(t / max(T, 1e-18), 0.0, 1.0)
    if kind == "minjerk":
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
    elif kind == "linear":
        s = u
    else:
        raise ValueError(kind)
    return start_m[None, :] + (end_m - start_m)[None, :] * s[:, None]


def x0ddot_of_t(start_m: np.ndarray, end_m: np.ndarray, t: np.ndarray,
                T: float, kind: str = "minjerk") -> np.ndarray:
    """ẍ0(t). start/end (K,), t (Nt,), returns (Nt, K)."""
    u = np.clip(t / max(T, 1e-18), 0.0, 1.0)
    D = (end_m - start_m)[None, :]          # (1, K)
    if kind == "minjerk":
        # d²s/du² = 60u − 180u² + 120u³ ; ẍ = (D/T²) s''(u)
        s2 = 60 * u - 180 * u**2 + 120 * u**3
        return D * (s2 / T**2)[:, None]
    if kind == "linear":
        return np.zeros((t.size, start_m.size))
    raise ValueError(kind)


# ── 1. analytic constant-ω heating ──────────────────────────────────

def nbar_analytic(start_m: np.ndarray, end_m: np.ndarray, T: float,
                  omega: float = OMEGA0_DEFAULT, kind: str = "minjerk",
                  n_quad: int = 4096, mass: float = M_RB87
                  ) -> tuple[np.ndarray, float]:
    """n̄_k from the Fourier component of ẍ0 at ω. Returns (n_k, mean)."""
    t = np.linspace(0.0, T, n_quad)
    acc = x0ddot_of_t(start_m, end_m, t, T, kind=kind)   # (Nt, K)
    kernel = np.exp(1j * omega * t)[:, None]
    integ = np.trapezoid(acc * kernel, t, axis=0)            # (K,)
    n_k = (mass / (2.0 * HBAR * omega)) * np.abs(integ) ** 2
    return n_k.astype(np.float64), float(n_k.mean())


def nbar_analytic_frames(x0_m: np.ndarray, dt: float,
                         omega: float = OMEGA0_DEFAULT,
                         mass: float = M_RB87
                         ) -> tuple[np.ndarray, float]:
    """Same formula on a discrete frame sequence x0_m (F, K).

    ẍ0 is the second difference, so a staircase of hologram centres
    is treated as the physical x0(t) the atom actually sees.
    """
    F, K = x0_m.shape
    if F < 3:
        return np.zeros(K), 0.0
    acc = np.zeros((F, K))
    acc[1:-1] = (x0_m[2:] - 2.0 * x0_m[1:-1] + x0_m[:-2]) / dt**2
    t = np.arange(F) * dt
    integ = np.trapezoid(acc * np.exp(1j * omega * t)[:, None], t, axis=0)
    n_k = (mass / (2.0 * HBAR * omega)) * np.abs(integ) ** 2
    return n_k.astype(np.float64), float(n_k.mean())


# ── 2. classical trajectory, time-varying ω ─────────────────────────

def nbar_classical(x0_m: np.ndarray, I_rel: np.ndarray, dt: float,
                   omega0: float = OMEGA0_DEFAULT, mass: float = M_RB87,
                   n_sub: int = 5000
                   ) -> tuple[np.ndarray, float]:
    """RK4 of ẍ = −ω(t)²(x−x0(t)), ω(t)=ω0√I_rel. x0_m/I_rel (F, K)."""
    F, K = x0_m.shape
    assert I_rel.shape == (F, K)
    h = dt / n_sub
    x = x0_m[0].copy()
    v = np.zeros(K)
    I_rel = np.clip(I_rel, 1e-6, None)

    def omega_at(f: int, frac: float) -> np.ndarray:
        f1 = min(f + 1, F - 1)
        I = (1.0 - frac) * I_rel[f] + frac * I_rel[f1]
        return omega0 * np.sqrt(I)

    def x0_at(f: int, frac: float) -> np.ndarray:
        f1 = min(f + 1, F - 1)
        return (1.0 - frac) * x0_m[f] + frac * x0_m[f1]

    for f in range(F - 1):
        for s in range(n_sub):
            frac = s / n_sub
            w = omega_at(f, frac)
            x0 = x0_at(f, frac)

            def acc(xx: np.ndarray, ww=w, xx0=x0) -> np.ndarray:
                return -(ww ** 2) * (xx - xx0)

            k1v = acc(x)
            k1x = v
            k2v = acc(x + 0.5 * h * k1x)
            k2x = v + 0.5 * h * k1v
            k3v = acc(x + 0.5 * h * k2x)
            k3x = v + 0.5 * h * k2v
            k4v = acc(x + h * k3x)
            k4x = v + h * k3v
            x = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

    w_f = omega0 * np.sqrt(I_rel[-1])
    E = 0.5 * mass * v**2 + 0.5 * mass * (w_f**2) * (x - x0_m[-1])**2
    n_k = np.maximum(0.0, E / (HBAR * w_f) - 0.5)
    return n_k.astype(np.float64), float(n_k.mean())


def summarize(n_k: np.ndarray) -> dict:
    return {
        "nbar_mean": float(n_k.mean()),
        "nbar_max": float(n_k.max()) if n_k.size else 0.0,
        "nbar_p95": float(np.percentile(n_k, 95)) if n_k.size else 0.0,
        "P0": float(np.exp(-n_k).mean()) if n_k.size else 1.0,  # coherent-state ground pop
    }


# 3. phase->motion coupling: nbar from frame-to-frame focal-phase jumps
# Coherent transport claim: if the SLM holograms' focal-plane phase at the
# moving trap jumps by dphi between frames, the atom gets a transverse momentum
# kick (phase gradient dphi/step). For a HO, momentum kick dp -> phonons
#   nbar_frame = (hbar*dphi/step)^2 / (2 m hbar omega)
def nbar_phase_kick(dphi, step_frames_m, dt, omega=OMEGA0_DEFAULT, mass=M_RB87,
                    lam_m=0.80e-6):
    """nbar from per-atom per-frame focal-phase jumps, with physical wavevector cap.

    Momentum kick per frame: dp = hbar * k_eff, where k_eff = the spatial phase
    gradient (dphi/step) capped at the free-space wavevector k0 = 2*pi/lam (a phase
    folded modulo 2pi cannot exceed one optical wavelength per radian of transport,
    so a phase gradient beyond k0 is unphysical for photon-coupled momentum).
    nbar_frame = dp^2 / (2 m hbar omega). Summed over frames.
    dphi (F-1,K) rad, step_frames_m (K,), returns (n_k, mean)."""
    dphi = np.abs(dphi)
    step_safe = np.maximum(np.asarray(step_frames_m, dtype=float), 1e-15)[None, :]
    k_eff = np.minimum(dphi / step_safe, 2.0*np.pi/max(lam_m,1e-9))
    per_frame = (HBAR * k_eff) ** 2 / (2.0 * mass * HBAR * omega)
    n_k = per_frame.sum(axis=0)
    return np.asarray(n_k, dtype=float), float(n_k.mean())
