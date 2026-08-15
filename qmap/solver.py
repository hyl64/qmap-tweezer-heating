"""Exact quantum engines for the strict quantum mapping.

Engines (all solve the same physics, cross-verified in verify.py):

  1. soft_solve        — split-operator Fourier transform on a real-space
                         grid; works for ANY potential (harmonic, Gaussian,
                         anharmonic), any ω(t), any x0(t).  'Gold standard'.
  2. fock_driven_ho    — driven harmonic oscillator in the comoving frame,
                         Fock basis, constant ω.  The strict mapping:
                             lab frame --(boost)--> comoving frame
                             H' = p²/2m + ½mω²y² − m·ẍ0(t)·y
                         exact solution from |0⟩ is a coherent state;
                         n̄ = (m/2ħω)|∫ ẍ0 e^{iωt} dt|².
  3. fock_variable_om  — instantaneous Fock basis for ω(t): includes the
                         non-adiabatic connection ⟨n|∂t|m⟩ = ∓(ω̇/4ω)√(…)
                         (parametric channel) plus the linear drive term.

Analytic benchmarks (independent of all engines):
    nbar_analytic_driven   constant-ω driven-HO Fourier formula
    nbar_staircase         discrete frame jumps (exact coherent sum)
    nbar_resonant_drive    resonant drive n̄ = F0²t²/(8mħω³)
    nbar_sudden            sudden displacement d²/(2a0²)
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.sparse import diags

from .constants import HBAR, M_RB87, a_ho


# ═════════════════════════════════════════════════════════════════════
# 1. Split-operator Fourier transform (grid, any potential)
# ═════════════════════════════════════════════════════════════════════

def soft_solve(V, x, psi0, t_eval, dt, mass=M_RB87, order=4):
    """Split-operator evolution on a periodic grid.

    V      : callable(x, t) -> (Nx,) potential energy [J]
    x      : (Nx,) uniform grid [m]
    psi0   : (Nx,) or (K, Nx) initial wavefunction(s)
    t_eval : (Nt,) snapshot times (multiples of dt from t_eval[0])
    dt     : time step [s]  (default order=4 Yoshida — phase-accurate over
             long runs; order=2 Strang is ~4x faster but its O(dt³) phase
             drift scrambles interference over >10^5 steps)
    Returns (psi (Nt,K,Nx), t_eval).
    """
    psi0 = np.atleast_2d(psi0).astype(complex)
    K, Nx = psi0.shape
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    if hasattr(V, "prepare") and V._cache is None:
        V.prepare(x)  # frame-constant potential: cache V_f on the grid
    t_eval = np.asarray(t_eval, dtype=float)
    t0, T = t_eval[0], t_eval[-1]
    n_steps = int(round((T - t0) / dt))
    snap_idx = np.round((t_eval - t0) / dt).astype(int)
    snap_idx = np.clip(snap_idx, 0, n_steps)
    out = np.zeros((len(t_eval), K, Nx), dtype=complex)
    psi = psi0.copy()
    out[0] = psi
    s = 1
    for i in range(1, n_steps + 1):
        t = t0 + (i - 1) * dt
        if order == 2:
            vh = np.asarray(V(x, t))
            if vh.ndim == 2:
                vh = vh[:, 0]
            psi = psi * np.exp(-1j * vh[None, :] * dt / (2.0 * HBAR))
            tk = np.exp(-1j * HBAR * k ** 2 * dt / (2.0 * mass))
            psi = np.fft.ifft(np.fft.fft(psi, axis=1) * tk[None, :], axis=1)
            vh = np.asarray(V(x, t + 0.5 * dt))
            if vh.ndim == 2:
                vh = vh[:, 0]
            psi = psi * np.exp(-1j * vh[None, :] * dt / (2.0 * HBAR))
        elif order == 4:
            g1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
            g2 = -2.0 ** (1.0 / 3.0) * g1
            g3 = 1.0 - 2.0 * (g1 + g2)
            tks = [np.exp(-1j * HBAR * k ** 2 * gg * dt / (2.0 * mass))
                   for gg in (g1, g2, g3, g2, g1)]
            for gg, tk in zip((g1, g2, g3, g2, g1), tks):
                dth = gg * dt / 2.0
                vh = np.asarray(V(x, t))
                if vh.ndim == 2:
                    vh = vh[:, 0]
                psi = psi * np.exp(-1j * vh[None, :] * dth / HBAR)
                psi = np.fft.ifft(np.fft.fft(psi, axis=1) * tk[None, :], axis=1)
                vh = np.asarray(V(x, t + 0.5 * gg * dt))
                if vh.ndim == 2:
                    vh = vh[:, 0]
                psi = psi * np.exp(-1j * vh[None, :] * dth / HBAR)
        else:
            raise ValueError(order)
        if s < len(snap_idx) and i == snap_idx[s]:
            out[s] = psi
            s += 1
    return out, t_eval


def harmonic_ground(x, x0, omega, mass=M_RB87):
    """Normalised HO ground state centred at x0 on the grid."""
    a0 = a_ho(omega, mass)
    psi = np.exp(-((x - x0) ** 2) / (2.0 * a0 ** 2))
    return psi / np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))


# ═════════════════════════════════════════════════════════════════════
# 2. Comoving-frame driven HO — Fock basis, constant ω (vectorised K)
# ═════════════════════════════════════════════════════════════════════

def _trajectory_derivs(x0_of_t, tq, K):
    """(x0, dx0/dt, d²x0/dt²) on tq from a shape-preserving spline."""
    X = np.atleast_2d(np.asarray(x0_of_t(tq), dtype=float))
    if X.shape[0] != len(tq):
        X = X.T
    x0 = X
    d1 = PchipInterpolator(tq, X, axis=0).derivative()(tq)
    d2 = PchipInterpolator(tq, d1, axis=0).derivative()(tq)
    return x0, d1, d2


def fock_driven_ho(x0_of_t, t_eval, omega, mass=M_RB87, Nmax=60,
                   rtol=1e-10, atol=1e-12, acc_of_t=None):
    """Driven HO in the comoving frame, Fock basis, constant ω.

    x0_of_t(t) -> (Nt,K).  If acc_of_t(t) -> (Nt,K) is given (analytic ẍ0),
    it is used for the drive (exact); otherwise ẍ0 comes from a Pchip spline
    of x0_of_t (interpolation error ~1e-2 for oscillatory trajectories —
    prefer acc_of_t for quantitative benchmarks).
    Returns dict: c (Nt,K,Nmax), nbar, P0, t.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    t0, T = t_eval[0], t_eval[-1]
    Nq = max(1024, 16 * len(t_eval))
    tq = np.linspace(t0, T, Nq)
    x0, d1, d2 = _trajectory_derivs(x0_of_t, tq, None)
    K = x0.shape[1]
    if acc_of_t is not None:
        A = np.atleast_2d(np.asarray(acc_of_t(tq), dtype=float))
        if A.shape[0] != len(tq):
            A = A.T
        F = -mass * A                              # (Nq,K) drive force
    else:
        F = -mass * d2                             # (Nq,K) drive force
    a0 = a_ho(omega, mass)
    n = np.arange(Nmax)
    sqp = np.sqrt(n + 1.0); sqp[-1] = 0.0
    sqm = np.sqrt(n)
    E = HBAR * omega * (n + 0.5)
    # drive matrix: (a0/√2)(√(n+1)|n⟩⟨n+1| + √n|n⟩⟨n−1|)
    off = [np.sqrt(np.arange(1, Nmax)), np.sqrt(np.arange(1, Nmax))]
    M_drive = diags(off, [1, -1], shape=(Nmax, Nmax), format="csr") * (a0 / np.sqrt(2.0))
    Fq = F  # (Nq,K)
    Fint = PchipInterpolator(tq, Fq, axis=0)

    def rhs(t, c):
        c = c.reshape(K, Nmax)
        fk = Fint(t)                                   # (K,)
        drive = fk[:, None] * (M_drive @ c.T).T        # (K, Nmax)
        dc = (-1j / HBAR) * (E[None, :] * c - drive)
        return dc.reshape(-1)

    c0 = np.zeros((K, Nmax), dtype=complex); c0[:, 0] = 1.0
    sol = solve_ivp(rhs, (t0, T), c0.reshape(-1), t_eval=t_eval,
                    method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"fock_driven_ho failed: {sol.message}")
    c = sol.y.T.reshape(len(t_eval), K, Nmax)
    nbar = np.einsum("tkn,n->tk", np.abs(c) ** 2, n)
    P0 = np.abs(c[:, :, 0]) ** 2
    return dict(c=c, nbar=nbar, P0=P0, t=t_eval, omega=omega)


def fock_variable_om(x0_of_t, omega_of_t, t_eval, mass=M_RB87, Nmax=60,
                     rtol=1e-10, atol=1e-12):
    """Instantaneous Fock basis for ω(t) (vectorised over K).

    Includes the non-adiabatic connection (parametric channel)
    ⟨n|∂t|n±2⟩ = ∓(ω̇/4ω)√(n(n±1)+…) and the linear drive −mẍ0·y.
    Returns dict: c, nbar, P0, t.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    t0, T = t_eval[0], t_eval[-1]
    Nq = max(1024, 16 * len(t_eval))
    tq = np.linspace(t0, T, Nq)
    x0, d1, d2 = _trajectory_derivs(x0_of_t, tq, None)
    K = x0.shape[1]
    F = -mass * d2
    W = np.atleast_2d(np.asarray(omega_of_t(tq), dtype=float))
    if W.shape[0] != len(tq):
        W = W.T
    if W.shape[1] == 1 and K > 1:
        W = np.repeat(W, K, axis=1)
    wd = PchipInterpolator(tq, W, axis=0).derivative()(tq)
    n = np.arange(Nmax)
    off1 = [np.sqrt(np.arange(1, Nmax)), np.sqrt(np.arange(1, Nmax))]
    M_drive = diags(off1, [1, -1], shape=(Nmax, Nmax), format="csr") / np.sqrt(2.0)
    # ⟨n|∂t|n+2⟩ = −(ω̇/4ω)√((n+1)(n+2)),  ⟨n|∂t|n−2⟩ = +(ω̇/4ω)√(n(n−1))
    # ⇒ in iħċ_n ⊃ −iħ⟨n|∂t|m⟩c_m:  +2 diag gets +,  −2 diag gets −
    off2 = [np.sqrt(np.arange(2, Nmax + 1) * np.arange(1, Nmax)),
            -np.sqrt(np.arange(2, Nmax + 1) * np.arange(1, Nmax))]
    M_conn = diags(off2, [2, -2], shape=(Nmax, Nmax), format="csr")
    Fint = PchipInterpolator(tq, F, axis=0)
    Wint = PchipInterpolator(tq, W, axis=0)
    WDint = PchipInterpolator(tq, wd, axis=0)

    def rhs(t, c):
        c = c.reshape(K, Nmax)
        wk = Wint(t)                                   # (K,)
        fk = Fint(t)
        wdk = WDint(t)
        a0t = np.sqrt(HBAR / (mass * np.maximum(wk, 1e-30)))
        E = HBAR * wk[:, None] * (n + 0.5)[None, :]
        drive = -fk[:, None] * a0t[:, None] * (M_drive @ c.T).T
        conn = (wdk[:, None] / (4.0 * wk[:, None])) * (M_conn @ c.T).T
        dc = (-1j / HBAR) * (E * c + drive) + conn
        return dc.reshape(-1)

    c0 = np.zeros((K, Nmax), dtype=complex); c0[:, 0] = 1.0
    sol = solve_ivp(rhs, (t0, T), c0.reshape(-1), t_eval=t_eval,
                    method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"fock_variable_om failed: {sol.message}")
    c = sol.y.T.reshape(len(t_eval), K, Nmax)
    nbar = np.einsum("tkn,n->tk", np.abs(c) ** 2, n)
    P0 = np.abs(c[:, :, 0]) ** 2
    return dict(c=c, nbar=nbar, P0=P0, t=t_eval)

def fock_variable_om_fixed(x0_of_t, omega_of_t, t_eval, mass=M_RB87,
                           Nmax=24, dt_w=0.05):
    """Instantaneous-basis parametric evolution with vectorised fixed-step RK4.

    Same physics as fock_variable_om but without solve_ivp: suitable for
    long trajectories (10^3+ oscillation periods) where the adaptive
    integrator's dense output dominates the cost.  dt = dt_w / max(ω).
    Returns dict: c (Nt,K,Nmax), nbar (Nt,K), P0 (Nt,K), t.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    t0, T = t_eval[0], t_eval[-1]
    Nq = max(1024, 16 * len(t_eval))
    tq = np.linspace(t0, T, Nq)
    x0, d1, d2 = _trajectory_derivs(x0_of_t, tq, None)
    K = x0.shape[1]
    F = -mass * d2
    W = np.atleast_2d(np.asarray(omega_of_t(tq), dtype=float))
    if W.shape[0] != len(tq):
        W = W.T
    if W.shape[1] == 1 and K > 1:
        W = np.repeat(W, K, axis=1)
    wd = PchipInterpolator(tq, W, axis=0).derivative()(tq)
    n = np.arange(Nmax)
    off1 = [np.sqrt(np.arange(1, Nmax)), np.sqrt(np.arange(1, Nmax))]
    M_drive = diags(off1, [1, -1], shape=(Nmax, Nmax), format="csr") / np.sqrt(2.0)
    off2 = [np.sqrt(np.arange(2, Nmax + 1) * np.arange(1, Nmax)),
            -np.sqrt(np.arange(2, Nmax + 1) * np.arange(1, Nmax))]
    M_conn = diags(off2, [2, -2], shape=(Nmax, Nmax), format="csr")
    Fint = PchipInterpolator(tq, F, axis=0)
    Wint = PchipInterpolator(tq, W, axis=0)
    WDint = PchipInterpolator(tq, wd, axis=0)

    w_max = float(np.max(np.abs(W))) * 1.05
    dt = dt_w / w_max
    n_steps = int(np.ceil((T - t0) / dt))
    dt = (T - t0) / n_steps
    snap = np.round((t_eval - t0) / dt).astype(int)
    snap = np.clip(snap, 0, n_steps)

    def rhs(t, c):
        wk = Wint(t)
        fk = Fint(t)
        wdk = WDint(t)
        a0t = np.sqrt(HBAR / (mass * np.maximum(wk, 1e-30)))
        E = HBAR * wk[:, None] * (n + 0.5)[None, :]
        drive = -fk[:, None] * a0t[:, None] * (M_drive @ c.T).T
        conn = (wdk[:, None] / (4.0 * wk[:, None])) * (M_conn @ c.T).T
        return (-1j / HBAR) * (E * c + drive) + conn

    c = np.zeros((K, Nmax), dtype=complex)
    c[:, 0] = 1.0
    out = np.zeros((len(t_eval), K, Nmax), dtype=complex)
    out[0] = c
    s = 1
    for i in range(1, n_steps + 1):
        tt = t0 + (i - 1) * dt
        k1 = rhs(tt, c)
        k2 = rhs(tt + dt / 2, c + dt * k1 / 2)
        k3 = rhs(tt + dt / 2, c + dt * k2 / 2)
        k4 = rhs(tt + dt, c + dt * k3)
        c = c + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if s < len(snap) and i == snap[s]:
            out[s] = c
            s += 1
    nbar = np.einsum("tkn,n->tk", np.abs(out) ** 2, n)
    P0 = np.abs(out[:, :, 0]) ** 2
    return dict(c=out, nbar=nbar, P0=P0, t=t_eval)


# ═════════════════════════════════════════════════════════════════════
# Analytic benchmarks (independent derivations)
# ═════════════════════════════════════════════════════════════════════

def nbar_analytic_driven(acc, t, omega, mass=M_RB87):
    """n̄ = (m/2ħω)|∫ ẍ0(t) e^{iωt} dt|²  on a sampled trajectory."""
    integ = np.trapezoid(acc * np.exp(1j * omega * t), t)
    return (mass / (2.0 * HBAR * omega)) * np.abs(integ) ** 2


def nbar_staircase(x0_frames, dt, omega, mass=M_RB87):
    """Exact n̄ for a staircase of frame jumps (K atoms):

    n̄ = (1/2a0²) |Σ_f δx_f e^{iω t_f}|²   (each jump = velocity impulse).
    """
    dx = np.diff(x0_frames, axis=0)
    t = np.arange(x0_frames.shape[0]) * dt
    a0 = a_ho(omega, mass)
    S = np.exp(1j * omega * t[1:]) @ dx
    return np.abs(S) ** 2 / (2.0 * a0 ** 2)


def nbar_resonant_drive(F0, t, omega, mass=M_RB87):
    """Resonant drive F = F0·cos(ωt): n̄(t) = F0² t²/(8 m ħ ω)  (t ≫ 1/ω).

    Derivation: ẍ0 = −(F0/m)cos(ωt) ⇒ ∫ẍ0 e^{iωs}ds ≈ −(F0/m)·t/2 on resonance,
    n̄ = (m/2ħω)|∫|² = F0²t²/(8mħω).  Cross-checked against the classical
    amplitude y_max = F0t/(2mω) with n̄ = y_max²/(2a0²) (same value).
    """
    return F0 ** 2 * t ** 2 / (8.0 * mass * HBAR * omega)


def nbar_analytic_frames_axis(x0_frames, dt, omega, mass=M_RB87):
    """n̄_k = (m/2ħω)|∫ ẍ0(t) e^{iωt} dt|² on discrete frames (per axis).

    ẍ0 = second difference of the frame sequence (smooth-interpolant view,
    i.e. the estimator used by the earlier pipeline reports).  For the
    physical SLM staircase use nbar_staircase instead.
    x0_frames (F,K) [m]; returns (K,) n̄ per atom.
    """
    x0 = np.atleast_2d(x0_frames)
    if x0.shape[0] == 1:
        return np.zeros(x0.shape[1])
    F, K = x0.shape
    acc = np.zeros_like(x0)
    if F >= 3:
        acc[1:-1] = (x0[2:] - 2.0 * x0[1:-1] + x0[:-2]) / dt ** 2
    t = np.arange(F) * dt
    integ = np.trapezoid(acc * np.exp(1j * omega * t)[:, None], t, axis=0)
    return (mass / (2.0 * HBAR * omega)) * np.abs(integ) ** 2


def nbar_sudden(dx, omega, mass=M_RB87):
    """Sudden trap displacement: n̄ = dx²/(2a0²)."""
    return dx ** 2 / (2.0 * a_ho(omega, mass) ** 2)


def coherent_amplitudes(alpha, Nmax):
    """Fock amplitudes of |α⟩: c_n = e^{−|α|²/2} αⁿ/√n!."""
    n = np.arange(Nmax)
    fact = np.array([np.math.factorial(i) for i in n])
    return np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(fact)

