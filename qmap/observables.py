"""Observables: phonon statistics, instantaneous eigenbases, Wigner functions.

n̄(t) and P0(t) are defined w.r.t. the *instantaneous* trap eigenbasis:
for a moving harmonic trap these are the Fock states of the instantaneous
frequency ω(t); for an anharmonic (Gaussian) tweezer we compute the true
eigenstates on the grid (Lanczos) and project onto them.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags

from .constants import HBAR, M_RB87, a_ho


def kinetic_matrix(Nx: int, dx: float, mass: float):
    """Kinetic energy matrix -(ħ²/2m)∂²x on a uniform grid (2nd order)."""
    off = -HBAR ** 2 / (2.0 * mass * dx ** 2)
    return diags([off, -2.0 * off, off], [-1, 0, 1], shape=(Nx, Nx), format="csr")


def instantaneous_eigenbasis(Vx, x, n_eig=16, mass=M_RB87):
    """Lowest n_eig eigenstates of -(ħ²/2m)∂²x + V(x) on the grid.

    Returns (E (n_eig,), psi (n_eig, Nx)) real, orthonormal.
    """
    dx = x[1] - x[0]
    H = kinetic_matrix(len(x), dx, mass) + diags(Vx, 0, shape=(len(x), len(x)), format="csr")
    E, psi = eigsh(H, k=min(n_eig, len(x) - 2), which="SM")
    order = np.argsort(E)
    return E[order], psi[:, order].T


def phonon_stats(psi, basis, n_phonon=40):
    """Project wavefunction(s) onto an orthonormal basis; return stats.

    psi   : (Nx,) or (K, Nx)  (complex)
    basis : (Nb, Nx) orthonormal (real ok)
    Returns dict: nbar (K,), P0 (K,), Pn (K, n_phonon).
    """
    psi = np.atleast_2d(psi)
    K = psi.shape[0]
    c = basis @ psi.T.conj()            # (Nb, K)
    c = c.T                             # (K, Nb)
    p = np.abs(c) ** 2
    n = np.arange(min(p.shape[1], n_phonon))
    pn = p[:, :len(n)]
    nbar = np.einsum("kn,n->k", pn, n)
    P0 = p[:, 0] if p.shape[1] > 0 else np.zeros(K)
    return dict(nbar=nbar, P0=P0, Pn=pn, c=c)


def wigner_1d(psi, x, p):
    """Discrete Wigner function W(x,p) = (1/πħ)∫dy ψ*(x+y)ψ(x-y) e^{2ipy/ħ}.

    Convention: ∫∫ W dx dp/(2πħ) = 1.  Quadrature on the grid (snapshots).
    """
    psi = np.asarray(psi).ravel().astype(complex)
    Nx = len(x)
    dx = x[1] - x[0]
    W = np.zeros((len(p), Nx), dtype=float)
    for i in range(Nx):
        for j in range(Nx):
            im, ip = i - j, i + j
            if 0 <= im < Nx and 0 <= ip < Nx:
                amp = psi[im] * np.conj(psi[ip])
                W[:, i] += 2.0 * np.real(amp * np.exp(2j * p * (x[j] - x[0]) / HBAR))
    W *= dx / (np.pi * HBAR)
    return W


def gaussian_fit_1d(I, x, x_guess=None):
    """Fit I(x) = A exp(-(x-x0)²/(2σ²)) + b by log-least-squares.

    Returns (A, x0, sigma, b).  Robust to flat tails.
    """
    I = np.asarray(I, dtype=float)
    if x_guess is None:
        x_guess = x[int(np.argmax(I))]
    k = int(np.argmin(np.abs(x - x_guess)))
    # crude width from FWHM around the max
    half = I.max() / 2.0
    left = np.where(I[:k] <= half)[0]
    right = np.where(I[k:] <= half)[0]
    wl = x[k] - x[left[-1]] if len(left) else 0.5 * (x[-1] - x[0])
    wr = x[k + right[0]] - x[k] if len(right) else 0.5 * (x[-1] - x[0])
    sig_g = max(0.5 * (wr + wl), 1e-12)
    b0 = I.min()
    y = np.log(np.clip(I - b0, 1e-12, None))
    w = np.clip(I - b0, 1e-12, None)
    A = np.vstack([np.ones_like(x), x, x ** 2]).T
    Wm = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * Wm[:, None], y * Wm, rcond=None)
    a2 = coef[2]
    if a2 >= 0:
        a2 = -1.0 / (2.0 * sig_g ** 2)
    sigma = np.sqrt(-1.0 / (2.0 * a2))
    x0 = -coef[1] / (2.0 * a2) if a2 != 0 else x_guess
    A0 = float(np.exp(np.clip(coef[0] - a2 * x0 ** 2, -700, 700)))
    if not np.isfinite(A0) or A0 <= 0:
        A0 = float(I.max())
    return A0, float(x0), float(sigma), float(b0)


def trap_params_from_intensity(I, x, omega_ref, mass=M_RB87):
    """Trap parameters from 1-D intensity slices of the focal field.

    I         : (F, Nx) intensity along the motion direction per frame
    x         : (Nx,) positions [m]
    omega_ref : reference frequency [rad/s] (calibrates depth)
    Returns (x0 (F,), omega (F,), U0 (F,), sigma (F,)).
    """
    F = I.shape[0]
    x0 = np.zeros(F); sigma = np.zeros(F); A0 = np.zeros(F)
    for f in range(F):
        A, xc, s, b = gaussian_fit_1d(I[f], x, x_guess=x[int(I[f].argmax())])
        x0[f] = xc; sigma[f] = s; A0[f] = A
    A_ref = A0.mean() if A0.mean() > 0 else 1.0
    omega = omega_ref * np.sqrt(np.clip(A0 / A_ref, 1e-6, None))
    U0 = 0.5 * mass * omega ** 2 * sigma ** 2
    return x0, omega, U0, sigma
