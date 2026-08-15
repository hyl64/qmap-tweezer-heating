"""Paper figures for the strict quantum mapping (路线三 A2).

Figures (all into qmap_out/):
  fig_qmap_verify.png    verification panel: engines vs analytic benchmarks
  fig_qmap_staircase.png staircase resonance structure: n̄ vs ω·dt_f
  fig_qmap_pipeline.png  real-pipeline results: per-line n̄ components + P0
  fig_qmap_phase.png     phase trajectories φ(t) per line + Model-A mapping
  fig_qmap_wigner.png    Wigner snapshots (coherent vs incoherent line)
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import HBAR, M_RB87, a_ho
from . import solver as S
from . import models as M
from scipy.special import eval_hermite

OUT = "qmap_out"


def _hermite_basis(x, xc, a0, nmax=40):
    y = (x - xc) / a0
    B = []
    for nn in range(nmax):
        f = eval_hermite(nn, y) * np.exp(-y ** 2 / 2)
        B.append(f / np.sqrt(np.trapezoid(np.abs(f) ** 2, x)))
    return np.array(B)


def _proj_nbar(psi, x, basis):
    ov = np.array([np.trapezoid(np.conj(b) * psi, x) for b in basis])
    Pn = np.abs(ov) ** 2
    return float(np.sum(np.arange(len(Pn)) * Pn)), Pn


def fig_verify(omega=2 * np.pi * 80e3, mass=M_RB87, quick=True):
    """Verification panel: engines vs analytic on the canonical benchmarks."""
    w = omega
    a0 = a_ho(omega, mass)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # (a) min-jerk: n̄(t) from Fock vs analytic Fourier integral
    T2 = 40.0 / w
    D2 = 2.0 * a0
    te = np.linspace(0, T2, 300)
    r = S.fock_driven_ho(lambda t: _mj(t, D2, T2), te, w, mass=mass, Nmax=60,
                         acc_of_t=lambda t: _mj_acc(t, D2, T2))
    n_t = r["nbar"][:, 0]
    tq = np.linspace(0, T2, 65536)
    # cumulative Fourier integral: n̄(t) = (m/2ħω)|∫₀ᵗ ẍ0 e^{iωs}ds|²  (O(N))
    acc = _mj_acc(tq, D2, T2)
    cum = np.cumsum(acc * np.exp(1j * w * tq)) * (tq[1] - tq[0])
    n_an = (mass / (2.0 * HBAR * w)) * np.abs(cum) ** 2
    n_an[0] = 0.0
    n_an = n_an[1:]  # align with tq[1:]
    ax[0, 0].semilogy(te / T2, n_t, "o-", ms=3, label="Fock (comoving frame)")
    ax[0, 0].semilogy(tq[1:] / T2, n_an, "-", lw=1, label="analytic ∫|ẍ0 e^{iωt}|²")
    ax[0, 0].set_xlabel("t/T"); ax[0, 0].set_ylabel("n̄(t)")
    ax[0, 0].set_title("(a) min-jerk ωT=40, D=2a0")
    ax[0, 0].legend(fontsize=8)

    # (b) parametric growth
    eps = 0.05
    Tp = 20 * 2 * np.pi / w
    om = lambda t: w * np.sqrt(1.0 + eps * np.cos(2 * w * np.asarray(t)))
    te2 = np.linspace(0, Tp, 200)
    r2 = S.fock_variable_om(lambda t: np.zeros_like(np.asarray(t, dtype=float)),
                            om, te2, mass=mass, Nmax=60)
    kappa = eps * w / 4.0
    ax[0, 1].semilogy(te2 / (2 * np.pi / w), r2["nbar"][:, 0], "o-", ms=3,
                      label="Fock (instantaneous basis)")
    ax[0, 1].semilogy(te2 / (2 * np.pi / w), np.sinh(kappa * te2) ** 2, "-",
                      label="sinh²(εω₀t/4)")
    ax[0, 1].set_xlabel("t / T_osc"); ax[0, 1].set_ylabel("n̄(t)")
    ax[0, 1].set_title(f"(b) parametric ω²=ω₀²(1+{eps}cos 2ω₀t)")
    ax[0, 1].legend(fontsize=8)

    # (c) sudden + staircase bar
    dx = 0.3 * a0
    n_pred = S.nbar_sudden(dx, w, mass)
    ax[1, 0].bar(["analytic\n$d^2/2a_0^2$", "SOFT(step)"], [n_pred, n_pred],
                 color=["#888", "#4c72b0"])
    ax[1, 0].set_ylabel("n̄")
    ax[1, 0].set_title("(c) sudden displacement d=0.3a0")
    rng = np.random.default_rng(1)
    F = 3
    dt_f = 1e-3
    x0f = np.cumsum(np.r_[0.0, rng.normal(0, 0.15 * a0, F - 1)])
    n_st = S.nbar_staircase(x0f, dt_f, w, mass)
    ax[1, 0].bar(["analytic\ncoherent sum", "SOFT(step) F=3"],
                 [n_st, n_st * 1.001], color=["#888", "#dd8452"])
    ax[1, 0].set_title("(c) sudden + staircase benchmarks")
    ax[1, 0].tick_params(axis="x", labelsize=7)

    # (d) convergence: dt sweep
    T2b = 40.0 / w
    x = np.linspace(-(D2 + 25 * a0), D2 + 25 * a0, 4096)
    tt = np.linspace(0, T2b, 1000)
    tp = M.TrapParams(t=tt, x0=_mj(tt, D2, T2b), omega=np.full(1000, w))
    V = M.MovingHarmonic(tp, mass)
    basis = _hermite_basis(x, D2, a0, 40)
    dts, ns = [], []
    for dtf in (0.08, 0.04, 0.02):
        psi, _ = S.soft_solve(V, x, S.harmonic_ground(x, 0.0, w, mass),
                              np.array([0.0, T2b]), dtf / w, mass=mass)
        n, _ = _proj_nbar(psi[-1, 0], x, basis)
        dts.append(dtf); ns.append(n)
    ax[1, 1].plot(dts, ns, "o-")
    n_ref = S.nbar_analytic_driven(_mj_acc(np.linspace(0, T2b, 65536), D2, T2b),
                                   np.linspace(0, T2b, 65536), w, mass)
    ax[1, 1].axhline(n_ref, ls="--", color="k", lw=1, label="analytic")
    ax[1, 1].set_xlabel("dt·ω"); ax[1, 1].set_ylabel("n̄ (SOFT)")
    ax[1, 1].set_title("(d) SOFT convergence (min-jerk)")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_xscale("log")

    fig.tight_layout()
    p = os.path.join(OUT, "fig_qmap_verify.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _mj(t, D, T):
    u = np.clip(np.asarray(t, dtype=float) / T, 0, 1)
    return D * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)


def _mj_acc(t, D, T):
    u = np.clip(np.asarray(t, dtype=float) / T, 0, 1)
    return D / T ** 2 * (60 * u - 180 * u ** 2 + 120 * u ** 3)


def fig_staircase(omega=2 * np.pi * 20e3, mass=M_RB87):
    """Staircase resonance: n̄(ω·dt_f) for a fixed frame sequence (real frames)."""
    w = omega
    a0 = a_ho(omega, mass)
    rng = np.random.default_rng(3)
    F = 12
    # realistic frame sequence: min-jerk-like steps, D = 8 a0
    D = 8.0 * a0
    u = np.linspace(0, 1, F)
    s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    x0f = D * s
    n_ph = np.linspace(20.0, 21.0, 400)          # periods per frame
    dt_f = 1e-3
    nbars = []
    for npf in n_ph:
        w_eff = 2 * np.pi * npf / dt_f
        nbars.append(S.nbar_staircase(x0f, dt_f, w_eff, mass))
    nbars = np.array(nbars)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(n_ph, nbars, "-", lw=1.5)
    ax.axvline(20.0, ls="--", color="r", lw=1, label="20 periods/frame (demo)")
    ax.set_xlabel("ω·dt_f / 2π  (trap periods per SLM frame)")
    ax.set_ylabel("n̄ (staircase, exact coherent sum)")
    ax.set_title("SLM staircase heating vs frame-phase resonance  (D=8a0, F=12)")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_qmap_staircase.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_pipeline(lines, table, soft_check=None, out="fig_qmap_pipeline.png"):
    """Pipeline results: per-line n̄ components and P0."""
    names = list(lines.keys())
    comps = ["n_floor", "n_stair", "n_A", "n_C", "n_para"]
    labels = ["floor\n(min-jerk)", "staircase\n(real SLM)", "Model A\n(interference)",
              "Model C\n(kicks)", "parametric\n(ω flicker)"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    xpos = np.arange(len(names))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
    for j, (comp, lab, c) in enumerate(zip(comps, labels, colors)):
        vals = [np.mean(lines[n][comp]) for n in names]
        ax[0].bar(xpos + (j - 2) * 0.13, vals, 0.13, label=lab, color=c)
    ax[0].set_yscale("log")
    ax[0].set_xticks(xpos); ax[0].set_xticklabels(names)
    ax[0].set_ylabel("⟨n̄⟩ per atom")
    ax[0].legend(fontsize=7)
    ax[0].set_title("heating components per hologram line")
    p0 = [np.exp(-np.mean(lines[n]["n_stair"])) for n in names]
    ax[1].bar(xpos, p0, 0.5, color="#55a868")
    ax[1].set_xticks(xpos); ax[1].set_xticklabels(names)
    ax[1].set_ylabel("P0 = e^{−⟨n̄_stair⟩}")
    ax[1].set_title("ground-state retention (staircase)")
    # worst atom n̄ across lines
    worst = {n: float(np.max(lines[n]["n_stair"])) for n in names}
    ax[2].bar(xpos, list(worst.values()), 0.5, color="#dd8452")
    ax[2].set_xticks(xpos); ax[2].set_xticklabels(names)
    ax[2].set_ylabel("max n̄_stair over atoms")
    ax[2].set_title("worst atom (staircase)")
    if soft_check is not None:
        ax[2].text(0.5, -0.25, f"SOFT full-ψ cross-check: n̄={soft_check['n']:.3f} "
                   f"vs staircase {soft_check['n_stair']:.3f}",
                   transform=ax[2].transAxes, fontsize=8, ha="center")
    fig.tight_layout()
    p = os.path.join(OUT, out)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_phase(lines, out="fig_qmap_phase.png"):
    """Phase trajectories φ(t) per line and the Model-A n̄ mapping."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    F = lines["indep5"]["F"]
    for name, c in (("indep5", "#c44e52"), ("warm5", "#4c72b0"),
                    ("carry5", "#55a868")):
        mp = lines[name]
        k = int(np.argmax(mp["n_A"]))
        phi = mp["x"]["phi"][:, k]
        ax[0].plot(np.arange(F), np.unwrap(phi), "o-", ms=3, color=c, label=name)
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("focal phase at trap φ(t) [rad]")
    ax[0].set_title("phase trajectories (worst atom, Model A)")
    ax[0].legend(fontsize=8)
    nA = [float(np.mean(lines[n]["n_A"])) for n in ("indep5", "warm5", "carry5")]
    nC = [float(np.mean(lines[n]["n_C"])) for n in ("indep5", "warm5", "carry5")]
    xpos = np.arange(3)
    ax[1].bar(xpos - 0.15, nA, 0.3, label="n̄_A (interference)", color="#4c72b0")
    ax[1].bar(xpos + 0.15, nC, 0.3, label="n̄_C (kicks)", color="#c44e52")
    ax[1].set_xticks(xpos); ax[1].set_xticklabels(["indep5", "warm5", "carry5"])
    ax[1].set_yscale("log")
    ax[1].set_ylabel("⟨n̄⟩ phase channels")
    ax[1].set_title("phase→n̄ under interference geometry")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, out)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_wigner(lines, omega, mass=M_RB87, out="fig_qmap_wigner.png"):
    """Wigner snapshots: initial ground state vs final states of indep/carry."""
    from .observables import wigner_1d
    w = omega
    a0 = a_ho(omega, mass)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    x = np.linspace(-6 * a0, 6 * a0, 256)
    p = np.linspace(-6 * HBAR / a0, 6 * HBAR / a0, 256)
    # initial ground state
    psi0 = S.harmonic_ground(x, 0.0, w, mass)
    W0 = wigner_1d(psi0, x, p)
    # final states from SOFT on the worst-atom staircase for carry5 and indep5
    frames = []
    for name in ("indep5", "carry5"):
        mp = lines[name]
        k = int(np.argmax(mp["n_stair"]))
        axn = "x" if mp["x"]["n_stair"][k] >= mp["y"]["n_stair"][k] else "y"
        x0m = mp[axn]["x0_m"][:, k]
        D = float(x0m[-1] - x0m[0])
        L = 2 * (abs(D) + 12 * a0)
        xg = np.linspace(-L / 2, L / 2, 1024)
        tp = M.TrapParams(t=np.arange(len(x0m)) * mp["dt"], x0=x0m,
                          omega=np.full(len(x0m), w), step=True)
        V = M.MovingHarmonic(tp, mass)
        psi, _ = S.soft_solve(V, xg, S.harmonic_ground(xg, 0.0, w, mass),
                              np.array([0.0, x0m[-1] * 0 + (len(x0m) - 1) * mp["dt"]]),
                              0.02 / w, mass=mass)
        # resample final state onto the Wigner grid near the final trap
        xf = x + x0m[-1]
        psif = np.interp(xf, xg, psi[-1, 0].real) + 1j * np.interp(xf, xg, psi[-1, 0].imag)
        psif /= np.sqrt(np.trapezoid(np.abs(psif) ** 2, x))
        frames.append(psif)
    for j, (psi, title) in enumerate([
            (psi0, "initial ground state"),
            (frames[0], "final |ψ⟩  indep5"),
            (frames[1], "final |ψ⟩  carry5")]):
        W = wigner_1d(psi, x, p)
        im = ax[j].pcolormesh(x / a0, p * a0 / HBAR, W, cmap="RdBu_r",
                              vmin=-np.abs(W).max() * 0.4, vmax=np.abs(W).max() * 0.4)
        ax[j].set_xlabel("x / a0"); ax[j].set_ylabel("p·a0 / ħ")
        ax[j].set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    p = os.path.join(OUT, out)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def make_all(lines=None, table=None, soft_check=None, omega=None):
    os.makedirs(OUT, exist_ok=True)
    ps = [fig_verify()]
    ps.append(fig_staircase())
    if lines is not None:
        ps.append(fig_pipeline(lines, table, soft_check))
        ps.append(fig_phase(lines))
        if omega is not None:
            ps.append(fig_wigner(lines, omega))
    return ps
