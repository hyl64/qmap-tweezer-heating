"""Pipeline integration: real P2WGS hologram lines → strict quantum mapping.

Chain (路线三 A2 on the real pipeline):
    SLM phase frames φ_SLM (indep5/warm5/carry5, from coherent_pipeline)
      → focal fields U_f (propagation)
      → per-atom trap extraction: centroid x0(t), frequency ω(t),
        focal phase φ(t) at the trap, intensity I(t)
      → coupling models (qmap.models) → effective trap parameters
      → exact quantum engines (qmap.solver) → n̄(t), P0(t), P(n)

What is computed per line/atom (both focal-plane axes x, y; n̄ = n̄x + n̄y):
  n̄_floor : commanded min-jerk smooth trajectory (adiabatic floor, exact
             Fourier formula — what the previous reports called the floor)
  n̄_stair : the REAL SLM trajectory — piecewise-constant trap position
             between frames (staircase).  Exact coherent sum
             n̄ = (1/2a0²)|Σ δx_f e^{iωt_f}|².  This is the dominant term
             whenever ω·dt_f is near a multiple of 2π (jumps add in phase)
             and is MISSED by smooth-trajectory estimators.
  n̄_A     : interference geometry (Model A): phase transduces to position
             δx = −φ/(2k_eff) ⇒ the phase trajectory becomes part of the
             trap trajectory.  The cleanest phase→n̄ channel.
  n̄_C     : phase-gradient momentum kicks (Model C, honest k0 cap).
  n̄_para  : intensity flicker → ω(t) modulation (parametric, off-resonant
             at frame rate — expected ≈ 0; verified).

Outputs: qmap_out/pipeline_*.npz (all trajectories + results), CSV table,
and per-line quantum numbers.
"""
from __future__ import annotations

import os, time
import numpy as np

from .constants import HBAR, M_RB87, a_ho, k_light
from . import solver as S
from . import models as M
from . import observables as OBS


# ─────────────────────────────────────────────────────────────────────
# trap extraction from focal fields
# ─────────────────────────────────────────────────────────────────────

def extract_traps(U_f: np.ndarray, spots_px: np.ndarray, px_m: float,
                  R: int = 4, min_sig_px: float = 0.3):
    """Per-atom trap parameters from focal fields.

    U_f      : (F, N, N) complex focal field per frame
    spots_px : (F, K, 2) trap pixel positions (row, col)
    px_m     : pixel size [m]
    Returns dict with arrays (F, K):
        cx, cy      centroid [px]  (Gaussian fit over R-window)
        sig_x, sig_y [px]
        A           peak intensity
        phi         focal phase at the centroid [rad]
    """
    F, N, _ = U_f.shape
    K = spots_px.shape[1]
    cx = np.full((F, K), np.nan); cy = np.full((F, K), np.nan)
    sigx = np.full((F, K), np.nan); sigy = np.full((F, K), np.nan)
    A = np.full((F, K), np.nan); phi = np.full((F, K), np.nan)
    coords = np.arange(N)
    for f in range(F):
        I = np.abs(U_f[f]) ** 2
        for k in range(K):
            r0, c0 = spots_px[f, k]
            r0i, c0i = int(round(r0)), int(round(c0))
            rl, rh = max(0, r0i - R), min(N, r0i + R + 1)
            cl, ch = max(0, c0i - R), min(N, c0i + R + 1)
            patch = I[rl:rh, cl:ch]
            if patch.size == 0 or patch.max() <= 0:
                continue
            rr = coords[rl:rh]; cc = coords[cl:ch]
            # 1-D Gaussian fits along each pixel axis (row profile / col profile)
            row_prof = patch.sum(axis=1)
            col_prof = patch.sum(axis=0)
            Ar = Ac = float(np.nan)
            try:
                Ar, x0r, sr, _ = OBS.gaussian_fit_1d(row_prof, rr, rr[int(np.argmax(row_prof))])
                Ac, x0c, sc, _ = OBS.gaussian_fit_1d(col_prof, cc, cc[int(np.argmax(col_prof))])
            except Exception:
                x0r = x0c = float("nan")
                sr = sc = float("nan")
            # sanity: centroid within 3 px of the commanded spot, sigma in range
            ok_r = (np.isfinite(x0r) and abs(x0r - r0) < 3.0
                    and np.isfinite(sr) and min_sig_px <= sr <= 8.0)
            ok_c = (np.isfinite(x0c) and abs(x0c - c0) < 3.0
                    and np.isfinite(sc) and min_sig_px <= sc <= 8.0)
            if not ok_r:
                x0r, sr, Ar = r0, 1.5, float(np.nan)
            if not ok_c:
                x0c, sc, Ac = c0, 1.5, float(np.nan)
            cx[f, k] = x0r; cy[f, k] = x0c
            sigx[f, k] = max(sr, min_sig_px); sigy[f, k] = max(sc, min_sig_px)
            # amplitude: geometric mean of the two fitted peaks (robust),
            # fall back to the raw pixel if both fits failed
            A[f, k] = np.sqrt(Ar * Ac) if (np.isfinite(Ar) and np.isfinite(Ac)) else I[r0i, c0i]
            phi[f, k] = np.angle(U_f[f, r0i, c0i])
    # NaN safety: bad fits already fall back; intensity floor for A
    A = np.where(np.isfinite(A) & (A > 0), A, np.nan)
    for f in range(F):
        for k in range(K):
            if not np.isfinite(A[f, k]):
                A[f, k] = np.nanmean(A[f]) if np.any(np.isfinite(A[f])) else 1.0
    return dict(cx=cx, cy=cy, sig_x=sigx, sig_y=sigy, A=A, phi=phi)


# ─────────────────────────────────────────────────────────────────────
# quantum mapping per line
# ─────────────────────────────────────────────────────────────────────

def nbar_stair_axis(x0_frames: np.ndarray, dt: float, omega: float,
                    om_frames: np.ndarray | None = None,
                    mass: float = M_RB87):
    """Exact staircase n̄ for (F,K) frame positions along one axis.

    om_frames: measured per-frame trap frequency.  When given, the phase
    carried by each jump is the ACTUAL accumulated phase Σω_j·dt (the
    intensity flicker de-phases the frame-to-frame coherent addition);
    otherwise the constant-ω (resonance-locked) phases e^{iωt_f} are used.
    Returns (n_coherent, n_incoherent).
    """
    F, K = x0_frames.shape
    dx = np.diff(x0_frames, axis=0)
    if om_frames is not None:
        ph = np.exp(1j * np.cumsum(np.asarray(om_frames)[:-1], axis=0) * dt)
    else:
        t = np.arange(F) * dt
        ph = np.exp(1j * omega * t[1:])
        ph = ph[:, None]                     # (F-1, 1) broadcast over K
    alpha = np.sum(ph * dx, axis=0)          # (K,) coherent sum
    a0 = a_ho(omega, mass)
    return np.abs(alpha) ** 2 / (2.0 * a0 ** 2), np.abs(dx) ** 2 / (2.0 * a0 ** 2)


def map_line(ext: dict, frames_mj: np.ndarray, px_m: float, dt: float,
             omega: float, f_khz: float, lam: float = 0.80e-6,
             mass: float = M_RB87, k_eff=None, step_m=None,
             do_soft_check: bool = False, seed: int = 0):
    """Full quantum mapping for one hologram line.

    ext       : extraction dict (cx, cy, sig_x, sig_y, A, phi) (F,K)
    frames_mj : commanded min-jerk frames (F,K,2) px
    Returns a dict of per-axis + total results.
    """
    F, K = ext["cx"].shape
    a0 = a_ho(omega, mass)
    if k_eff is None:
        k_eff = k_light(lam)
    if step_m is None:
        step_m = 2.0 * px_m            # ~trap diameter as gradient baseline
    w = omega
    res = dict(F=F, K=K, dt=dt, omega=omega, a0=a0, px_m=px_m, f_khz=f_khz)

    for ax, (col, sigcol) in (("x", ("cx", "sig_x")), ("y", ("cy", "sig_y"))):
        x0_cmd_px = frames_mj[:, :, 1 if col == "cy" else 0]   # col index for x?
        # NOTE: cx is the row (axis 0 of the array) = 'y' in physics; keep labels
        cent_px = ext[col]                          # (F,K) px
        cent_m = cent_px * px_m
        x0_m = cent_m
        I_rel = ext["A"] / np.maximum(np.nanmean(ext["A"], axis=0, keepdims=True), 1e-18)
        om_t = omega * np.sqrt(np.clip(I_rel, 1e-6, None))     # (F,K)
        phi = ext["phi"]                            # (F,K)
        t_f = np.arange(F) * dt

        # 1) smooth commanded floor (analytic, per atom)
        #    commanded frames along this axis
        cmd = frames_mj[:, :, 1] if col == "cy" else frames_mj[:, :, 0]
        x0_cmd = cmd * px_m
        n_floor = S.nbar_analytic_frames_axis(x0_cmd, dt, omega, mass)
        # 2) staircase of measured centroids (the real SLM trajectory)
        n_stair, n_stair_inc = nbar_stair_axis(x0_m, dt, omega, mass=mass)
        #    phase-corrected: the intensity flicker de-phases the coherent sum
        n_stair_ph, _ = nbar_stair_axis(x0_m, dt, omega, om_frames=om_t, mass=mass)
        # 3) Model A: interference transduction — effective position
        phi_u = np.unwrap(phi, axis=0)
        x0_A = x0_m - phi_u / (2.0 * k_eff)
        n_A, n_A_inc = nbar_stair_axis(x0_A, dt, omega, mass=mass)
        n_A_ph, _ = nbar_stair_axis(x0_A, dt, omega, om_frames=om_t, mass=mass)
        # 4) Model C: kicks
        dp = np.sign(np.diff(phi_u, axis=0)) * HBAR * np.minimum(
            np.abs(np.diff(phi_u, axis=0)) / step_m, k_light(lam))
        ph = np.exp(1j * omega * t_f[1:])
        alpha_c = (ph @ dp) / np.sqrt(2.0 * mass * HBAR * omega)
        n_C = np.abs(alpha_c) ** 2
        n_C_inc = np.sum(dp ** 2 / (2.0 * mass * HBAR * omega), axis=0)
        # 5) parametric (intensity flicker → ω(t)).
        # First-order exact estimate (|0⟩→|2⟩ channel, valid for the
        # off-resonant case):  c₂(T) = ∫ (ω̇/4ω) e^{i 2∫ω dt'} dt'
        #   n̄_para ≈ 2|c₂|²
        # The SOFT full-trap cross-check (below) covers this channel exactly.
        n_para = np.zeros(K)
        if np.any(np.abs(np.diff(om_t, axis=0)).max(axis=0) > 1e-9 * omega):
            n_para = _parametric_first_order(t_f, om_t)
        res[ax] = dict(x0_m=x0_m, om_t=om_t, phi=phi, I_rel=I_rel,
                       n_floor=n_floor, n_stair=n_stair, n_stair_inc=n_stair_inc,
                       n_stair_ph=n_stair_ph,
                       n_A=n_A, n_A_inc=n_A_inc, n_A_ph=n_A_ph,
                       n_C=n_C, n_C_inc=n_C_inc,
                       n_para=n_para)
    # totals
    rx, ry = res["x"], res["y"]
    for comp in ("floor", "stair", "stair_inc", "stair_ph",
                 "A", "A_inc", "A_ph", "C", "C_inc", "para"):
        res[f"n_{comp}"] = rx[f"n_{comp}"] + ry[f"n_{comp}"]
    res["n_total_model0"] = res["n_stair"] + res["n_floor"] + res["n_para"]
    res["n_total_modelA"] = res["n_A"] + res["n_floor"] + res["n_para"]
    res["n_total_modelC"] = res["n_C"] + res["n_floor"] + res["n_para"]
    return res


def _parametric_first_order(t_f, om_t, mass=M_RB87):
    """First-order parametric heating from the |0⟩→|2⟩ channel.

    Exact to leading order in the (small) ω(t) modulation:
        c̃₂(T) = √2 ∫₀ᵀ (ω̇/4ω) e^{i 2∫₀ᵗ ω dt'} dt',
        n̄ ≈ 2|c₂|² = 4 |∫ (ω̇/4ω) e^{i2∫ω}|².
    ω(t) is piecewise-linear between frames (np.interp), so each frame
    segment is integrated on a fine grid (no aliasing of the fast phase).
    The SOFT full-trap cross-check verifies this channel exactly.
    """
    F, K = om_t.shape
    n_sub = 4000                      # points per frame segment
    integ = np.zeros(K, dtype=complex)
    for f in range(F - 1):
        tau = np.linspace(0, t_f[1] - t_f[0], n_sub)
        w0f = om_t[f]                 # (K,)
        w1f = om_t[f + 1]
        wq = w0f[None, :] + (w1f - w0f)[None, :] * (tau / (t_f[1] - t_f[0]))[:, None]
        wd = (w1f - w0f)[None, :] / (t_f[1] - t_f[0])
        phase = np.cumsum(wq, axis=0) * (tau[1] - tau[0])
        integ += np.trapezoid((wd / (4.0 * wq)) * np.exp(2j * phase), tau, axis=0)
    return 4.0 * np.abs(integ) ** 2


def _interp_axis(tq, t_f, y):
    y = np.atleast_2d(y)
    if y.shape[0] != len(t_f):
        y = y.T
    out = np.column_stack([np.interp(tq, t_f, y[:, k]) for k in range(y.shape[1])])
    return out



# ─────────────────────────────────────────────────────────────────────
# hologram-line runner (GPU) — mirrors coherent_pipeline.run_hologram_line
# but also returns the per-frame focal fields
# ─────────────────────────────────────────────────────────────────────

def run_lines_holograms(frames_xy, A_slm, N, Kreal, iters, warm, carry, dev,
                        bits=None):
    """P2WGS hologram sequence; returns (phi_list, U_f (F,N,N) complex, I (F,K)).

    bits: optional SLM phase quantization (8 or 10) applied BEFORE propagation
    (real SLM constraint: phase-only + bit depth), mirroring slm_sim.slm_forward.
    """
    import torch
    import propagator as P
    import p2wgs as P2
    F, K, _ = frames_xy.shape
    phi_list, U_list, I_list = [], [], []
    sp0 = frames_xy[0]
    T0 = P2.trajectory_phase_target(None, A_slm, torch.from_numpy(sp0).to(dev).float(),
                                    spot_radius_px=2.0, N=N, device=dev, seed=0)
    spots0 = torch.zeros(1, max(K, 1), 2, device=dev).fill_(-1.)
    spots0[0, :K] = torch.from_numpy(sp0).to(dev)
    valid = torch.zeros(1, max(K, 1), dtype=torch.bool, device=dev)
    valid[0, :K] = True
    phi, _, _ = P2.p2wgs_batch(A_slm, T0, spots0, valid, iterations=50, gamma=0.6,
                               use_phase_weights=False, lock_mode="off",
                               spot_radius_px=2.0, seed=0, norm="ortho", device=dev)
    if bits is not None:
        L = 2 ** bits
        phi = torch.round(phi / (2 * np.pi) * L) / L * (2 * np.pi)
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
    U = P.propagate(A_slm[None], torch.cos(phi), torch.sin(phi), norm="ortho")
    U_list.append(U[0].cpu().numpy())
    mp = P.spot_mask_pixels(N, spots0, 2.0, device=dev)
    I_list.append(P.spot_intensities_sparse(U.real.pow(2) + U.imag.pow(2), mp, valid)[0, :K].cpu().numpy())
    phi_list.append(phi)
    for f in range(1, F):
        spf = frames_xy[f]
        spots_f = torch.zeros(1, max(K, 1), 2, device=dev).fill_(-1.)
        spots_f[0, :K] = torch.from_numpy(spf).to(dev)
        if carry:
            T = P2.trajectory_phase_target(phi_list[-1][0], A_slm,
                                           torch.from_numpy(spf).to(dev).float(),
                                           spot_radius_px=2.0, N=N, device=dev, seed=0)
        else:
            T = P2._gaussian_complex_target(N, torch.from_numpy(spf).to(dev).float(),
                                            torch.zeros(K, device=dev), 2.0, dev)
        init = phi_list[-1] if warm else None
        phi, _, _ = P2.p2wgs_batch(A_slm, T, spots_f, valid, iterations=iters, gamma=0.6,
                                   use_phase_weights=False, lock_mode="off",
                                   spot_radius_px=2.0, seed=0, norm="ortho",
                                   device=dev, phi_init=init)
        if bits is not None:
            L = 2 ** bits
            phi = torch.round(phi / (2 * np.pi) * L) / L * (2 * np.pi)
            phi = (phi + np.pi) % (2 * np.pi) - np.pi
        U = P.propagate(A_slm[None], torch.cos(phi), torch.sin(phi), norm="ortho")
        U_list.append(U[0].cpu().numpy())
        mp = P.spot_mask_pixels(N, spots_f, 2.0, device=dev)
        I_list.append(P.spot_intensities_sparse(U.real.pow(2) + U.imag.pow(2), mp, valid)[0, :K].cpu().numpy())
        phi_list.append(phi)
    return phi_list, np.stack(U_list, 0), np.stack(I_list, 0)


def soft_crosscheck(x0_m, om_t, omega, dt, mass=M_RB87, n_grid=2048,
                    margin_a0=25.0, dt_w=0.05, scale_d=1.0):
    """SOFT quantum run on the measured trap (one atom, one axis).

    Uses the measured frame-constant trap: staircase x0(t) AND the measured
    ω(t) (intensity flicker) — includes the parametric channel exactly.

    scale_d < 1 rescales the DISPLACEMENT only (x0 → x0_0 + s(x0 − x0_0)),
    keeping the frame timing, phases e^{iωt_f} and the relative ω-flicker.
    The split-operator error scales as (dt·ω·n̄)⁵, so the full-D pipeline
    case (n̄ ≈ 10³) is out of reach for the grid method; the analytic
    staircase formula (verified vs SOFT at n̄ ≤ 10 in verify.T4, multiple
    scales) is exact for constant-ω staircases.  This cross-check therefore
    validates the full extraction→potential→measurement machinery at
    accessible n̄ (scale_d ≈ 1/20 → n̄ ≈ 2), where SOFT is exact.
    """
    F = len(x0_m)
    a0 = a_ho(omega, mass)
    # relative trajectory: only the DISPLACEMENT matters (the atom starts
    # in the trap; an absolute offset would sit the atom at a huge well-edge
    # potential and poison the energy measurement)
    x0s = scale_d * (x0_m - x0_m[0])
    D = float(np.abs(x0s[-1] - x0s[0]))
    L = 2.0 * (D + margin_a0 * a0)
    x = np.linspace(-L / 2, L / 2, n_grid)
    t_f = np.arange(F) * dt
    om_f = np.asarray(om_t, dtype=float)
    tp = M.TrapParams(t=t_f, x0=x0s, omega=om_f, step=True)
    V = M.MovingHarmonic(tp, mass)
    V.prepare(x)
    psi0 = S.harmonic_ground(x, 0.0, omega, mass)
    t0 = time.time()
    psi, _ = S.soft_solve(V, x, psi0, np.array([0.0, t_f[-1]]), dt_w / omega,
                          mass=mass)
    psif = psi[-1, 0]
    # n̄ from the energy-moment formula (exact for any state, no truncation):
    #   n̄ = ⟨p²⟩/2mħω + mω⟨(x−x0f)²⟩/2ħ − ½
    dxg = x[1] - x[0]
    kk = 2 * np.pi * np.fft.fftfreq(len(x), d=dxg)
    dp = HBAR * kk * np.fft.fft(psif)
    T = np.trapezoid(np.abs(np.fft.ifft(dp)) ** 2, x) / (2 * mass)
    xf = float(x0s[-1])
    om_final = float(np.asarray(om_f)[-1])
    Vrel = 0.5 * mass * om_final ** 2 * np.trapezoid(np.abs(psif) ** 2 * (x - xf) ** 2, x)
    n = float((T + Vrel) / (HBAR * om_final) - 0.5)
    # analytic prediction at the SAME (scaled) trajectory: exact coherent sum
    # with the REAL flicker phases (leading order; the SOFT result includes
    # the a0(t) variation and is the definitive answer)
    n_pred, _ = nbar_stair_axis(x0s[:, None], dt, omega,
                                om_frames=np.asarray(om_f, dtype=float)[:, None],
                                mass=mass)
    n_pred = float(np.asarray(n_pred).ravel()[0])
    return dict(n=n, n_pred=n_pred, scale_d=scale_d,
                norm=float(np.trapezoid(np.abs(psif) ** 2, x)),
                sec=time.time() - t0)


# ─────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────

def run(seed=0, K=16, frames=14, Ngrid=256, f_khz=20.0, dt_ms=1.0,
        px_um=0.5, iters=5, device="cuda", lam=0.80e-6,
        do_soft_check=True, out_prefix="qmap_out/pipeline", bits=None):
    import torch
    from coherent_pipeline import plan_traj_auction
    import propagator as P
    import minjerk as MJ
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    dev = device if torch.cuda.is_available() else "cpu"

    # real rearrangement path -> K largest-displacement atoms (pipeline setup)
    traj, info, src, tgt, assign = plan_traj_auction(K, frames, seed, dev)
    matched = np.flatnonzero(assign >= 0)
    disp = np.linalg.norm(src[matched] - tgt[assign[matched]], axis=1)
    order = np.argsort(-disp)[:K]
    sel = matched[order]
    s0, e0 = src[sel], tgt[assign[sel]]
    all_pts = np.concatenate([s0, e0])
    lo, hi = all_pts.min(0), all_pts.max(0)
    span = (hi - lo).clip(min=1e-6)
    scale_px = (Ngrid - 48) / span.max()
    cen = (lo + hi) / 2
    cg = (Ngrid - 1) / 2
    s_px = (s0 - cen) * scale_px + cg
    e_px = (e0 - cen) * scale_px + cg
    Rc = 4
    cl = lambda a: np.clip(a, Rc, Ngrid - 1 - Rc)
    frames_mj = cl(MJ.interpolate_traj(s_px, e_px, frames, kind="minjerk"))

    dt = dt_ms * 1e-3
    T = (frames - 1) * dt
    px_m = px_um * 1e-6
    omega = 2.0 * np.pi * f_khz * 1e3
    a0 = a_ho(omega)
    D_px = np.linalg.norm(e_px - s_px, axis=1)
    print("=" * 78)
    print(f"strict quantum mapping on REAL rearrangement path  (seed={seed}, "
          f"K={K}, frames={frames})")
    print(f"  omega/2pi={f_khz:.3g}kHz  a0={1e9*a0:.1f}nm  dt_f={dt_ms}ms  "
          f"omega*dt_f/2pi={omega*dt/(2*np.pi):.3f} periods/frame  omegaT={omega*T:.0f}")
    print(f"  D max={D_px.max():.1f}px={1e6*D_px.max()*px_m:.2f}um  "
          f"= {D_px.max()*px_m/a0:.1f} a0")
    print("=" * 78)

    A_slm = P.gaussian_beam_torch(Ngrid, 40.0, device=dev)
    specs = [("indep5", iters, False, False),
             ("warm5", iters, True, False),
             ("carry5", iters, True, True)]
    lines = {}
    for name, it, warm, carry in specs:
        t0 = time.time()
        phi_list, U_f, I_k = run_lines_holograms(frames_mj, A_slm, Ngrid, K,
                                                 it, warm, carry, dev, bits=bits)
        ext = extract_traps(U_f, frames_mj, px_m, R=4)
        # NaN fallback -> commanded frames
        for col in ("cx", "cy"):
            bad = ~np.isfinite(ext[col])
            ext[col][bad] = frames_mj[..., 1 if col == "cy" else 0][bad]
        mp = map_line(ext, frames_mj, px_m, dt, omega, f_khz, lam=lam,
                      mass=M_RB87)
        mp["U_f"] = U_f
        mp["ext"] = ext
        mp["I_k"] = I_k
        mp["sec"] = time.time() - t0
        lines[name] = mp
        print(f"  [{name}] holograms+extract {mp['sec']:.1f}s")

    # ── table ──
    print()
    hdr = (f"{'line':<8} {'n̄_floor':>10} {'n̄_stair':>10} {'n̄_stair_ph':>12} "
           f"{'n̄_A':>10} {'n̄_C':>10} {'n̄_para':>10} {'P0_stair':>9}")
    print(hdr)
    table = []
    for name, mp in lines.items():
        row = dict(line=name,
                   n_floor=float(mp["n_floor"].mean()),
                   n_stair=float(mp["n_stair"].mean()),
                   n_stair_ph=float(mp["n_stair_ph"].mean()),
                   n_stair_inc=float(mp["n_stair_inc"].mean()),
                   n_A=float(mp["n_A"].mean()),
                   n_C=float(mp["n_C"].mean()),
                   n_para=float(mp["n_para"].mean()),
                   P0_stair=float(np.exp(-mp["n_stair"]).mean()),
                   n_stair_max=float(mp["n_stair"].max()),
                   n_A_max=float(mp["n_A"].max()),
                   n_C_max=float(mp["n_C"].max()))
        table.append(row)
        print(f"{name:<8} {row['n_floor']:10.3e} {row['n_stair']:10.3e} "
              f"{row['n_stair_ph']:12.3e} {row['n_A']:10.3e} {row['n_C']:10.3e} "
              f"{row['n_para']:10.3e} {row['P0_stair']:9.4f}")
    print()
    print("note: n̄_stair = measured-centroid staircase (the REAL SLM trajectory);")
    print("      n̄_stair_inc = per-jump incoherent sum; n̄_A = interference geometry")
    print("      (phase→position); n̄_C = phase-gradient kicks (illustrative);")
    print("      n̄_para = intensity-flicker parametric (should be ~0).")

    # ── save ──
    npz_path = f"{out_prefix}_s{seed}_K{K}_F{frames}_f{f_khz}.npz"
    save = dict(frames_mj=frames_mj, s_px=s_px, e_px=e_px,
                omega=omega, dt=dt, px_m=px_m, a0=a0, lam=lam,
                table=table)
    for name, mp in lines.items():
        for ax in ("x", "y"):
            for q in ("x0_m", "om_t", "phi", "I_rel"):
                save[f"{name}_{ax}_{q}"] = mp[ax][q]
        for q in ("n_floor", "n_stair", "n_stair_inc", "n_stair_ph", "n_A", "n_C", "n_para"):
            save[f"{name}_{q}"] = mp[q]
        save[f"{name}_U_f"] = mp["U_f"]
    np.savez_compressed(npz_path, **save)
    print(f"\nsaved: {npz_path}")

    # ── SOFT cross-check: worst atom, dominant axis (full exact ψ) ──
    soft_check = None
    if do_soft_check:
        try:
            mp = lines["carry5"]
            k = int(np.argmax(mp["n_stair"]))
            ax = "x" if mp["x"]["n_stair"][k] >= mp["y"]["n_stair"][k] else "y"
            x0_m = mp[ax]["x0_m"][:, k]
            om_tk = mp[ax]["om_t"][:, k]
            n_stair_k = mp[ax]["n_stair"][k]
            D_full = float(abs(x0_m[-1] - x0_m[0]))
            print(f"\nSOFT cross-check (scaled D): atom {k}, axis {ax}, "
                  f"D_full={1e6*D_full:.3f}um, nbar_stair={n_stair_k:.1f}")
            sc = soft_crosscheck(x0_m, om_tk, omega, dt, mass=M_RB87,
                                 scale_d=1.0/20.0)
            soft_check = dict(atom=k, axis=ax, n_stair=n_stair_k, **sc)
            print(f"  scaled-D SOFT nbar = {sc['n']:.4f}   "
                  f"analytic (same traj) = {sc['n_pred']:.4f}   "
                  f"ratio = {sc['n']/max(sc['n_pred'],1e-30):.4f}   "
                  f"norm={sc['norm']:.8f}  ({sc['sec']:.0f}s)")
            print(f"  full-D nbar (analytic coherent sum, exact) = {n_stair_k:.1f}")
            with np.load(npz_path, allow_pickle=True) as d0:
                save = dict(d0)
            save["soft_check"] = np.array([sc["n"], sc["n_pred"], n_stair_k])
            np.savez_compressed(npz_path, **save)
        except Exception as ex:
            print(f"SOFT cross-check failed: {type(ex).__name__}: {ex}")
    return lines, table, soft_check


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--frames", type=int, default=14)
    ap.add_argument("--Ngrid", type=int, default=256)
    ap.add_argument("--f_khz", type=float, default=20.0)
    ap.add_argument("--dt_ms", type=float, default=1.0)
    ap.add_argument("--px_um", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--no_soft", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bits", type=int, default=None)
    args = ap.parse_args()
    run(seed=args.seed, K=args.K, frames=args.frames, Ngrid=args.Ngrid,
        f_khz=args.f_khz, dt_ms=args.dt_ms, px_um=args.px_um,
        iters=args.iters, device=args.device, do_soft_check=not args.no_soft,
        bits=args.bits)

