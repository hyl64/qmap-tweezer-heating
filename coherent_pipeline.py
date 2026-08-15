"""coherent_pipeline: 终极目标 —— 真实重排轨迹上的相干移动 demo.

把相干移动三要素接到真实路径规划产出的重排轨迹上:
 ① min-jerk 时间插值 (替代冷管线用的线性插值)
 ② P2WGS warm-start + phase carry 逐帧全息 (帧间相位连续)
 ③ 加热核算 (atom_dynamics): 命令路径 nbar(绝热下限) vs 全息线 nbar_cl(含强度闪烁)

三层对照 (同一重排轨迹):
   indep5  每帧随机初相, 5 iter          (非相干: 相位乱跳)
   warm5   phi_init=上一帧, 5 iter        (P2WGS: 帧间相位连续)
   carry5  warm5 + trajectory_phase_target (论文 phase-aware)

加热输入: 命令 min-jerk 路径是绝热下限; P2WGS 全息的光强闪烁(I_cv)才是真实加热来源.
输出: CP(相位连续)/CI(强度连续)/eff + nbar_mean/max + P0(地态保留率).

用法:
   ./.venv/bin/python coherent_pipeline.py --K 24 --frames 20 --Ngrid 256 --seed 0
   # 更胶烈(更大位移->加热可见): --f_khz 40 --shift_scale 3
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
from auction_pipeline import plan_traj_auction, map_grid
import propagator as P, p2wgs as P2, p2wgs_metrics as PM, minjerk as MJ
import atom_dynamics as AD


def minjerk_frames(src_sel, tgt_sel, frames):
    """min-jerk 时间插值: start(K,2) -> end(K,2), frames 帧."""
    start = src_sel.astype(np.float32)
    end = tgt_sel.astype(np.float32)
    return MJ.interpolate_traj(start, end, frames, kind="minjerk")


def run_hologram_line(A_slm, frames_xy, N, Kreal, iters, warm, carry, dev):
    """逐帧 P2WGS 全息 (warm/carry), 返回 phi_list + I_k(F,Kreal) + 每帧耗时."""
    F, K, _ = frames_xy.shape
    phi_list = []
    I_list = []
    t_gen = 0.0
    # frame 0: 全收敛 (共享物理起点)
    sp0 = frames_xy[0]
    T0 = P2.trajectory_phase_target(None, A_slm, torch.from_numpy(sp0).to(dev).float(),
                                    spot_radius_px=2.0, N=N, device=dev, seed=0)
    spots0 = torch.zeros(1, max(K, 1), 2, device=dev).fill_(-1.)
    spots0[0, :K] = torch.from_numpy(sp0).to(dev)
    valid = torch.zeros(1, max(K, 1), dtype=torch.bool, device=dev)
    valid[0, :K] = True
    phi, _, _ = P2.p2wgs_batch(A_slm, T0, spots0, valid, iterations=50, gamma=0.6,
                               use_phase_weights=False, lock_mode="off", spot_radius_px=2.0,
                               seed=0, norm="ortho", device=dev)
    U = P.propagate(A_slm[None], torch.cos(phi), torch.sin(phi), norm="ortho")
    I = U.real.pow(2) + U.imag.pow(2)
    mp = P.spot_mask_pixels(N, spots0, 2.0, device=dev)
    I_k = P.spot_intensities_sparse(I, mp, valid)[0, :K].cpu().numpy()
    phi_list.append(phi); I_list.append(I_k)

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
        t0 = time.perf_counter()
        phi, _, _ = P2.p2wgs_batch(A_slm, T, spots_f, valid, iterations=iters, gamma=0.6,
                                   use_phase_weights=False, lock_mode="off", spot_radius_px=2.0,
                                   seed=0, norm="ortho", device=dev, phi_init=init)
        torch.cuda.synchronize(); t_gen += time.perf_counter() - t0
        U = P.propagate(A_slm[None], torch.cos(phi), torch.sin(phi), norm="ortho")
        I = U.real.pow(2) + U.imag.pow(2)
        mp = P.spot_mask_pixels(N, spots_f, 2.0, device=dev)
        I_k = P.spot_intensities_sparse(I, mp, valid)[0, :K].cpu().numpy()
        phi_list.append(phi); I_list.append(I_k)
    return dict(phi=phi_list, I=np.stack(I_list, 0),
                ms_per_frame=1e3 * t_gen / max(F - 1, 1))


def continuity(line, frames_xy, A_slm, N, dev):
    F, K, _ = frames_xy.shape
    CP, CI = [], []
    for f in range(F - 1):
        sp_c = frames_xy[f + 1]
        sp_t = torch.zeros(1, max(K, 1), 2, device=dev).fill_(-1.)
        sp_t[0, :K] = torch.from_numpy(sp_c).to(dev)
        vd = torch.zeros(1, max(K, 1), dtype=torch.bool, device=dev)
        vd[0, :K] = True
        cp = PM.continuity_CP(line["phi"][f], line["phi"][f + 1], sp_t, vd, A_slm, N).item()
        U_p = P.propagate(A_slm[None], torch.cos(line["phi"][f]), torch.sin(line["phi"][f]), norm="ortho")
        U_c = P.propagate(A_slm[None], torch.cos(line["phi"][f + 1]), torch.sin(line["phi"][f + 1]), norm="ortho")
        I_p = U_p.real.pow(2) + U_p.imag.pow(2)
        I_c = U_c.real.pow(2) + U_c.imag.pow(2)
        ci = PM.continuity_CI(I_p, I_c, sp_t, vd, N).item()
        CP.append(cp); CI.append(ci)
    return float(np.mean(CP)), float(np.mean(CI))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--Ngrid", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--dt_ms", type=float, default=1.0)
    ap.add_argument("--px_um", type=float, default=0.5)
    ap.add_argument("--f_khz", type=float, default=40.0)
    ap.add_argument("--shift_scale", type=float, default=1.0,
                    help="放大重排位移(像素), 让加热可见")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    # 真实路径规划 -> 最大位移原子
    traj, info, src, tgt, assign = plan_traj_auction(args.K, args.frames, args.seed, dev)
    # 直接取最大位移原子的 start/end (lattice 坐标)
    matched = np.flatnonzero(assign >= 0)
    disp = np.linalg.norm(src[matched] - tgt[assign[matched]], axis=1)
    order = np.argsort(-disp)[:args.K]
    sel = matched[order]
    s0 = src[sel]; e0 = tgt[assign[sel]]

    # 坐标映射到 Ngrid 焦平面 (含位移放大)
    all_pts = np.concatenate([s0, e0])
    lo, hi = all_pts.min(0), all_pts.max(0)
    span = (hi - lo).clip(min=1e-6)
    scale_px = (args.Ngrid - 48) / span.max() * args.shift_scale
    cen = (lo + hi) / 2
    cg = (args.Ngrid - 1) / 2
    s_px = (s0 - cen) * scale_px + cg
    e_px = (e0 - cen) * scale_px + cg
    D_px = np.linalg.norm(e_px - s_px, axis=1)

    # 时间参数化: linear vs minjerk (clip 到焦平面有效区, 防 spot 掩码越界)
    R = 4
    cl = lambda a: np.clip(a, R, args.Ngrid - 1 - R)
    frames_lin = cl(MJ.interpolate_traj(s_px, e_px, args.frames, kind="linear"))
    frames_mj = cl(MJ.interpolate_traj(s_px, e_px, args.frames, kind="minjerk"))

    # 物理参数
    dt = args.dt_ms * 1e-3
    T = (args.frames - 1) * dt
    px_m = args.px_um * 1e-6
    omega0 = 2.0 * np.pi * args.f_khz * 1e3
    a0 = AD.a_ho(omega0)
    D_m = float(np.mean(D_px * px_m))
    print("=" * 74)
    print("coherent transport on REAL rearrangement path (终极目标)")
    print(f"  原子 {args.K}  frames={args.frames}  dt={args.dt_ms}ms  T={1e3*T:.0f}ms")
    print(f"  ω/2π={args.f_khz}kHz  a_ho={1e9*a0:.1f}nm  ⟨D⟩={1e6*D_m:.2f}µm  "
          f"ωT={omega0*T:.0f}  D/a_ho={D_m/a0:.1f}")
    print(f"  位移 max={D_px.max():.1f}px  mean={D_px.mean():.1f}px  (scale={args.shift_scale})")
    print("=" * 74)

    # ① 命令路径加热 (linear vs minjerk, 连续+阶梯)
    print("\n--- 1 commanded-path heating (绝热下限, 无全息误差) ---")
    print(f"{'kind':<10} {'n_mean':>10} {'n_max':>10} {'P0':>8}")
    for name, fr in (("linear", frames_lin), ("minjerk", frames_mj)):
        n_k, n_m = AD.nbar_analytic(fr[0, :, 1] * px_m, fr[-1, :, 1] * px_m, T, omega=omega0, kind=name)
        n_d, m_d = AD.nbar_analytic_frames(fr[:, :, 1] * px_m, dt, omega=omega0)
        print(f"{name:<10} {n_m:10.3e} {n_k.max():10.3e} {np.exp(-n_k).mean():8.4f}")
        print(f"{name+'+disc':<10} {m_d:10.3e} {n_d.max():10.3e} {np.exp(-n_d).mean():8.4f}")

    # ② 全息线 (min-jerk 帧)
    A_slm = P.gaussian_beam_torch(args.Ngrid, 40.0, device=dev)
    specs = [("indep5", args.iters, False, False),
             ("warm5",  args.iters, True,  False),
             ("carry5", args.iters, True,  True)]
    print("\n--- 2/3 hologram lines (min-jerk frames) + heating ---")
    print(f"{'line':<8} {'CP':>7} {'CI':>7} {'eff':>7} {'n̄_cl':>9} {'n̄_cl_max':>10} {'P0_cl':>7} {'I_cv':>7}")
    lines = {}
    for name, iters, warm, carry in specs:
        lines[name] = run_hologram_line(A_slm, frames_mj, args.Ngrid, args.K, iters, warm, carry, dev)
        CP, CI = continuity(lines[name], frames_mj, A_slm, args.Ngrid, dev)
        lines[name]["cp"] = CP; lines[name]["ci"] = CI
        eff = None
        I_k = lines[name]["I"]
        # 强度闪烁 (全息误差 -> 经典加热)
        I_cv = float((I_k.std(axis=0) / np.maximum(I_k.mean(axis=0), 1e-18)).mean())
        x0_m = frames_mj[:, :, 1] * px_m
        I_rel = I_k / np.maximum(I_k.mean(axis=0, keepdims=True), 1e-18)
        n_cl, n_cl_m = AD.nbar_classical(x0_m, I_rel, dt, omega0=omega0)
        print(f"{name:<8} {CP:7.4f} {CI:7.4f} {'--':>7} {n_cl_m:9.3e} {n_cl.max():10.3e} "
              f"{np.exp(-n_cl).mean():7.4f} {I_cv:7.4f}")

    print("\n--- heat-up summary ---")
    n0 = lines["indep5"]
    for nm in ("warm5", "carry5"):
        ln = lines[nm]
        I0 = n0["I"]; Il = ln["I"]
        # 全息光强闪烁 (经典加热驱动, 论文 CI 口径)
        cv0 = float((I0.std(axis=0)/np.maximum(I0.mean(axis=0),1e-18)).mean())
        cvl = float((Il.std(axis=0)/np.maximum(Il.mean(axis=0),1e-18)).mean())
        print(f"  {nm:<7} vs indep5: 强度闪烁 I_cv {cv0:.4f} -> {cvl:.4f}")
    print("done")

if __name__ == "__main__":
    main()
