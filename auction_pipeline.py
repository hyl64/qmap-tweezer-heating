"""auction_pipeline: 纯算法(代替 GNN)路径规划 → P2WGS 相干全息, 端到端.

路径规划用 match 包的 auction/greedy+tail 硬解码 (零训练, 确定性), 替代 GNN+Hungarian:
    MatchCands.build_frame -> scores=-d2 -> decode -> match_to_assign -> assign(S,)
然后复用 joint_pipeline 的轨迹插值 + p2wgs(可用 CUDA graph 快速后端)生成全息.

用法:
    ./.venv/bin/python auction_pipeline.py --K 48 --frames 20 --Ngrid 256 --fast
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from match_cands import MatchCands
from match_decode import greedy_decode
from match_numba_tail import bottleneck_all
from gnn_data import source_array, target_lattice
import propagator as P
import p2wgs as P2
import p2wgs_metrics as PM


def plan_traj_auction(K_cap, frames, seed, dev):
    """纯算法路径规划: MatchCands + greedy/tail decode -> assign -> 多帧轨迹."""
    mc = MatchCands(side_src=142, side_tgt=101, k_pre=128, C=32, device=dev)
    rng = np.random.default_rng(seed)
    src = source_array(142, 0.55, 1.0, rng=rng)            # (S,2) 捕获原子
    # 目标晶格 (与 MatchCands 同一几何: 间距 = 1.0*(142-1)/(101-1), 中心对齐)
    tgt_spacing = 1.0 * (142 - 1) / (101 - 1)
    tgt = target_lattice(101, spacing=tgt_spacing)
    src_c = (142 - 1) * 1.0 / 2
    tgt_c = (101 - 1) * tgt_spacing / 2
    tgt = tgt + (src_c - tgt_c)
    S, T = len(src), len(tgt)

    # build_frame + 打分(负平方距离) + 硬解码 (纯算法, 零训练)
    fr = mc.build_frame(src)
    scores = -fr.cand_d2
    t0 = time.perf_counter()
    match, owner, rounds = greedy_decode(scores, fr, fr.S)  # GPU 贪心 bulk
    torch.cuda.synchronize()
    n_left = int((match < 0).sum().item())
    # ---- 生产尾拍: numba 代际版 min-max BFS, 替代纯 Python compact_cpu_tail ----
    ca = fr.cand_atom.detach().cpu().numpy().astype(np.int64)
    d2cpu = fr.cand_d2.detach().cpu().numpy()
    mto = match.detach().cpu().numpy().astype(np.int64).copy()
    ow = np.full(fr.S, -1, dtype=np.int64)
    ok = mto >= 0
    ow[mto[ok]] = np.flatnonzero(ok)
    MH = 64
    n_ok, n_fail = bottleneck_all(ca, d2cpu, mto, ow, MH)
    torch.cuda.synchronize()
    t_path = (time.perf_counter() - t0) * 1000
    # mto(T,) -> assign(S,)
    assign = np.full(fr.S, -1, dtype=np.int64)
    mm = mto >= 0
    assign[mto[mm]] = np.flatnonzero(mm)
    cov = (assign >= 0).mean()
    info_tail = dict(n_left=n_left, n_ok=n_ok, n_fail=n_fail, rounds=rounds)

    # 轨迹: 选位移最大的 K_cap 个匹配原子
    matched = np.flatnonzero(assign >= 0)
    disp = np.linalg.norm(src[matched] - tgt[assign[matched]], axis=1)
    order = np.argsort(-disp)[:K_cap]
    sel = matched[order]
    s = src[sel]; t = tgt[assign[sel]]
    ft = np.linspace(0, 1, frames)[:, None, None]
    traj = s[None] + (t - s)[None] * ft
    info = dict(S=S, T=T, cov=cov, t_path=t_path,
                disp_mean=float(disp.mean()), disp_max=float(disp.max()), N=len(sel),
                n_left=n_left, n_ok=n_ok, n_fail=n_fail, rounds=rounds)
    return traj, info, src, tgt, assign


def map_grid(traj, N_grid, pad=24):
    mg = traj.reshape(-1, 2)
    lo, hi = mg.min(0), mg.max(0)
    span = (hi - lo).clip(min=1e-6)
    scale = (N_grid - 2 * pad) / span.max()
    cen = (lo + hi) / 2
    cg = (N_grid - 1) / 2
    return (traj - cen) * scale + cg, float(scale)


def run_p2wgs_fast(traj_grid, N_grid, dev, iters=5, fast=True):
    """逐帧 P2WGS (warm-start), 默认 CUDA graph 快速后端."""
    frames, N, _ = traj_grid.shape
    A_slm = P.gaussian_beam_torch(N_grid, 40.0, device=dev)
    if fast:
        from p2wgs_fast import p2wgs_batch_fast_cached
    phi_prev = None
    times = []
    for f in range(frames):
        sp_t = torch.from_numpy(traj_grid[f]).to(dev).float()
        T0 = (P2.trajectory_phase_target(phi_prev[0], A_slm, sp_t, 2.0, N_grid, dev)
              if (phi_prev is not None) else
              P2._gaussian_complex_target(N_grid, sp_t, torch.zeros(N, device=dev), 2.0, device=dev))
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        if fast:
            phi, _, _ = p2wgs_batch_fast_cached(A_slm, T0, sp_t, torch.ones(N, dtype=torch.bool, device=dev),
                                                iterations=iters, gamma=0.6, spot_radius_px=2.0,
                                                seed=0, phi_init=phi_prev, device=dev, return_amp=False)
        else:
            phi, _, _ = P2.p2wgs_batch(A_slm, T0, sp_t[None],
                                       torch.ones(1, N, dtype=torch.bool, device=dev),
                                       iterations=iters, gamma=0.6, use_phase_weights=False,
                                       lock_mode="off", spot_radius_px=2.0, seed=0, norm="ortho",
                                       device=dev, phi_init=phi_prev)
        e1.record(); torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
        phi_prev = phi
    return float(np.median(times[1:])), float(np.sum(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--Ngrid", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", type=str, default="fig_auction_pipeline.png")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"纯算法(代GNN)路径规划 -> P2WGS 端到端 | K={args.K} frames={args.frames} "
          f"Ngrid={args.Ngrid} iters={args.iters} fast={args.fast} | {dev}")

    traj, info, src, tgt, assign = plan_traj_auction(args.K, args.frames, args.seed, dev)
    tg, scale = map_grid(traj, args.Ngrid)
    print(f"[路径规划] S={info['S']} T={info['T']} assign覆盖={info['cov']*100:.1f}% "
          f"greedy({info['rounds']}轮,残{info['n_left']})+numba尾拍({info['n_ok']}补齐,{info['n_fail']}fail) "
          f"解码共={info['t_path']:.1f}ms")
    print(f"[轨迹] N_active={info['N']} 位移 mean={info['disp_mean']:.3f} max={info['disp_max']:.3f} "
          f"(lattice) -> 实际 max={info['disp_max']*scale:.1f}px")

    t_med, t_sum = run_p2wgs_fast(tg, args.Ngrid, dev, iters=args.iters, fast=args.fast)
    print(f"[P2WGS] 单帧 median={t_med:.3f}ms  全部{args.frames}帧={t_sum:.2f}ms  "
          f"(fast={'CUDA graph' if args.fast else 'reference'})")
    print(f"[E2E]   路径规划({info['t_path']:.0f}ms) + P2WGS({t_sum:.0f}ms) = "
          f"{info['t_path']+t_sum:.0f} ms  ({args.frames}帧重排)")
    print("DONE")


if __name__ == "__main__":
    main()
