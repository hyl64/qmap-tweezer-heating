"""P2WGS 帧间连续性 + 光束质量物理指标 (批量, torch, GPU).

分两类:
  (A) 帧间连续性 (P2WGS 的核心卖点, 跨帧测):
      CI   强度连续 (论文口径, 5×5 窗)
      CP   相位连续 (论文口径, 最近像素): mean|wrap(Δφ)|/2π ∈ [0,0.5], 0=完美记忆
      ∇φ   阱芯相位梯度 (复数差分法): 原子感知的"运动项"
      R    相位推进一致性: |wrap(Δφ_meas − ∇φ·v_trap)|/2π
      Ripple 阱内相位纹波: 5×5 窗内相位 std

  (B) 单帧光束质量 (仿真校验光强用, 单帧测):
      eff  衍射效率 (掩码并集能量/总能量, ortho 帕塞瓦尔定总能量)
      uni  均匀性 (1 − std/mean, 仅 valid)
      waist 束腰 1/e² 直径 (二阶矩: sqrt(Σ I·r²/Σ I)·2, 复用 benchmark.py 口径)
      depth 阱深 (阱芯峰值强度 / 背景中位数, 原子束缚能的相对量)
      crosstalk 阱间串扰 (非目标镊位置的能量泄漏 / 阱内能量)

口径约定 (与 p2wgs.py docstring 一致):
  - spots/target/mask/权重基 全在同一帧新位置 (旧位置的 bug 已在 _diag_shift_scan_fixed.py 复现并修正).
  - 相位用最近整数像素采样 (与 eval_paper._phase_err 同口径), 不受相邻镊高斯窗干涉污染.
  - wrap 用 atan2(sin,cos), autograd 无 2π 断点 (本模块 inference-only, 但保持友好).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

import propagator as P


# ─── 基本工具 ───

def angle_wrap(d: Tensor) -> Tensor:
    """把相位角 wrap 到 [-π, π], atan2(sin,cos) 形式 (无 2π 环绕去破断)."""
    return torch.atan2(torch.sin(d), torch.cos(d))


def _nearest_pixel_complex(field: Tensor, spots: Tensor, valid: Tensor, N: int) -> Tensor:
    """每个镊子中心的最近整数像素复场采样 → (B,K) complex.

    field (B,N,N) complex. spots (B,K,2) [行,列]. 与 eval_paper._phase_err 同口径
    (round 到最近像素, clamp 边界). invalid 镊子返回 0.
    """
    r = spots[..., 0].round().long().clamp(0, N - 1)
    c = spots[..., 1].round().long().clamp(0, N - 1)
    flat = (r * N + c)                                            # (B,K)
    g = field.reshape(field.shape[0], -1).gather(1, flat)         # (B,K) complex
    return g * valid.to(g.dtype).to(torch.complex64)


# ─── (A) 帧间连续性指标 ───

def continuity_CP(phi_prev: Tensor, phi_cur: Tensor, spots_cur: Tensor,
                  valid: Tensor, A_slm: Tensor, N: int, norm: str = "ortho") -> Tensor:
    """相位连续 CP = mean_k |wrap(φ_k(n+1) − φ_k(n))| / (2π). (B,) ∈ [0, 0.5].

    φ_k = 前向传播后在镊子中心最近像素的复场角 (焦面相位, 不是 SLM 整图相位 ——
    原子感知的是焦面相位). spots_cur 是 frame n+1 的新镊位置, 两帧都在此位置采样
    (位置一致, 只相位变). 0 = 完美相位记忆; 0.5 = 完全随机.
    """
    B = A_slm.shape[0]
    U0 = P.propagate(A_slm, torch.cos(phi_prev), torch.sin(phi_prev), norm=norm)
    U1 = P.propagate(A_slm, torch.cos(phi_cur), torch.sin(phi_cur), norm=norm)
    g0 = _nearest_pixel_complex(U0, spots_cur, valid, N)         # (B,K)
    g1 = _nearest_pixel_complex(U1, spots_cur, valid, N)
    d = angle_wrap(g1.angle() - g0.angle())                      # (B,K)
    v = valid.to(d.dtype)
    n = v.sum(dim=1).clamp_min(1)
    return (d.abs() * v).sum(dim=1) / n / (2 * np.pi)             # (B,)


def continuity_CI(I_prev: Tensor, I_cur: Tensor, spots: Tensor,
                 valid: Tensor, N: int, R: int = 2) -> Tensor:
    """强度连续 CI = mean_k [ Σ_{j∈5×5} |I_j(n+1)−I_j(n)| / (½Σ(I+I_prev) + ε) ]. (B,) ∈[0,2].

    论文口径 (5×5 窗=2R+1, R=2). spots 取最近整数像素中心, 越界裁切. 0 = 强度完全不变.
    I_prev, I_cur (B,N,N); spots (B,K,2); valid (B,K).
    """
    B, K, _ = spots.shape
    spots_np = spots.cpu().numpy()
    valid_np = valid.cpu().numpy()
    out = torch.zeros(B, device=I_prev.device, dtype=I_prev.dtype)
    for b in range(B):
        acc = 0.0; cnt = 0
        for k in range(K):
            if not valid_np[b, k]:
                continue
            r = int(round(float(spots_np[b, k, 0])))
            c = int(round(float(spots_np[b, k, 1])))
            r0, r1 = max(0, r - R), min(N, r + R + 1)
            c0, c1 = max(0, c - R), min(N, c + R + 1)
            w0 = I_prev[b, r0:r1, c0:c1]
            w1 = I_cur[b, r0:r1, c0:c1]
            num = (w1 - w0).abs().sum()
            den = (0.5 * (w0 + w1)).sum().clamp_min(1e-12)
            acc += float((num / den).item()); cnt += 1
        out[b] = acc / max(cnt, 1)
    return out


def trap_phase_gradient(U_focal: Tensor, spots: Tensor, valid: Tensor,
                        N: int, direction: str = "row") -> Tensor:
    """阱芯相位梯度 ∇φ (复数差分法, rad/px). direction: 'row' 或 'col'. (B,K).

    ∇φ_k = arg( Σ_j conj(U(j)) · U(j+δ) ), j 遍历镊子 5×5 窗, δ 为单像素位移 (行/列).
    这是"波前斜率"的复数估计 (无 atan2 断点, 适合梯度). 原子阱内相位斜率 = 感知到的
    等效"势阱倾斜", 与阱运动方向应正相关 (R 指标用).
    """
    B, K, _ = spots.shape
    spots_np = spots.cpu().numpy()
    valid_np = valid.cpu().numpy()
    out = torch.zeros(B, K, device=U_focal.device, dtype=torch.float32)
    R = 2
    for b in range(B):
        for k in range(K):
            if not valid_np[b, k]:
                continue
            r = int(round(float(spots_np[b, k, 0])))
            c = int(round(float(spots_np[b, k, 1])))
            r0, r1 = max(0, r - R), min(N, r + R + 1)
            c0, c1 = max(0, c - R), min(N, c + R + 1)
            win = U_focal[b, r0:r1, c0:c1]                         # (h,w) complex
            if direction == "row" and win.shape[0] > 1:
                p = win[:-1, :]; q = win[1:, :]
            elif direction == "col" and win.shape[1] > 1:
                p = win[:, :-1]; q = win[:, 1:]
            else:
                continue
            grad = (p.conj() * q).sum()
            out[b, k] = grad.angle()
    return out


def advance_consistency_R(phi_prev: Tensor, phi_cur: Tensor, spots_prev: Tensor,
                          spots_cur: Tensor, valid: Tensor, A_slm: Tensor,
                          N: int, norm: str = "ortho") -> Tensor:
    """相位推进一致性 R = mean_k |wrap(Δφ_meas − ∇φ·v_trap)| / (2π). (B,) ∈ [0, 0.5].

    Δφ_meas = 实测相邻帧阱芯相位差 (在 frame n+1 新位置).
    ∇φ·v_trap = 预测: 阱芯相位梯度 (frame n, row 方向) × 镊子位移 (px).
    若波前结构正确 (相位随阱运动平滑推进), R→0. 0.5 = 完全不匹配.
    """
    B = A_slm.shape[0]
    U0 = P.propagate(A_slm, torch.cos(phi_prev), torch.sin(phi_prev), norm=norm)
    U1 = P.propagate(A_slm, torch.cos(phi_cur), torch.sin(phi_cur), norm=norm)
    g0 = _nearest_pixel_complex(U0, spots_cur, valid, N)          # 在新位置测 frame n 相位
    g1 = _nearest_pixel_complex(U1, spots_cur, valid, N)          # 在新位置测 frame n+1 相位
    dphi_meas = angle_wrap(g1.angle() - g0.angle())              # (B,K) 实测 Δφ
    # 预测项: ∇φ(frame n, row) × v_trap(行位移)
    grad = trap_phase_gradient(U0, spots_prev, valid, N, "row")  # (B,K) rad/px
    v_trap = (spots_cur[..., 0] - spots_prev[..., 0])             # (B,K) 行位移 px
    pred = grad * v_trap                                          # (B,K) rad
    R = angle_wrap(dphi_meas - pred).abs()                        # (B,K)
    v = valid.to(R.dtype)
    n = v.sum(dim=1).clamp_min(1)
    return (R * v).sum(dim=1) / n / (2 * np.pi)                   # (B,)


def phase_ripple(U_focal: Tensor, spots: Tensor, valid: Tensor, N: int,
                 R: int = 2) -> Tensor:
    """阱内相位纹波 Ripple = sqrt(mean(δφ²)), δφ = wrap(φ_j − φ_center). (B,) rad.

    5×5 窗内相位起伏的 RMS, 衡量镊子内部相位是否平滑 (光镊质量; 纹波大 → 阱畸变).
    """
    B, K, _ = spots.shape
    spots_np = spots.cpu().numpy()
    valid_np = valid.cpu().numpy()
    out = torch.zeros(B, device=U_focal.device, dtype=torch.float32)
    for b in range(B):
        acc = 0.0; cnt = 0
        for k in range(K):
            if not valid_np[b, k]:
                continue
            r = int(round(float(spots_np[b, k, 0])))
            c = int(round(float(spots_np[b, k, 1])))
            r0, r1 = max(0, r - R), min(N, r + R + 1)
            c0, c1 = max(0, c - R), min(N, c + R + 1)
            win = U_focal[b, r0:r1, c0:c1]
            center = win[win.shape[0] // 2, win.shape[1] // 2] if win.numel() > 0 else win
            d = angle_wrap(win.angle() - center.angle())
            acc += float((d.pow(2).mean()).sqrt().item()); cnt += 1
        out[b] = acc / max(cnt, 1)
    return out


# ─── (B) 单帧光束质量 ───

def beam_quality(I_focal: Tensor, U_focal: Tensor, spots: Tensor,
                 valid: Tensor, mask_pixels, N: int, spot_radius_px: float = 2.0) -> dict:
    """单帧光束质量: eff / uni / waist / depth / crosstalk + Ripple.

    I_focal, U_focal (B,N,N). spots (B,K,2). mask_pixels (稀疏掩码, 来自 spot_mask_pixels).
    返回 dict, 每项 (B,) cpu tensor 或 float (waist 是 (B,) 平均后的标量).
    """
    B = I_focal.shape[0]
    spot_I = P.spot_intensities_sparse(I_focal, mask_pixels, valid)          # (B,K)
    v = valid.to(I_focal.dtype)
    n = v.sum(dim=1).clamp_min(1)
    mean_I = (spot_I.sum(dim=1)) / n                                          # (B,)
    diff = (spot_I - mean_I[:, None]) * v
    var = (diff.pow(2).sum(dim=1)) / n
    uni = 1.0 - var.clamp_min(0).sqrt() / mean_I.clamp_min(1e-12)             # (B,)
    eff = P.efficiency_sparse(I_focal, mask_pixels, valid)                    # (B,)

    # 束腰 1/e² 直径 (二阶矩, 整个掩码并集): w = 2·sqrt(Σ I·r²/Σ I)
    coords = torch.arange(N, device=I_focal.device, dtype=torch.float32)
    YY, XX = torch.meshgrid(coords, coords, indexing="ij")
    waist = torch.zeros(B, device=I_focal.device, dtype=torch.float32)
    for b in range(B):
        seen = set()
        for k in range(len(mask_pixels[b])):
            if not valid[b, k]:
                continue
            ri, ci = mask_pixels[b][k]
            for ii, jj in zip(ri.tolist(), ci.tolist()):
                seen.add((ii, jj))
        if not seen:
            continue
        rs = torch.tensor([x[0] for x in seen], device=I_focal.device)
        cs = torch.tensor([x[1] for x in seen], device=I_focal.device)
        Ivals = I_focal[b][rs, cs]
        w = Ivals.sum().clamp_min(1e-12)
        cy = (Ivals * rs.to(I_focal.dtype)).sum() / w
        cx = (Ivals * cs.to(I_focal.dtype)).sum() / w
        r2 = (rs.to(I_focal.dtype) - cy) ** 2 + (cs.to(I_focal.dtype) - cx) ** 2
        w0 = (Ivals * r2).sum() / w
        waist[b] = (2.0 * w0.clamp_min(0).sqrt()) if w0 > 0 else 0.0

    # 阱深 = 阱芯峰值强度 / 背景中位数 (阱内最强像素 / 焦面非阱区中位数)
    depth = torch.zeros(B, device=I_focal.device, dtype=torch.float32)
    for b in range(B):
        peak = I_focal[b].amax()
        # 背景中位数 (非零区): 用全图 median 近似 (阱区面积小, median ≈ 背景)
        bg = torch.median(I_focal[b].flatten().topk(max(1, I_focal[b].numel() // 2)).values)
        depth[b] = peak / bg.clamp_min(1e-12)

    ripple = phase_ripple(U_focal, spots, valid, N, R=int(max(2, np.ceil(spot_radius_px))))

    return {
        "efficiency": eff.cpu(),
        "uniformity": uni.cpu(),
        "waist_px": waist.cpu(),       # 1/e² 直径 (px)
        "depth_ratio": depth.cpu(),    # 阱深/背景
        "ripple_rad": ripple.cpu(),    # 阱内相位纹波 RMS
    }


# ─── 折叠: 整条帧序列的连续性 ───

def compute_frame_metrics(frames_phi: list, frames_spots: list, frames_valid: list,
                          A_slm: Tensor, N: int, norm: str = "ortho") -> dict:
    """整条帧序列的连续性指标. 返回 dict, 每项 list 长度=帧数-1 (相邻帧对).

    frames_phi: [phi_0, phi_1, ...] 每个 (B,N,N). frames_spots: [spots_0, ...] 每个 (B,K,2).
    逐相邻帧对算 CP / CI / R; ∇φ/Ripple 是单帧量, 附在第 1..N-1 帧上.
    """
    A = A_slm if A_slm.dim() == 3 else A_slm[None]
    CP, CI, R = [], [], []
    for i in range(len(frames_phi) - 1):
        phi_p, phi_c = frames_phi[i], frames_phi[i + 1]
        sp_p, sp_c = frames_spots[i], frames_spots[i + 1]
        vd = frames_valid[i]
        CP.append(continuity_CP(phi_p, phi_c, sp_c, vd, A, N, norm))
        # CI 需要 I (强度), 这里现算
        I_p = (P.propagate(A, torch.cos(phi_p), torch.sin(phi_p), norm=norm)).abs().pow(2)
        I_c = (P.propagate(A, torch.cos(phi_c), torch.sin(phi_c), norm=norm)).abs().pow(2)
        CI.append(continuity_CI(I_p[0], I_c[0], sp_c, vd, N))
        R.append(advance_consistency_R(phi_p, phi_c, sp_p, sp_c, vd, A, N, norm))
    return {"CP": CP, "CI": CI, "R": R}
