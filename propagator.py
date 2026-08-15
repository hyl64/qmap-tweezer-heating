"""单一可微 FFT 传播器 + 可微光镊指标 (批量, torch, GPU)。

这个模块是整个项目的物理栈底座,被两处复用:
  1. wgs_torch.wgs_batch  —— 批量 GPU 版 WGS (生成 warmup 标签 + benchmark 基线)
  2. losses.physics_loss  —— 可微物理损失 (主训练目标)

设计要点 (与 wgs.py 的 numpy 实现对齐, 但改为批量 + 可微):
  - norm="ortho" 全程: 帕塞瓦尔 sum|F|² = sum|U_slm|², 使"焦平面总能量 = SLM 入射能量"
    固定 (与相位 φ 无关) -> 物理损失里 `1 - efficiency` 无法靠"把整体调暗"作弊,
    只能把能量搬进光镊掩码; `cv` 再把能量在各光镊间拉平 -> 损失非退化。
    (Plan agent 已对真实 wgs.py 实测: ortho 下 sum|F|² = sum|A_slm|² = 5028.059 恒定.)
  - 相位用 (cos φ, sin φ) 两通道单位向量表示, 直接构造复场
    U_slm = A_slm * (cos + i*sin), 图内绝不出现 atan2 (梯度干净, 无 2π 环绕断点).
    atan2 只在推理时用 (model.predict_phase).
  - 所有移位与 wgs.py 一致: fftshift(fft2(ifftshift(U_slm), norm=norm)).
  - std 用 unbiased=False 对齐 numpy ddof=0; mean 加 eps 防除零.
  - 变 K: batch 内所有样本 pad 到 K_max, 假光镊 (-1,-1), valid (B,K) bool 掩码;
    指标忽略 invalid 光镊.
"""
from __future__ import annotations

import torch
from torch import Tensor


# ─── 复场构造 + 传播 ───

def make_complex_field(
    A_slm: Tensor, cos_phi: Tensor, sin_phi: Tensor
) -> Tensor:
    """从 (cos φ, sin φ) 构造 SLM 面复场 U_slm = A_slm * (cos + i sin).

    A_slm: (B, N, N) 实振幅 (入射高斯, phase-only 约束 |U|=A_slm 构造性满足).
    cos_phi, sin_phi: (B, N, N) 单位向量两通道 (调用方负责末尾单位归一化,
      这里不强求单位 —— 但 |(cos,sin)|=1 时 |U|=A_slm 严格成立).
    返回 (B, N, N) complex64. 无 atan2.
    """
    return A_slm.to(torch.complex64) * (cos_phi.to(torch.complex64)
                                        + 1j * sin_phi.to(torch.complex64))


def propagate(
    A_slm: Tensor, cos_phi: Tensor, sin_phi: Tensor, norm: str = "ortho"
) -> Tensor:
    """SLM 面 -> 焦平面 (原子面) 传播 = 透镜 = FFT.

    直接从 (cos, sin) 构造 U_slm, 然后 fftshift(fft2(ifftshift(U_slm), norm)).
    返回 (B, N, N) complex64 = U_focal. 可微 (autograd 流过 FFT).
    """
    U_slm = make_complex_field(A_slm, cos_phi, sin_phi)
    # 与 wgs.py 完全一致的移位布局
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(U_slm, dim=(-2, -1)), norm=norm),
        dim=(-2, -1),
    )


def focal_intensity(
    A_slm: Tensor, cos_phi: Tensor, sin_phi: Tensor, norm: str = "ortho"
) -> Tensor:
    """焦平面强度 |U_focal|². 可微 (|z|² = z.real² + z.imag², 自身可微)."""
    U = propagate(A_slm, cos_phi, sin_phi, norm=norm)
    return U.real.pow(2) + U.imag.pow(2)


# ─── 光镊掩码 + 逐镊强度 (掩码是常数, 梯度只过强度图) ───

def spot_masks(
    N: int, spots: Tensor, radius: float, device=None
) -> Tensor:
    """每个光镊一个圆盘布尔掩码, 批量化.

    spots: (B, K, 2) 各光镊中心 (行, 列, 像素); 假光镊 (-1,-1) 由 valid 标记,
      但掩码仍生成 (后续 valid 掩码会清零其贡献, 所以掩码落在哪里无所谓).
    返回 (B, K, N, N) bool. 内存: B*K*N*N bool -> 对 B=64,K=200,N=256 ~0.8GB,
      可接受; 若 OOM 改用稀疏索引 (见 spot_intensities 的备选注释).
    """
    B, K, _ = spots.shape
    if device is None:
        device = spots.device
    coords = torch.arange(N, device=device)
    # BUGFIX (2026-08-14, b1 项目): 原代码 meshgrid(indexing="xy") 的返回顺序
    # 与命名相反 (第一个返回值是行网格), 导致掩码中心落在转置位置, 密口径
    # efficiency/uniformity_cv 全部错位。改为 indexing="ij" 并显式命名。
    YY, XX = torch.meshgrid(coords, coords, indexing="ij")  # YY=行, XX=列
    XX = XX[None, None]  # (1,1,N,N)
    YY = YY[None, None]
    spots = spots.to(device=device, dtype=torch.float32)  # (B,K,2)
    r = spots[..., 0:1]  # 行
    c = spots[..., 1:2]  # 列
    # (B,K,1,1)
    r = r[..., None]
    c = c[..., None]
    masks = ((XX - c) ** 2 + (YY - r) ** 2) <= radius ** 2
    return masks  # (B,K,N,N) bool


def spot_intensities(
    I_focal: Tensor, masks: Tensor, valid: Tensor
) -> Tensor:
    """每个光镊在其圆盘掩码内积分的强度. 可微 (掩码是常数, 梯度只过 I_focal).

    I_focal: (B, N, N). masks: (B, K, N, N) bool. valid: (B, K) bool.
    返回 (B, K) float. invalid 光镊 (valid=False) 返回 0 (后续指标会按 valid 平均).
    """
    m = masks.to(I_focal.dtype)  # 0/1
    # (B,K,N,N) * (B,1,N,N) -> sum over N,N
    per_spot = (m * I_focal[:, None, :, :]).sum(dim=(-2, -1))  # (B,K)
    per_spot = per_spot * valid.to(per_spot.dtype)
    return per_spot


# ─── 稀疏掩码 (大 N 用, 避免 B*K*N*N bool 爆显存) ───

def spot_mask_pixels(
    N: int, spots: Tensor, radius: float, device=None
) -> list[list[Tensor]]:
    """每个光镊返回掩码内像素的 (行,列) 索引张量 (1D). 稀疏, 省显存.

    spots: (B, K, 2). 返回 list[B] of list[K] of (行索引, 列索引) 两个 1D LongTensor.
    假光镊 (-1,-1) 返回空索引 (0 像素).
    """
    B, K, _ = spots.shape
    if device is None:
        device = spots.device
    spots = spots.to(device=device, dtype=torch.float32).cpu().numpy()
    R = int(radius)
    out = []
    for b in range(B):
        bl = []
        for k in range(K):
            r, c = spots[b, k]
            if r < 0 or c < 0:           # 假光镊
                idx = (torch.zeros(0,dtype=torch.long), torch.zeros(0,dtype=torch.long))
            else:
                r0 = max(0, int(r)-R); r1 = min(N, int(r)+R+1)
                c0 = max(0, int(c)-R); c1 = min(N, int(c)+R+1)
                yy, xx = torch.meshgrid(torch.arange(r0,r1), torch.arange(c0,c1), indexing="ij")
                m = ((xx - c)**2 + (yy - r)**2) <= radius**2
                # 全局坐标 (已经是绝对坐标, 因为 yy/xx 从 r0/c0 开始)
                idx = ((yy[m]).to(device), (xx[m]).to(device))
            bl.append(idx)
        out.append(bl)
    return out


def spot_intensities_sparse(
    I_focal: Tensor, mask_pixels: list[list], valid: Tensor
) -> Tensor:
    """稀疏版 spot_intensities. I_focal (B,N,N). 返回 (B,K) float, 可微."""
    B, K = len(mask_pixels), len(mask_pixels[0])
    out = torch.zeros(B, K, device=I_focal.device, dtype=I_focal.dtype)
    for b in range(B):
        for k in range(K):
            ri, ci = mask_pixels[b][k]
            if ri.numel() > 0 and valid[b, k]:
                out[b, k] = I_focal[b][ri, ci].sum()
    return out


def efficiency_sparse(
    I_focal: Tensor, mask_pixels: list[list], valid: Tensor, eps: float = 1e-12
) -> Tensor:
    """稀疏版 efficiency: 并集内能量/总能量."""
    B = I_focal.shape[0]
    num = torch.zeros(B, device=I_focal.device, dtype=I_focal.dtype)
    for b in range(B):
        seen = set()
        for k in range(len(mask_pixels[b])):
            if not valid[b, k]: continue
            ri, ci = mask_pixels[b][k]
            for ii, jj in zip(ri.tolist(), ci.tolist()):
                if (ii, jj) not in seen:
                    seen.add((ii, jj))
        if seen:
            rs = torch.tensor([x[0] for x in seen], device=I_focal.device)
            cs = torch.tensor([x[1] for x in seen], device=I_focal.device)
            num[b] = I_focal[b][rs, cs].sum()
    den = I_focal.sum(dim=(-2, -1)).clamp_min(eps)
    return num / den


def uniformity_cv_sparse(
    I_focal: Tensor, mask_pixels: list[list], valid: Tensor, eps: float = 1e-12
) -> Tensor:
    """稀疏版 uniformity_cv."""
    per = spot_intensities_sparse(I_focal, mask_pixels, valid)  # (B,K)
    v = valid.to(per.dtype)
    n = v.sum(dim=1).clamp_min(1)
    mean = per.sum(dim=1) / n
    diff = (per - mean[:, None]) * v
    var = (diff.pow(2).sum(dim=1)) / n
    std = var.clamp_min(0).sqrt()
    return std / mean.clamp_min(eps)


# ─── 可微指标 ───

def efficiency(I_focal: Tensor, masks: Tensor, valid: Tensor, eps: float = 1e-12) -> Tensor:
    """衍射效率 = 落在光镊掩码并集内的能量 / 焦平面总能量. (B,)

    物理含义 (ortho): 焦平面总能量固定 = SLM 入射能量, 所以 efficiency ∈ [0,1]
    且不能靠调暗作弊. valid 光镊的并集参与分子; 假光镊掩码不贡献.
    """
    # 并集: 任一 valid 光镊的掩码为 True 即并集
    v = valid[:, :, None, None].to(torch.bool)  # (B,K,1,1)
    union = (masks & v).any(dim=1)  # (B,N,N)
    union_e = union.to(I_focal.dtype)
    num = (I_focal * union_e).sum(dim=(-2, -1))
    den = I_focal.sum(dim=(-2, -1)).clamp_min(eps)
    return num / den


def uniformity_cv(
    I_focal: Tensor, masks: Tensor, valid: Tensor, eps: float = 1e-12
) -> Tensor:
    """各光镊强度的变异系数 CV = std/mean. (B,)

    uniformity = 1 - cv. 用 CV 而非 1-std/mean 是为了梯度稳定 (mean 在分母, 已加 eps).
    std 用 unbiased=False 对齐 numpy ddof=0. 只在 valid 光镊上统计.
    cv 越小越均匀 (理想 0). 训练最小化 cv.
    """
    per_spot = spot_intensities(I_focal, masks, valid)  # (B,K)
    v = valid.to(per_spot.dtype)  # (B,K)
    n = v.sum(dim=1).clamp_min(1)  # 每样本有效光镊数
    mean = (per_spot.sum(dim=1)) / n  # 仅 valid 贡献 (invalid 已置 0)
    # 方差: sum((x-mean)²) over valid / n
    diff = (per_spot - mean[:, None]) * v  # invalid 也置 0 (其 x=0, 但 v=0 抹掉)
    var = (diff.pow(2).sum(dim=1)) / n
    std = var.clamp_min(0).sqrt()
    cv = std / mean.clamp_min(eps)
    return cv  # (B,) 最小化目标


# ─── 入射高斯 + 目标阵列 (torch 版, 对齐 wgs.gaussian_beam / target_array) ───

def gaussian_beam_torch(
    N: int, waist_px: float, device=None, dtype=torch.float32
) -> Tensor:
    """SLM 面入射高斯振幅 (束腰 waist_px, 归一化到 1), 对齐 wgs.gaussian_beam."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    coords = torch.arange(N, device=device, dtype=dtype) - (N - 1) / 2.0
    Y, X = torch.meshgrid(coords, coords, indexing="xy")
    A = torch.exp(-(X ** 2 + Y ** 2) / (2.0 * waist_px ** 2))
    A = A / A.max()
    return torch.clamp(A, min=1e-6)


def target_array_torch(
    N: int, spots: Tensor, spot_radius_px: float = 2.0, device=None
) -> Tensor:
    """目标光镊阵列振幅 sqrt(I_target), 对齐 wgs.target_array.

    spots: (K, 2) 或 (B, K, 2). 返回 (N,N) 或 (B,N,N).
    把点目标软化成有限大小高斯斑 (相位恢复需带限目标).
    """
    if device is None:
        device = spots.device
    coords = torch.arange(N, device=device, dtype=torch.float32)
    YY, XX = torch.meshgrid(coords, coords, indexing="xy")

    if spots.dim() == 2:
        A = torch.zeros(N, N, device=device, dtype=torch.float32)
        for r, c in spots.tolist():
            A += torch.exp(-((XX - c) ** 2 + (YY - r) ** 2)
                           / (2.0 * spot_radius_px ** 2))
        peak = A.max()
        if peak > 0:
            A = A / peak
        return A
    else:
        B = spots.shape[0]
        A = torch.zeros(B, N, N, device=device, dtype=torch.float32)
        for b in range(B):
            for r, c in spots[b].tolist():
                A[b] += torch.exp(-((XX - c) ** 2 + (YY - r) ** 2)
                                  / (2.0 * spot_radius_px ** 2))
            peak = A[b].max()
            if peak > 0:
                A[b] = A[b] / peak
        return A
