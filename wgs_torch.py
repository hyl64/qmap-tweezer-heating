"""批量 GPU 版 Weighted Gerchberg-Saxton (WGS), 直接移植 wgs.wgs, 复用 propagator。

用于:
  1. 生成 warmup 监督标签 (固定 seed=0, 每个目标一个一致标签, 降低非唯一性方差)
  2. benchmark 基线 (NN vs WGS-torch 同硬件公平对比)

物理栈与 numpy wgs.wgs 完全一致 (ortho FFT, 相同移位布局, 相同权重更新),
唯一区别: 批量化 + torch.inference_mode + GPU. 不写两套 FFT.

注意: 这里相位用 raw phi (弧度) 表示 (内部用 exp(1j*phi) 构造复场),
因为是推理/标签生成, 不需要梯度; propagator 的 (cos,sin) 接口给训练用.
返回 phi (B,N,N) 弧度, 训练时再 cos(phi)/sin(phi) 喂物理损失或当标签.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

import propagator as P


def _init_phi(B: int, N: int, seed: int, device) -> Tensor:
    """固定种子的随机初相 (B, N, N). 种子固定 -> 同一目标 -> 一致标签 (降低非唯一性).

    关键: 所有 B 个样本必须拿到**同一个** seed-0 初相块, 不能是同一 stream 的不同块,
    否则 batch=2 与单独跑 batch=1 的初相不同 -> WGS 输出不同 (违反"批处理与原始算法
    逐位一致"). 故只生成一块 (N,N) 再 broadcast 到 B.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    phi0 = torch.rand(1, N, N, generator=g, device="cpu") * (2 * torch.pi)
    return phi0.expand(B, N, N).to(device=device)



def wgs_batch(
    A_slm: Tensor,              # (B, N, N) 入射高斯振幅
    A_target: Tensor,           # (B, N, N) 目标光镊阵列振幅 sqrt(I_target)
    spots: Tensor,              # (B, K, 2) 光镊中心 (行,列,像素); 假光镊 (-1,-1)
    valid: Tensor,              # (B, K) bool, 哪些是真实光镊
    iterations: int = 50,
    gamma: float = 0.5,
    spot_radius_px: float = 2.0,
    seed: int = 0,
    norm: str = "ortho",
    device: str | None = None,
    phi_init: Tensor | None = None,   # (B,N,N) 初始相位 (warm-start, 帧间连续用)
    use_sparse: bool = True,          # True=稀疏基(scatter_add,快); False=稠密N²重建(对照)
) -> tuple[Tensor, Tensor, dict]:
    """批量 WGS. 返回 (phi (B,N,N) 弧度, A_focal (B,N,N) 振幅, history).

    权重更新 w_k <- w_k * (mean_I / I_k)^gamma, 把能量往暗光镊挪, 拉平强度.
    phi_init: 给初相 (如上一帧 WGS 结果), 使帧间相位连续 (重排轨迹标签用).
    use_sparse: 权重场实现. 默认稀疏 (快 ~300x); False 走稠密 _weight_field (对照,
    用于量化"优化后 vs 原始"一致性, 两者共用同一循环骨架只差权重场构造).
    """
    B, N, _ = A_slm.shape
    if device is None:
        device = A_slm.device
    else:
        A_slm = A_slm.to(device); A_target = A_target.to(device)
        spots = spots.to(device); valid = valid.to(device)

    K = spots.shape[1]
    if phi_init is None:
        phi = _init_phi(B, N, seed, device)             # (B,N,N)
    else:
        phi = phi_init.to(device=device, dtype=torch.float32)  # warm-start
    weights = torch.ones(B, K, device=device)        # (B,K)
    # 稀疏掩码 (大 N 不爆显存; 小 N 也兼容)
    mask_pixels = P.spot_mask_pixels(N, spots, spot_radius_px, device=device)
    # 镊子坐标 run 内不变 -> 预计算高斯基一次 (300x 加速 vs 每 iter 重建 N²)
    flat_idx, basis, pad_msk = _weight_basis(N, spots, spot_radius_px, device=device)

    history = {"uniformity": [], "efficiency": []}

    with torch.inference_mode():
        for _ in range(iterations):
            # SLM 面 -> 焦平面 (cos, sin) 喂 propagator.propagate, 一致物理栈
            cos_p = torch.cos(phi); sin_p = torch.sin(phi)
            U_focal = P.propagate(A_slm, cos_p, sin_p, norm=norm)  # (B,N,N) complex
            I_focal = U_focal.real.pow(2) + U_focal.imag.pow(2)    # |U|²
            A_focal = I_focal.sqrt()

            # 逐光镊测当前强度 + 算指标 (稀疏掩码, 与 numpy 对齐)
            spot_I = P.spot_intensities_sparse(I_focal, mask_pixels, valid)  # (B,K)
            v = valid.to(torch.float32)
            n = v.sum(dim=1).clamp_min(1)
            mean_I_b = (spot_I.sum(dim=1)) / n                    # (B,) 每样本 valid 均值
            # 均匀性 1 - std/mean (ddof=0, 仅 valid)
            diff = (spot_I - mean_I_b[:, None]) * v
            var = (diff.pow(2).sum(dim=1)) / n
            std = var.clamp_min(0).sqrt()
            uni = 1.0 - std / mean_I_b.clamp_min(1e-12)            # (B,)
            eff = P.efficiency_sparse(I_focal, mask_pixels, valid)  # (B,)
            history["uniformity"].append(uni.cpu())
            history["efficiency"].append(eff.cpu())

            # WGS 核心: 焦平面振幅替换时按权重缩放 sqrt(I_target).
            # 权重场 W: 各光镊处填 sqrt(w_k) 的高斯 (与 wgs._weight_field 同构).
            if use_sparse:
                W = _weight_field_from_basis(flat_idx, basis, pad_msk, weights, N)
            else:
                W = _weight_field(N, spots, weights, spot_radius_px, device=device)
            A_target_w = A_target * W.sqrt()
            phi_f = torch.angle(U_focal)                          # 保留焦平面相位
            U_focal = A_target_w * torch.exp(1j * phi_f)          # 替换振幅, 保留相位

            # 回到 SLM 面, phase-only
            U_slm = torch.fft.fftshift(
                torch.fft.ifft2(torch.fft.ifftshift(U_focal, dim=(-2, -1)), norm=norm),
                dim=(-2, -1),
            )
            phi = torch.angle(U_slm)                               # phase-only 约束

            # 权重更新: 太暗的光镊加大权重. (每样本独立按其 valid 光镊归一化)
            mean_I_exp = mean_I_b[:, None].clamp_min(1e-12)       # (B,1)
            weights = weights * (mean_I_exp / spot_I.clamp_min(1e-12)) ** gamma
            weights = weights * v                                # invalid 不更新
            wn = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            weights = weights / wn * n[:, None]                    # 归一化: 总权重 = 有效光镊数

    return phi, A_focal, history


def _weight_basis(
    N: int, spots: Tensor, radius: float, device=None,
) -> tuple[Tensor, Tensor, Tensor]:
    """预计算镊子高斯权重图的稀疏基 (run 一次, 循环外调用).

    spots (B,K,2) [行,列]. 每镊子在 ±3σ (σ=radius/2) 局部 patch 内算高斯值,
    阈值 >1e-6 取非零像素, 装进三个 (B,K,Pmax) buffer:
      flat_idx : 展平到 N² 的绝对像素索引 (row*N+col), long
      basis    : 预计算高斯值 (只依赖镊子坐标, run 内不变), float
      pad_msk  : 有效位 (padding 用 0), float
    返回 (flat_idx, basis, pad_msk), 缓存 <<1MB (K*Pmax*K_max 大小).

    vs _weight_field (每 iter 重建整张 (B,N,N)): 99.996% 像素是零,
    sparse scatter_add 重算权重图快 ~300x (8192²: 545ms→1.8ms).
    """
    B, K, _ = spots.shape
    if device is None:
        device = spots.device
    spots = spots.to(device=device, dtype=torch.float32)
    sigma = radius / 2.0
    inv = 1.0 / (2.0 * sigma ** 2)
    R = int(max(1, np.ceil(3.0 * sigma)))            # 局部 patch 半径 (像素)
    # 坐标网格 (绝对像素, float 中心): 与 _weight_field 的 (XX-c)²+(YY-r)² 严格一致
    coords = torch.arange(N, device=device, dtype=torch.float32)
    YY, XX = torch.meshgrid(coords, coords, indexing="ij")   # 行=YY, 列=XX
    sp_cpu = spots.cpu().numpy()
    idxs, vals = [], []
    for b in range(B):
        for k in range(K):
            r, c = sp_cpu[b, k]
            if r < 0 or c < 0:                        # 假光镊 → 空 patch (保持 (P,2) 维度)
                idxs.append(torch.zeros(0, 2, dtype=torch.long, device=device))
                vals.append(torch.zeros(0, device=device, dtype=torch.float32))
                continue
            r0 = max(0, int(r) - R); r1 = min(N, int(r) + R + 1)
            c0 = max(0, int(c) - R); c1 = min(N, int(c) + R + 1)
            yy = YY[r0:r1, c0:c1]; xx = XX[r0:r1, c0:c1]   # 绝对坐标 (float)
            g = torch.exp(-((xx - c) ** 2 + (yy - r) ** 2) * inv)
            m = g > 1e-6
            idxs.append(torch.stack([yy[m].long(), xx[m].long()], 1))   # (P,2) [row,col]
            vals.append(g[m])
    Pmax = max(v.shape[0] for v in vals)
    # pad 到 (B,K,Pmax), 装三个 buffer
    flat_idx = torch.zeros(B, K, Pmax, dtype=torch.long, device=device)
    basis = torch.zeros(B, K, Pmax, device=device, dtype=torch.float32)
    pad_msk = torch.zeros(B, K, Pmax, device=device, dtype=torch.float32)
    n = 0
    for b in range(B):
        for k in range(K):
            p = vals[n].shape[0]
            flat_idx[b, k, :p] = idxs[n][:, 0] * N + idxs[n][:, 1]
            basis[b, k, :p] = vals[n]
            pad_msk[b, k, :p] = 1.0
            n += 1
    return flat_idx, basis, pad_msk


def _prep_vectorized(
    N: int, spots: Tensor, radius: float, valid: Tensor, device=None,
) -> dict:
    """全向量化预计算 wgs_batch_vectorized 所需全部索引/权重 (替代 4 个 B×K Python 循环).

    原来 setup 是 spot_mask_pixels + _weight_basis + _spot_index_from_mask +
    _eff_index_from_mask 四个 B×K 双层 Python 循环, @1024² K=1024 实测 ~230ms,
    比 20 次迭代体还大 —— 两个真实负载 (1024²×20, 8192²×60) 的主要瓶颈.
    本函数全部在 GPU 上用固定 (P,P) patch 张量一次算完, 出界/无效像素归零,
    数值与旧路径一致 (每个镊子截取的是同一像素集合).

    返回 dict:
      flat_idx : (B,K,W2) long  权重场 scatter 索引 (出界/无效→0, pad 挡 0)
      basis    : (B,K,W2) float 高斯权重值
      pad      : (B,K,W2) float 有效位 (keep=1)
      spot_flat: (B,K,D2) long  圆盘掩码像素 flat (供 gather)
      spot_ok  : (B,K,D2) float 圆盘有效位 (0/1)
      eff_flat : list[B] of (Me,) long  全部有效镊子圆盘像素并集 (去重)
      n_valid  : (B,) float 每 batch valid 镊子数
    """
    B, K, _ = spots.shape
    if device is None:
        device = spots.device
    o = spots.to(device=device, dtype=torch.float32)
    bad = (o[..., 0] < 0) | (o[..., 1] < 0)                     # (B,K)
    sigma = radius / 2.0
    inv = 1.0 / (2.0 * sigma ** 2)
    Rw = max(1, int(np.ceil(3.0 * sigma)))
    W2 = (2 * Rw + 1) ** 2
    Rd = int(radius)
    D2 = (2 * Rd + 1) ** 2
    sr = o[..., 0]                                              # (B,K) 浮点中心
    sc = o[..., 1]
    # 整数像素网格: 与 _weight_basis 一致 (int(r)+k, int(c)+k), gaussian 用浮点中心偏移
    ri = sr.floor().to(torch.long)                              # (B,K) int 行中心
    ci = sc.floor().to(torch.long)                              # int 列中心

    # ── 权重场: 高斯 patch (±3σ), 出界/无效归零 (与 _weight_basis 同集合) ──
    rr = torch.arange(-Rw, Rw + 1, device=device, dtype=torch.long)
    Gr, Gc = torch.meshgrid(rr, rr, indexing="ij")              # (W,W) {行,列} 相对 int
    Ar = (ri[..., None, None] + Gr[None, None]).float()         # (B,K,W,W) 绝对行 (整数)
    Ac = (ci[..., None, None] + Gc[None, None]).float()         # 绝对列 (整数)
    bad_kk = bad[..., None, None]                               # (B,K,1,1)
    g = torch.exp(-((Ar - sr[..., None, None]) ** 2 + (Ac - sc[..., None, None]) ** 2) * inv)
    Arl = Ar.long(); Acl = Ac.long()                            # 整数索引 (与旧 .long() 一致)
    inside = (Arl >= 0) & (Arl < N) & (Acl >= 0) & (Acl < N) & ~bad_kk
    keep = inside & (g > 1e-6)
    flat_idx = (Arl * N + Acl).clamp(0, N * N - 1).reshape(B, K, W2).to(torch.long)
    basis = g.reshape(B, K, W2)
    pad = keep.float().reshape(B, K, W2)

    # ── 圆盘掩码 (≤radius): spotI 与 eff 的像素集合 ──
    # 完全复刻 spot_mask_pixels 语义: 窗口 = int(r)±Rd, 圆盘判定用浮点中心
    rd = int(radius)
    w_ = 2 * rd + 1
    off_r = torch.arange(-rd, rd + 1, device=device, dtype=torch.long)
    Dr, Dc = torch.meshgrid(off_r, off_r, indexing="ij")         # (D,D) int 相对
    ar2 = (ri[..., None, None] + Dr[None, None])                 # (B,K,D,D) 绝对行 int
    ac2 = (ci[..., None, None] + Dc[None, None])
    arf = ar2.float(); acf = ac2.float()
    disk = ((arf - sr[..., None, None]) ** 2 + (acf - sc[..., None, None]) ** 2) <= radius ** 2
    bad_kk2 = bad[..., None, None]
    inside2 = ((ar2 >= 0) & (ar2 < N) & (ac2 >= 0) & (ac2 < N) &
               ~bad_kk2).reshape(B, K, w_ * w_)
    disk2 = disk.reshape(B, K, w_ * w_)
    ok = (inside2 & disk2).float()                                # (B,K,D2) 有效位
    spot_flat = (ar2 * N + ac2).clamp(0, N * N - 1).reshape(B, K, w_ * w_).to(torch.long)
    spot_ok = ok                                                          # (B,K,D2)

    # ── eff: 全部有效镊子圆盘像素并集 (GPU 去重, 免 .tolist()) ──
    eff_flat = []
    for b in range(B):
        keepb = ok[b].bool()                                    # (K,D2)
        fb = (ar2[b] * N + ac2[b]).clamp(0, N * N - 1).reshape(K, D2).to(torch.long)
        eff_flat.append(torch.unique(fb[keepb]))

    return {
        "flat_idx": flat_idx, "basis": basis, "pad": pad,
        "spot_flat": spot_flat, "spot_ok": spot_ok,
        "eff_flat": eff_flat,
    }


def _weight_field_from_basis(
    flat_idx: Tensor, basis: Tensor, pad_msk: Tensor, weights: Tensor,
    N: int,
) -> Tensor:
    """从预计算稀疏基 + 当前 weights 重算权重图 (B,N,N). scatter_add.

    循环内每 iter 调用: contrib = basis * weights * pad_msk (B,K,Pmax),
    scatter_add 到展平 (B,N*N) 上, 再 reshape (B,N,N) + Wmax 归一化.
    无 exp, 无 N² 重建. 与 _weight_field 数值一致 (浮点中心保留).
    """
    B, K, Pmax = basis.shape
    W = torch.zeros(B, N * N, device=basis.device, dtype=torch.float32)
    contrib = (basis * weights[:, :, None] * pad_msk).reshape(B, -1)   # (B,K*Pmax)
    W.scatter_add_(1, flat_idx.reshape(B, -1), contrib)
    W = W.view(B, N, N)
    Wmax = W.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return W / Wmax


def _spot_index_from_mask(mask_pixels, N, K, device):
    """把单 batch 的 spot_mask_pixels [list[(ri,ci)]] 压成向量化 spotI 的 (flat, seg).

    flat = ri*N + ci (long), seg = 每像素所属光镊 k (long), 仅本 batch (0..K-1).
    与 wgs_batch 的 mask 语义一致: 假光镊/无效镊子不收录 (spotI 保持 0).
    一次预计算, 迭代内 spotI 只需一次 gather + scatter_add (免 K 次 Python 循环).
    """
    flats, segs = [], []
    for k in range(K):
        ri, ci = mask_pixels[k]
        if ri.numel() > 0:
            flats.append(ri.long() * N + ci.long())
            segs.append(torch.full((ri.numel(),), k, dtype=torch.long, device=device))
    if not flats:
        return torch.zeros(0, dtype=torch.long, device=device), \
               torch.zeros(0, dtype=torch.long, device=device)
    return torch.cat(flats), torch.cat(segs)


def _eff_index_from_mask(mask_pixels, N, K, valid, device):
    """把单 batch 的 (ri,ci) 掩码块压成 efficiency 向量化用的并集 flat 索引.

    efficiency = 落在所有 valid 光镊圆盘**并集**内的能量 / 焦面总能量.
    这里把 mask_pixels[k] 的像素去重合并成一段 flat 索引 (ri*N+ci), 一次 gather+sum
    即得并集能量 —— 替代 propagator.efficiency_sparse 每轮 K 次 Python .tolist()
    重建 GPU tensor (@512² K=484 实测 11.4ms → 27μs). 数值逐位一致 (去重后同一集合).
    """
    B = len(mask_pixels)
    outs = []
    for b in range(B):
        seen = set()
        for k in range(len(mask_pixels[b])):
            if not valid[b, k]:
                continue
            ri, ci = mask_pixels[b][k]
            for ii, jj in zip(ri.tolist(), ci.tolist()):
                seen.add((ii, jj))
        outs.append(torch.tensor(sorted(r * N + c for r, c in seen), device=device, dtype=torch.long))
    return outs   # list[B] of (M_b,) flat idx

def wgs_batch_vectorized(
    A_slm: Tensor,              # (B,N,N) 入射高斯振幅
    A_target: Tensor,           # (B,N,N) 目标光镊阵列振幅 sqrt(I_target)
    spots: Tensor,              # (B,K,2) 光镊中心; 假光镊 (-1,-1)
    valid: Tensor,              # (B,K) bool
    iterations: int = 50,
    gamma: float = 0.5,
    spot_radius_px: float = 2.0,
    seed: int = 0,
    norm: str = "ortho",
    device: str | None = None,
    phi_init: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict]:
    """批量 WGS, 向量化 spotI (与 wgs_batch 数值逐位一致, 只删 K 次 Python 循环).

    与 wgs_batch(use_sparse=True) 共用同一循环骨架 (同物理栈: cos/sin → propagate →
    同权重场 _weight_field_from_basis → 同权重更新/归一化), 唯一点不同:
      spotI: flatten gather + torch.scatter_add (一次向量化) 替代
             spot_intensities_sparse 的 B×K 次 Python 循环 (大 K/大 N 的瓶颈, 实测
             @512² K=484 占一轮 ~98%).
    数值: 同一索引布局 → 同一求和顺序 → 与 wgs_batch 逐位一致 (float32 不引入差异).
    返回 (phi (B,N,N) 弧度, A_focal (B,N,N) 振幅, history).
    """
    B, N, _ = A_slm.shape
    if device is None:
        device = A_slm.device
    else:
        A_slm = A_slm.to(device); A_target = A_target.to(device)
        spots = spots.to(device); valid = valid.to(device)

    K = spots.shape[1]
    if phi_init is None:
        phi = _init_phi(B, N, seed, device)
    else:
        phi = phi_init.to(device=device, dtype=torch.float32)
    weights = torch.ones(B, K, device=device)

    # 全向量化预计算: 一次产出 权重基 + 圆盘掩码 + 并集 (替代 4 个 B×K Python 循环,
    # @1024² K=1024 setup ~230ms → ~10ms)
    prep = _prep_vectorized(N, spots, spot_radius_px, valid, device=device)
    flat_idx, basis, pad_msk = prep["flat_idx"], prep["basis"], prep["pad"]
    spot_flat, spot_ok = prep["spot_flat"], prep["spot_ok"]
    eff_idx = prep["eff_flat"]                                   # list[B] of (Me,)
    D2 = spot_flat.shape[-1]

    history = {"uniformity": [], "efficiency": []}
    v = valid.to(torch.float32)
    n = v.sum(dim=1).clamp_min(1)                       # (B,)

    with torch.inference_mode():
        for _ in range(iterations):
            cos_p = torch.cos(phi); sin_p = torch.sin(phi)
            U_focal = P.propagate(A_slm, cos_p, sin_p, norm=norm)
            I_focal = U_focal.real.pow(2) + U_focal.imag.pow(2)
            A_focal = I_focal.sqrt()

            # ── 向量化 spotI: 每 batch 按圆盘 gather + 掩码求和 (免 scatter_add) ──
            I_f = I_focal.reshape(B, -1)                 # (B, N*N)
            spot_I = torch.zeros(B, K, device=device, dtype=I_focal.dtype)
            for b in range(B):
                sf = spot_flat[b].reshape(-1)            # (K*D2,)
                vals = I_f[b].index_select(0, sf).reshape(K, D2)   # (K,D2)
                spot_I[b] = (vals * spot_ok[b]).sum(dim=1)         # (K,)
            spot_I = spot_I * v                         # invalid 镊子不参与

            mean_I_b = (spot_I.sum(dim=1)) / n          # (B,)
            diff = (spot_I - mean_I_b[:, None]) * v
            var = (diff.pow(2).sum(dim=1)) / n
            std = var.clamp_min(0).sqrt()
            uni = 1.0 - std / mean_I_b.clamp_min(1e-12)
            # 向量化 eff: 并集 flat gather + sum
            eff = torch.empty(B, device=device, dtype=I_focal.dtype)
            for b in range(B):
                eff[b] = I_f[b][eff_idx[b]].sum() / I_f[b].sum().clamp_min(1e-12)
            history["uniformity"].append(uni.cpu())
            history["efficiency"].append(eff.cpu())

            W = _weight_field_from_basis(flat_idx, basis, pad_msk, weights, N)
            A_target_w = A_target * W.sqrt()
            phi_f = torch.angle(U_focal)
            U_focal = A_target_w * torch.exp(1j * phi_f)
            U_slm = torch.fft.fftshift(
                torch.fft.ifft2(torch.fft.ifftshift(U_focal, dim=(-2, -1)), norm=norm),
                dim=(-2, -1),
            )
            phi = torch.angle(U_slm)

            mean_I_exp = mean_I_b[:, None].clamp_min(1e-12)
            weights = weights * (mean_I_exp / spot_I.clamp_min(1e-12)) ** gamma
            weights = weights * v
            wn = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            weights = weights / wn * n[:, None]

    return phi, A_focal, history


def _weight_field(
    N: int, spots: Tensor, weights: Tensor, radius: float, device=None,
    chunk: int = 4,
) -> Tensor:
    """[稠密兜底/对照] 把逐光镊权重展开成焦平面同尺寸权重图, 对齐 wgs._weight_field.

    spots (B,K,2), weights (B,K) -> W (B,N,N). GPU 向量化分块 (避免 (B,K,N,N) 爆显存):
    每次处理 chunk 个镊子, w 形状 (B,chunk,N,N), 累加到 (B,N,N). 大 N/K 慢 (重建 N² 图).

    生产路径用 _weight_basis + _weight_field_from_basis (稀疏 scatter_add, 快 ~300x).
    本函数保留作正确性对照与 _weight_basis 失效时的兜底.
    """
    B, K, _ = spots.shape
    if device is None:
        device = spots.device
    coords = torch.arange(N, device=device, dtype=torch.float32)
    YY, XX = torch.meshgrid(coords, coords, indexing="ij")   # (N,N)
    spots = spots.to(device=device, dtype=torch.float32)
    weights = weights.to(device=device, dtype=torch.float32)
    sigma = radius / 2.0
    inv = 1.0 / (2.0 * sigma ** 2)
    W = torch.zeros(B, N, N, device=device, dtype=torch.float32)
    for i in range(0, K, chunk):
        k = min(chunk, K - i)
        r = spots[:, i:i+k, 0][:, :, None, None]            # (B,k,1,1)
        c = spots[:, i:i+k, 1][:, :, None, None]            # (B,k,1,1)
        wt = weights[:, i:i+k][:, :, None, None]            # (B,k,1,1)
        w = torch.exp(-((XX[None, None] - c) ** 2 + (YY[None, None] - r) ** 2) * inv)  # (B,k,N,N)
        W = W + (wt * w).sum(1)                              # (B,N,N)
    Wmax = W.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return W / Wmax
