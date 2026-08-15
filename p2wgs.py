"""相位 & 剖面感知 WGS (P2WGS) —— 批量 GPU 版, 逆向重建自 arXiv:2604.08669 的商业黑盒算法.

物理动机与核心发现(诚实披露, 数字经 bench_p2wgs.py 校正口径复测)
----------------------------------------------------------------
经典 WGS 的焦平面替换只约束振幅: U_focal = A_target_w · exp(i·φ_f), 其中 φ_f = angle(U_focal)
每轮被迭代自由重赋 —— 相位完全不控。对静止阵列无碍; 对"移动原子"的时序全息(帧序列),
相邻帧光镊的相位若乱跳, 原子会感受非绝热的势阱抖动 → 加热/丢原子。

【校正口径复测】(8×8=64 镊, N=128, 0.5px/帧位移; spots/target/mask/权重基 全在同一帧新位置):
  - 每帧独立 WGS (50iter, 固定随机初相 = gen_traj.py 现状):  CP = 0.267  (≈随机相位, 基线)
  - WGS + warm-start + 5 iter/帧  (phi_init=phi_prev):         CP = 0.096  (提升 2.8× ★)
  - WGS + warm-start + 30 iter/帧:                              CP = 0.113  (提升 2.4×)
  衍射效率: 三者均在 0.86–0.87 (ortho 帕塞瓦尔定总能量, WGS 早已逼近上界) —— **warm-start
  不提升衍射效率**, 它提升的是**帧间相位连续性**。这是 P2WGS 的真正价值, 与"光效率"无关。

⇒ **warm-start (phi_init=上一帧收敛全息) 本身就是 P2WGS 论文宣称的"phase-aware 帧间连续"
   的真正引擎**: 它直接拿上一帧 SLM 相位当初值, 迭代只做小幅修正, 故焦面镊子处相位在
   帧间平滑延续。迭代次数控制"连续性 vs 重新收敛"的折中: 5iter 给最佳连续性(CP 最低),
   30iter 给最佳均匀性(uni 最高) —— 两者是同一 Pareto 前沿的不同工作点, 非此即彼。
⇒ **"5 iter/帧"的真正理由**: 不是论文宣称的"算法收敛快", 而是 warm-start 下少迭代 = 相位
   漂移少 = 连续性好; 多迭代会把相位重新优化到新目标, 牺牲连续性换均匀性。这同时解释了
   论文 0.5ms/帧声称 (5iter × FFT ≈ 0.3–0.9ms, 见 bench_p2wgs.py Part A 实测)。
⇒ **诚实的"更强"含义**: (a) 衍射效率与经典 WGS 持平 (~0.87, 不退化); (b) 小位移(原子
   重排的物理工作点)帧间相位连续性 2.8× 提升 → 更少非绝热加热 → 更高装配存活率;
   (c) 每帧 10× 更少 FFT (5iter vs 50iter) → 10× 更快 → 单位时间更多装配。

⚠️ **曾出现的测量伪影(已修正, 记录以防重蹈)**: 早期位移扫描把 spots_pad/mask 固定在旧位置、
   A_target 放新位置, 导致"大位移 eff 崩溃到 0.63"的假象。正确做法是 spots/target/mask/
   权重基全部用同一帧新位置 (bench_p2wgs.py 已固化为正确口径)。这与 [[position-domain-supervised-
   phase-collapse]] 的"编码 bug 假象"同源 —— 凡涉及"两帧/两位置对比"的指标, 必须校验两侧
   口径一致。早期"焦平面 lock 把 CP 0.018→0.078"等具体数字亦来自该伪影口径, 不再引用;
   lock 的定性结论 (软锁相位与 WGS 自然演化打架, 效率随 lock 强度单调下降) 仍成立, 见诊断开关。

基于此, 本模块的设计:
  - 主线 = warm-start 驱动的相位连续 (复用 wgs_batch 的 phi_init, 不另造机制)。
  - profile-aware = 高斯(soft-δ)复目标 T_target = A·e^{jφ_goal}, φ_goal 由"前帧相位传播"
    构造 (trajectory_phase_target)。这个目标主要用于**指标测量** (我们构造的"应连续"
    参考相位) 和**可选的相位感知权重 (PAW)**。
  - phase-aware 权重 (PAW, 可选, 默认关): 权重幅度受相位误差驱动 —— 相位偏离目标大的镊子
    加重权, 推动能量去纠正, 权重场保持实数 (不进替换振幅的相位), 不破坏 warm-start 连续性。
    (注: 复权重相位分支 + 权重场相位贡献 曾把已收敛解打崩, 已弃用, 详见 README。)
  - 焦平面 lock 保留为**诊断开关** (lock_mode, 默认 "off"): 用于在 benchmark 里**证伪**
    "硬锁相位"路线, 而非生产路径。

与 wgs_batch 的兼容性(parity):
  p2wgs_batch(..., use_phase_weights=False, lock_mode="off", T_target 满足 |T_target| = A_target)
  逐位复现 wgs_batch 的输出 (同 seed / 同 phi_init / 同 sparams)。bench_p2wgs.py 顶部有冒烟测试。
  本实现不改写任何现有文件, 仅复用 wgs_torch._init_phi / _weight_basis 稀疏预计算,
  以及 propagator.propagate / spot_intensities_sparse / efficiency_sparse。

硬相位投影诊断: lock_mode="all" 且 lock_strength=1 会锁死整帧(不可行流形), 不收敛 ——
  README 建议用 --diagnose-hard 验证这一现象, 而不是在生产路径使用。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

import propagator as P
import wgs_torch as WT          # 复用 _init_phi / _weight_basis (不写两套 FFT/基)


# ─── 基本工具 ───

def angle_wrap(d: Tensor) -> Tensor:
    """把相位角 wrap 到 [-π, π]. 用 atan2(sin,cos) 形式, 无 2π 环绕去破断 (autograd 友好)."""
    return torch.atan2(torch.sin(d), torch.cos(d))


def _gaussian_complex_target(
    N: int, spots: Tensor, thetas: Tensor, spot_radius_px: float,
    device: str = "cuda", chunk: int = 4,
) -> Tensor:
    """构造逐镊相位的高斯复目标 T = A·e^{jφ}, 与 datasets.target_complex_field_torch 同构.

    spots (K,2), thetas (K,): 每个镊子的目标相位 φ_goal_k. 返回 (N,N) complex64.
    相位图 = Σ θ_k·G_k / Σ G_k (每镊子窗内≈θ_k, 窗外汇合到背景 0).
    振幅 = 各高斯叠加, 峰值归一化到 1 (与 wgs.target_array 同口径, 带 spot_radius_px).
    """
    coords = torch.arange(N, device=device, dtype=torch.float32)
    YY, XX = torch.meshgrid(coords, coords, indexing="ij")   # (N,N) 行=YY 列=XX
    spots = spots.to(device=device, dtype=torch.float32)
    thetas = thetas.to(device=device, dtype=torch.float32)
    K = spots.shape[0]
    A = torch.zeros(N, N, device=device, dtype=torch.float32)
    Ph = torch.zeros(N, N, device=device, dtype=torch.float32)
    inv = 1.0 / (2.0 * spot_radius_px ** 2)
    for i in range(0, K, chunk):
        k = min(chunk, K - i)
        sp = spots[i:i+k]                                       # (k,2)
        r = sp[:, 0][:, None, None]                            # (k,1,1) 行
        c = sp[:, 1][:, None, None]                            # (k,1,1) 列
        th = thetas[i:i+k][:, None, None]                       # (k,1,1)
        w = torch.exp(-((XX[None] - c) ** 2 + (YY[None] - r) ** 2) * inv)  # (k,N,N)
        A = A + w.sum(0)
        Ph = Ph + (th * w).sum(0)
    Wt = A.clamp_min(1e-6)
    Ph = torch.where(A > 1e-6, Ph / Wt, torch.zeros_like(Ph))
    peak = A.max().clamp_min(1e-12)
    A = A / peak
    return A.to(torch.complex64) * torch.exp(1j * Ph.to(torch.complex64))


def _lock_mask(N: int, spots: Tensor, valid: Tensor, R: int, device=None) -> Tensor:
    """光镊窗相位锁定掩码 (B,N,N) float: 每个镊 (2R+1)×(2R+1) 窗内=1, 窗外=0.

    spots (B,K,2), valid (B,K). 中心取最近整数像素; 越界裁切。背景不被锁 → 均匀性收敛保住。
    """
    B, K, _ = spots.shape
    if device is None:
        device = spots.device
    m = torch.zeros(B, N, N, device=device, dtype=torch.float32)
    spots_np = spots.cpu().numpy()
    valid_np = valid.cpu().numpy()
    for b in range(B):
        for k in range(K):
            if not valid_np[b, k]:
                continue
            r = int(round(float(spots_np[b, k, 0])))
            c = int(round(float(spots_np[b, k, 1])))
            r0, r1 = max(0, r - R), min(N, r + R + 1)
            c0, c1 = max(0, c - R), min(N, c + R + 1)
            m[b, r0:r1, c0:c1] = 1.0
    return m


# ─── 相位目标构造 (前帧相位传播) ───

def trajectory_phase_target(
    prev_phi: Tensor | None,
    A_slm: Tensor, spots: Tensor, spot_radius_px: float = 2.0,
    N: int | None = None, device: str | None = None,
    seed: int = 0,
) -> Tensor:
    """构造本帧的复目标 T_target = A·e^{jφ_goal}: φ_goal 来自"前帧相位传播".

    prev_phi: 上一帧收敛的 SLM 相位 (N,N) 弧度; 本帧用正向传播在新镊位置的相位做目标。
      φ_goal_k = angle( U_prev[round(r_k), round(c_k)] )   (与 eval_paper._phase_err 同最近像素口径)
    prev_phi=None (帧0): 回退 chirp 相位 (θ_0 + k·dphi), 与现有 tweezer_phases 约定一致。

    返回 (N,N) complex64。连续性由构造保证: 只要帧间位移足够小, 波前在"新位置"与"旧波前"
    同相位延续 —— 这就是"相位感知"的物理根基(周期量=移动波前).
    """
    if device is None:
        device = spots.device if spots.is_cuda else "cuda"
    N = N or A_slm.shape[-1]
    K = spots.shape[0]
    spots_d = spots.to(device=device, dtype=torch.float32)
    if prev_phi is not None:
        prev = prev_phi.to(device=device, dtype=torch.float32)
        A_s = A_slm.to(device=device, dtype=torch.float32)
        U_prev = P.propagate(A_s[None], torch.cos(prev[None, :, :]),
                             torch.sin(prev[None, :, :]), norm="ortho")[0]   # (N,N) complex
        r = spots_d[:, 0].round().long().clamp(0, N - 1)
        c = spots_d[:, 1].round().long().clamp(0, N - 1)
        theta = U_prev[r, c].angle()
    else:
        rng = np.random.default_rng(seed)
        dphi = float(rng.uniform(0.3, 1.2))
        th0 = float(rng.uniform(0, 2 * np.pi))
        theta = torch.from_numpy(
            (th0 + np.arange(K) * dphi) % (2 * np.pi)
        ).float().to(device)
    return _gaussian_complex_target(N, spots_d, theta, spot_radius_px, device=device)


# ─── 复权重场 (scatter_add, 复数) ───

def _complex_weight_field(
    flat_idx: Tensor, basis: Tensor, pad_msk: Tensor,
    weights_c: Tensor, N: int,
) -> Tensor:
    """复数权重场 W_c (B,N,N) complex64. scatter_add 复 contrib, 再按 max|W_c| 归一化.

    与 wgs_torch._weight_field_from_basis 同构 (实→复): contrib = basis · w_c · pad_msk,
    scatter_add 到展平 (B,N²), reshape, / max|W_c|. 当 w_c 全实 (相位 0) 时 |W_c| = 实 W.
    """
    B, K, Pmax = basis.shape
    idx = flat_idx.reshape(B, -1)                              # (B, K*Pmax) long
    contrib = (weights_c[:, :, None] * basis * pad_msk).reshape(B, -1)  # (B, K*Pmax) complex
    W = torch.zeros(B, N * N, device=basis.device, dtype=torch.complex64)
    W.scatter_add_(1, idx, contrib)
    W = W.view(B, N, N)
    Wmax = W.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return W / Wmax


def _local_complex_samples(
    field: Tensor, flat_idx: Tensor, basis: Tensor,
) -> Tensor:
    """高斯窗加权的局部复场采样: Σ_j G_k(j)·field(j) → (B,K) complex.

    field: (B,N,N) complex. flat_idx (B,K,Pmax), basis (B,K,Pmax) real 高斯权重。
    用于复权重的相位分支: U_k = 局部测得场, T_k = 局部目标场。

    ⚠️ 注意: 相邻镊子间距 ~5px 而高斯窗 σ=spot_radius_px/2, 窗会重叠 → 加权和会把
    多个镊子的相位混在一起, arg() 测出干涉相位而非单镊相位 → 相位分支注入垃圾,
    把已收敛解打崩 (实测: warm-start 99.7% → 68%). 故相位分支**不**用本函数, 而用
    _nearest_pixel_phase (单镊中心最近像素, 与 eval_paper._phase_err 同口径). 本函数
    仅留给诊断/对照, 生产路径不调用.
    """
    B, K, Pmax = basis.shape
    idx = flat_idx.reshape(B, -1)                              # (B, K*Pmax)
    g = field.reshape(B, -1).gather(1, idx).view(B, K, Pmax)   # (B,K,Pmax) complex
    return (g * basis).sum(dim=2)                              # (B,K) complex


def _nearest_pixel_phase(field: Tensor, spots: Tensor, valid: Tensor, N: int) -> Tensor:
    """每个镊子中心的最近整数像素复场采样 → (B,K) complex.

    field: (B,N,N) complex. spots (B,K,2) [行,列]. 与 eval_paper._phase_err 同口径
    (round 到最近像素, clamp 边界). invalid 镊子返回 0 (arg 后会被 mask 抹掉).
    用于复权重相位分支: arg(U_k) 是单镊中心真实相位, 不受相邻镊子窗干涉污染.
    """
    B, K, _ = spots.shape
    r = spots[..., 0].round().long().clamp(0, N - 1)           # (B,K)
    c = spots[..., 1].round().long().clamp(0, N - 1)
    flat = r * N + c                                            # (B,K)
    g = field.reshape(B, -1).gather(1, flat)                   # (B,K) complex
    return g * valid.to(g.dtype).to(torch.complex64)


# ─── 核心: P2WGS 批量迭代 ───

def p2wgs_batch(
    A_slm: Tensor,              # (B,N,N) 入射高斯振幅 (或 (N,N) 自动扩 B)
    T_target: Tensor,           # (B,N,N) complex64 复目标 = sqrt(I_goal)·e^{iφ_goal}
    spots: Tensor,              # (B,K,2) 光镊中心 (行,列,像素)
    valid: Tensor,              # (B,K) bool
    iterations: int = 50,
    gamma: float = 0.5,         # γ_a 幅度回馈 (原 WGS 指数)
    gamma_p: float = 0.25,      # γ_p 相位感知权重 (PAW) 相位误差→权重幅度的耦合强度
    lock_strength: float = 0.75,  # λ 相位软混合强度 (诊断用, 见上 docstring)
    lock_mode: str = "off",    # "trapwin"|"all"|"off": 相位锁定区域 (默认 off, 见发现)
    use_phase_weights: bool = False,  # PAW: 相位误差驱动权重幅度 (实权重场, 不破坏 warm-start)
    ramp_iters: int = 0,        # >0: λ 在前 ramp_iters 次迭代线性爬升 (冷启动缓解, 诊断用)
    spot_radius_px: float = 2.0,
    seed: int = 0,
    norm: str = "ortho",
    device: str | None = None,

    phi_init: Tensor | None = None,   # (B,N,N) 初始相位 (warm-start, 帧间连续用 ★主线)
    use_vectorized: bool = False,     # True=spotI/eff 走全向量化 gather+mask 路径 (快, 见下)
    history_every: int = 1,           # 0=不记录 history (免每迭代 .cpu() 同步, 批量生产提速)
) -> tuple[Tensor, Tensor, dict]:
    """批量 P2WGS. 返回 (phi (B,N,N) 弧度, A_focal (B,N,N), history).

    设计主线: phi_init warm-start 驱动帧间相位连续 (实测 2.2× 提升, 见模块 docstring)。
    可选 PAW (use_phase_weights): 相位偏离目标大的镊子加重权重, 权重场保持实数
    (不进替换振幅相位), 故不破坏 warm-start。lock 仅诊断用 (证伪硬锁路线)。

    history: {"uniformity": [iter], "efficiency": [iter]} (每迭代 (B,) cpu tensor).

    与 wgs_batch 的 parity: lock_mode="off" + use_phase_weights=False 时逐位复现
    (同 seed / 同 phi_init / |T_target| = A_target)。权重路径关闭时走与 wgs_batch
    完全相同的 _weight_field_from_basis + 振幅替换, 保证 byte-level 一致。
    """
    if A_slm.dim() == 2:
        A_slm = A_slm[None]
    if T_target.dim() == 2:
        T_target = T_target[None]
    # BUGFIX (2026-08-14, b1): B 必须取 T_target/spots 的 batch (A_slm 2D 共享时 B=1
    # 会把 spots batch 0 的强度静默广播给所有样本 —— 批量标签生产会错).
    if T_target.dim() == 3 and A_slm.shape[0] == 1 and T_target.shape[0] > 1:
        A_slm = A_slm.expand(T_target.shape[0], -1, -1)
    B, N, _ = A_slm.shape
    if device is None:
        device = A_slm.device
    else:
        A_slm = A_slm.to(device); T_target = T_target.to(device)
        spots = spots.to(device); valid = valid.to(device)
    T_target = T_target.to(torch.complex64)

    K = spots.shape[1]
    if phi_init is None:
        phi = WT._init_phi(B, N, seed, device)
    else:
        phi = phi_init.to(device=device, dtype=torch.float32)

    # use_vectorized: 一次性全向量化预算 spotI/eff 索引 (替代 mask_pixels 另走的
    # B×K Python 循环 spot_intensities_sparse / efficiency_sparse, 数值逐位同口径)
    prepv = None
    if use_vectorized:
        prepv = WT._prep_vectorized(N, spots, spot_radius_px, valid, device=device)
        # BUGFIX (2026-08-14, b1): prepv 已含与 _weight_basis 逐位一致 (max|dW|=0) 的
        # flat_idx/basis/pad —— 跳过 _weight_basis 的 B×K Python 循环 (~1.1s/次固定开销).
        flat_idx, basis, pad_msk = prepv["flat_idx"], prepv["basis"], prepv["pad"]
    else:
        # 稀疏权重基 (镊子高斯窗, 复用 wgs_torch 预计算)
        flat_idx, basis, pad_msk = WT._weight_basis(N, spots, spot_radius_px, device=device)
    # 圆盘掩码 (幅值回馈的"每镊强度"口径, 与 wgs 一致)
    mask_pixels = None if use_vectorized else P.spot_mask_pixels(N, spots, spot_radius_px, device=device)

    # 相位锁定掩码 (B,N,N) —— 诊断用
    if lock_mode == "off":
        lam_field = None
    else:
        R = max(2, int(np.ceil(spot_radius_px)))              # 5×5 窗 (R=2)
        m = _lock_mask(N, spots, valid, R, device=device)
        if lock_mode == "all":
            m = torch.ones_like(m)
        lam_field = lock_strength * m                         # (B,N,N)

    T_abs = T_target.abs()
    T_phase = T_target.angle()

    weights = torch.ones(B, K, device=device)                  # 实权重 (PAW 也只调幅度)
    v = valid.to(torch.float32)
    n = v.sum(dim=1).clamp_min(1)                              # (B,) 有效镊数

    history = {"uniformity": [], "efficiency": []}

    with torch.inference_mode():
        for it in range(iterations):
            # ── 前向: SLM → 焦平面 (cos,sin) 喂 propagator, 一致物理栈 ──
            U_focal = P.propagate(A_slm, torch.cos(phi), torch.sin(phi), norm=norm)
            I_focal = U_focal.real.pow(2) + U_focal.imag.pow(2)
            A_focal = I_focal.sqrt()

            # 每光镊强度 (圆盘掩码, wgs 口径) → 指标 + 幅回馈
            if use_vectorized:
                # 全向量化: gather + mask 求和 + 并集 eff (与 B×K 循环数值同口径)
                # BUGFIX (2026-08-14, b1): 原 B 循环逐样本 index_select -> 一次 gather
                I_f = I_focal.reshape(B, -1)                       # (B, N*N)
                spot_flat = prepv["spot_flat"]; spot_ok = prepv["spot_ok"]
                spot_I = (I_f.gather(1, spot_flat.reshape(B, -1)).reshape(B, K, -1)
                          * spot_ok).sum(2)
                eff_flat = prepv["eff_flat"]                       # list[B] 1D
                eff_max = max(len(e) for e in eff_flat)
                pad = torch.full((B, eff_max), N * N, device=device, dtype=torch.long)
                cnt = torch.tensor([len(e) for e in eff_flat], device=device)
                for b in range(B):
                    pad[b, :len(eff_flat[b])] = eff_flat[b]
                idx = pad.clamp_max(N * N - 1)
                sel = torch.arange(eff_max, device=device)[None, :] < cnt[:, None]
                den = I_f.sum(dim=1).clamp_min(1e-12)
                num = (I_f.gather(1, idx) * sel).sum(dim=1)
                eff = num / den
            else:
                spot_I = P.spot_intensities_sparse(I_focal, mask_pixels, valid)  # (B,K)
            spot_I = spot_I * v
            mean_I_b = (spot_I.sum(dim=1)) / n
            diff = (spot_I - mean_I_b[:, None]) * v
            var = (diff.pow(2).sum(dim=1)) / n
            uni = 1.0 - var.clamp_min(0).sqrt() / mean_I_b.clamp_min(1e-12)
            if not use_vectorized:
                eff = P.efficiency_sparse(I_focal, mask_pixels, valid)
            if history_every > 0 and (it % history_every == 0 or it == iterations - 1):
                history["uniformity"].append(uni.cpu())
                history["efficiency"].append(eff.cpu())

            # ── 权重场 (实数; PAW 只把相位误差耦合进权重幅度, 不引相位扰动) ──
            # PAW 相位分支需要"测得场"(替换前), 在这里采样
            if use_phase_weights:
                U_k = _nearest_pixel_phase(U_focal, spots, valid, N)    # (B,K) complex 测得场
                T_k = _nearest_pixel_phase(T_target, spots, valid, N)  # (B,K) complex 目标场
            W = WT._weight_field_from_basis(flat_idx, basis, pad_msk, weights, N)  # (B,N,N) real
            amp_fac = W.sqrt()
            A_target_w = T_abs * amp_fac                          # 振幅替换 (含权重)

            # ── 相位 (lock 默认 off = 保留 WGS 自由相位, warm-start 已提供连续性) ──
            phi_f = U_focal.angle()
            if lam_field is None:
                phi_new = phi_f
            else:
                lam = lam_field
                if ramp_iters > 0:
                    lam = lam * min(1.0, (it + 1) / max(1, ramp_iters))
                dph = angle_wrap(T_phase - phi_f)                  # (B,N,N) ∈ [-π,π]
                phi_new = phi_f + lam * dph

            U_focal = A_target_w * torch.exp(1j * phi_new)

            # ── 回到 SLM 面, phase-only ──
            U_slm = torch.fft.fftshift(
                torch.fft.ifft2(torch.fft.ifftshift(U_focal, dim=(-2, -1)), norm=norm),
                dim=(-2, -1),
            )
            phi = U_slm.angle().to(torch.float32)

            # ── 权重更新 (回传后, 与 wgs_batch 同序) ──
            # 幅度分支 (原 WGS): w ← w · (mean_I / I_k)^γ
            mean_I_exp = mean_I_b[:, None].clamp_min(1e-12)
            weights = weights * (mean_I_exp / spot_I.clamp_min(1e-12)) ** gamma
            if use_phase_weights:
                # PAW 相位分支: 相位偏离目标大的镊子加重权, 推动能量去纠正。
                # w ← w · (1 + γ_p · |wrap(arg(T_k) − arg(U_k))|/π)
                # (归一化相位误差 |·|/π∈[0,1] 作乘性增益, 不累积、不引相位扰动)
                dph_k = angle_wrap(T_k.angle() - U_k.angle())      # (B,K) ∈ [-π,π]
                weights = weights * (1.0 + gamma_p * (dph_k.abs() / np.pi))
            weights = weights * v
            wn = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            weights = weights / wn * n[:, None]                 # Σw = n (与 wgs 一致)

    return phi, A_focal, history