# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Chunk-based Gated Delta Rule Triton Kernels (Forward Only)

这个模块实现了 chunk 模式的 gated delta rule 操作的 Triton kernels（仅前向传播）。
Chunk 模式适用于推理和长序列场景，通过并行化计算提高效率。

主要组件：
1. chunk_gated_delta_rule_fwd_h_kernel - 计算隐状态 h 的前向 kernel
2. 辅助函数：recompute_w_u, chunk_fwd_o 等（前向传播相关）

注意：本实现仅支持前向传播（推理），不支持反向传播（训练）。
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# 辅助函数 - 这些函数的实现将在后续添加
# ============================================================================

def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    在 chunk 内部进行局部累积和。
    
    TODO: 实现此函数
    """
    raise NotImplementedError("chunk_local_cumsum will be implemented later")


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    计算缩放的 K^T @ K 矩阵。
    
    TODO: 实现此函数
    """
    raise NotImplementedError("chunk_scaled_dot_kkt_fwd will be implemented later")


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    output_dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    求解下三角矩阵方程。
    
    TODO: 实现此函数
    """
    raise NotImplementedError("solve_tril will be implemented later")


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    重新计算 w 和 u（WY 表示）。
    
    TODO: 实现此函数
    """
    raise NotImplementedError("recompute_w_u_fwd will be implemented later")


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    计算输出 o = qKv + qh。
    
    TODO: 实现此函数
    """
    raise NotImplementedError("chunk_fwd_o will be implemented later")


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    准备 chunk 索引。
    
    TODO: 实现此函数
    """
    raise NotImplementedError("prepare_chunk_indices will be implemented later")


def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    准备 chunk 偏移量。
    
    TODO: 实现此函数
    """
    raise NotImplementedError("prepare_chunk_offsets will be implemented later")


# ============================================================================
# 主要的 Triton Kernels
# ============================================================================

@triton.jit
def _exp(x):
    """Triton exp 函数包装"""
    return tl.exp(x)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3]
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Chunk Gated Delta Rule 前向传播 kernel - 计算隐状态 h
    
    这个 kernel 计算每个 chunk 的隐状态 h，支持：
    - 变长序列 (通过 cu_seqlens)
    - 门控机制 (g, gk)
    - 初始状态和最终状态
    - 保存新的 value (v_new)
    
    Args:
        k: keys [B, T, H, K]
        v: values (实际上是 u) [B, T, H, V]
        w: weights [B, T, H, K]
        v_new: 新的 values (可选) [B, T, H, V]
        g: 门控值 (可选) [B, T, H]
        gk: key 门控值 (可选) [B, T, H, K]
        h: 输出隐状态 [B, NT, H, K, V]
        h0: 初始状态 (可选) [N, H, K, V]
        ht: 最终状态 (可选) [N, H, K, V]
        cu_seqlens: 累积序列长度 (可选) [N+1]
        chunk_offsets: chunk 偏移量 (可选) [N]
        T: 序列长度
        H: 头数
        K: key 维度
        V: value 维度
        BT: chunk 大小
        BV: value 块大小
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    
    # 处理变长序列
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # 初始化隐状态 (支持 K <= 256，分成最多 4 个 64 维块)
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # 计算偏移量
    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    
    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # 加载初始状态
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # 主循环：遍历所有 chunks
    for i_t in range(NT):
        # 存储当前 chunk 的隐状态
        p_h1 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        # 计算 b_v = v - w @ h
        p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        # 保存新的 value
        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        # 应用门控 g
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
            b_g_last = tl.exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        # 应用 key 门控 gk
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k1, mask=(o_k1 < K), other=0.)
            b_h1 *= tl.exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k2, mask=(o_k2 < K), other=0.)
                b_h2 *= tl.exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k3, mask=(o_k3 < K), other=0.)
                b_h3 *= tl.exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k4, mask=(o_k4 < K), other=0.)
                b_h4 *= tl.exp(b_gk_last4)[:, None]
        
        b_v = b_v.to(k.dtype.element_ty)

        # 更新隐状态：h = h + k @ v
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)

    # 存储最终状态
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Chunk Gated Delta Rule 前向传播 - 计算隐状态 h
    
    Args:
        k: keys [B, T, H, K]
        w: weights [B, T, H, K]
        u: values [B, T, H, V]
        g: 门控值 (可选) [B, T, H]
        gk: key 门控值 (可选) [B, T, H, K]
        initial_state: 初始状态 (可选) [N, H, K, V]
        output_final_state: 是否输出最终状态
        chunk_size: chunk 大小，默认 64
        save_new_value: 是否保存新的 value
        cu_seqlens: 累积序列长度 (可选) [N+1]
        chunk_indices: chunk 索引 (可选)
        
    Returns:
        h: 隐状态 [B, NT, H, K, V]
        v_new: 新的 values [B, T, H, V] (如果 save_new_value=True)
        final_state: 最终状态 [N, H, K, V] (如果 output_final_state=True)
    """
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    
    # N: 实际的序列数量
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    
    assert K <= 256, "当前 kernel 不支持 head dimension 大于 256"

    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None
    
    def grid(meta):
        return (triton.cdiv(V, meta['BV']), N * H)
    
    chunk_gated_delta_rule_fwd_kernel_h[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    
    return h, v_new, final_state


# ============================================================================
# 高层接口函数
# ============================================================================

def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    """
    Chunk Gated Delta Rule 前向传播的完整流程。
    
    这个函数组合了所有必要的步骤：
    1. 局部累积和 (chunk_local_cumsum)
    2. 计算 WY 表示 (chunk_scaled_dot_kkt_fwd, solve_tril, recompute_w_u_fwd)
    3. 计算隐状态 (chunk_gated_delta_rule_fwd_h)
    4. 计算输出 (chunk_fwd_o)
    
    Args:
        q: queries [B, T, H, K]
        k: keys [B, T, H, K]
        v: values [B, T, H, V]
        g: 门控值 (log space) [B, T, H]
        beta: betas [B, T, H]
        scale: 缩放因子
        initial_state: 初始状态 [N, H, K, V]
        output_final_state: 是否输出最终状态
        cu_seqlens: 累积序列长度 (可选) [N+1]
        
    Returns:
        g: 累积后的门控值 [B, T, H]
        o: 输出 [B, T, H, V]
        A: WY 表示的矩阵 A
        final_state: 最终状态 [N, H, K, V] (如果 output_final_state=True)
    """
    # Step 1: 局部累积和
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    
    # Step 2: 计算 WY 表示
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    
    # Step 3: 计算隐状态
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    
    # Step 4: 计算输出
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    
    return g, o, A, final_state

