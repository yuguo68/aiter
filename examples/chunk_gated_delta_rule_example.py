#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
示例脚本：演示如何使用 aiter 中的 chunk_gated_delta_rule 函数

这个示例展示了如何使用 chunk 模式进行 gated delta rule 计算，
适用于训练和长序列场景。
"""

import torch
import torch.nn.functional as F

try:
    from aiter.ops.triton.gated_delta_rule import chunk_gated_delta_rule
    print("✓ 成功导入 chunk_gated_delta_rule")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("请确保已安装 flash-linear-attention:")
    print("  pip install flash-linear-attention")
    exit(1)


def example_basic():
    """基础示例：等长序列"""
    print("\n" + "="*60)
    print("示例 1: 基础用法 (等长序列)")
    print("="*60)
    
    # 设置参数
    B, T, H, K, V = 2, 1024, 4, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"批大小: {B}, 序列长度: {T}, 头数: {H}, K维度: {K}, V维度: {V}")
    print(f"设备: {device}")
    
    # 创建输入张量
    q = torch.randn(B, T, H, K, device=device)
    k = F.normalize(torch.randn(B, T, H, K, device=device), p=2, dim=-1)
    v = torch.randn(B, T, H, V, device=device)
    beta = torch.rand(B, T, H, device=device).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, device=device))
    h0 = torch.randn(B, H, K, V, device=device)
    
    print(f"\n输入形状:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print(f"  beta: {beta.shape}")
    print(f"  g: {g.shape}")
    print(f"  h0: {h0.shape}")
    
    # 执行计算
    print("\n正在计算...")
    o, ht = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
    )
    
    print(f"\n输出形状:")
    print(f"  o: {o.shape}")
    print(f"  ht: {ht.shape}")
    print(f"\n✓ 计算成功完成!")


def example_varlen():
    """变长序列示例"""
    print("\n" + "="*60)
    print("示例 2: 变长序列")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    H, K, V = 4, 64, 64
    
    # 创建变长序列的累积长度
    cu_seqlens = torch.tensor([0, 256, 768, 1536, 2048], dtype=torch.long, device=device)
    T = cu_seqlens[-1].item()
    N = len(cu_seqlens) - 1
    
    print(f"序列数: {N}")
    print(f"总长度: {T}")
    print(f"累积长度: {cu_seqlens.tolist()}")
    print(f"各序列长度: {[cu_seqlens[i+1] - cu_seqlens[i] for i in range(N)]}")
    
    # 创建输入张量 (注意：batch size 必须为 1)
    q = torch.randn(1, T, H, K, device=device)
    k = F.normalize(torch.randn(1, T, H, K, device=device), p=2, dim=-1)
    v = torch.randn(1, T, H, V, device=device)
    beta = torch.rand(1, T, H, device=device).sigmoid()
    g = F.logsigmoid(torch.rand(1, T, H, device=device))
    h0 = torch.randn(N, H, K, V, device=device)  # N 个初始状态
    
    print(f"\n输入形状:")
    print(f"  q: {q.shape} (batch=1 用于变长)")
    print(f"  h0: {h0.shape} (N 个初始状态)")
    
    # 执行计算
    print("\n正在计算...")
    o, ht = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    
    print(f"\n输出形状:")
    print(f"  o: {o.shape}")
    print(f"  ht: {ht.shape} (N 个最终状态)")
    print(f"\n✓ 变长序列计算成功完成!")


def example_with_l2norm():
    """使用 QK L2 归一化的示例"""
    print("\n" + "="*60)
    print("示例 3: 使用 QK L2 归一化")
    print("="*60)
    
    B, T, H, K, V = 2, 512, 4, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"批大小: {B}, 序列长度: {T}, 头数: {H}")
    print("启用 kernel 内 L2 归一化")
    
    # 创建输入张量 (不需要手动归一化 q 和 k)
    q = torch.randn(B, T, H, K, device=device)
    k = torch.randn(B, T, H, K, device=device)  # 不需要归一化
    v = torch.randn(B, T, H, V, device=device)
    beta = torch.rand(B, T, H, device=device).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, device=device))
    
    # 执行计算，启用 kernel 内 L2 归一化
    print("\n正在计算...")
    o, _ = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        use_qk_l2norm_in_kernel=True,
        output_final_state=False,
    )
    
    print(f"\n输出形状: {o.shape}")
    print(f"✓ 使用 kernel 内 L2 归一化计算成功!")


def main():
    print("\n" + "="*60)
    print("Aiter Chunk Gated Delta Rule 示例")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("\n警告: CUDA 不可用，将在 CPU 上运行（可能较慢）")
    
    try:
        # 运行所有示例
        example_basic()
        example_varlen()
        example_with_l2norm()
        
        print("\n" + "="*60)
        print("所有示例执行成功! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

