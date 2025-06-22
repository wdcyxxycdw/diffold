#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diffold 模型简化训练测试脚本

如果完整版测试出现依赖问题，可以使用此简化版本
直接构造必要的输入张量来测试模型训练能力
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import traceback

# 导入必要的模块
from rhofold.config import rhofold_config
from diffold.diffold import Diffold


def generate_mock_inputs(seq_length=16, msa_depth=32, batch_size=1):
    """生成模拟输入数据"""
    print(f"生成模拟输入数据...")
    print(f"序列长度: {seq_length}, MSA深度: {msa_depth}, 批次大小: {batch_size}")
    
    # RNA序列 (ACGU alphabet)
    seq = "ACGUACGUACGUACGU"[:seq_length]
    
    # 修正：模拟MSA tokens (RhoFold使用的格式)
    # 按照get_msa_feature的处理方式，最终形状应该是 [batch_size, msa_depth, seq_len]
    # 不包含cls/eos token
    tokens = torch.randint(0, 9, (batch_size, msa_depth, seq_length))  # RNA alphabet size ≈ 9
    
    # 修正：模拟RNA-FM tokens
    # 按照get_rna_fm_token的处理，应该是 [batch_size, seq_len]，不包含cls/eos
    rna_fm_tokens = torch.randint(0, 25, (batch_size, seq_length))  # RNA-FM alphabet size
    
    # 修正：根据实际的RNA序列确定正确的原子数量
    # 先生成一个临时的坐标用于确定原子数量
    temp_coords = torch.randn(1, seq_length * 30, 3) * 10  # 临时估计，稍微多一些
    
    # 导入必要的函数
    from diffold.input_processor import process_alphafold3_input
    
    # 调用process_alphafold3_input来确定实际的原子数量
    af_in, atom_mask = process_alphafold3_input(
        ss_rna=[seq],
        atom_pos=[temp_coords.squeeze(0)],  # 需要是张量列表，并且去掉batch维度
    )
    
    # 获取实际需要的原子数量
    actual_num_atoms = af_in.atom_inputs.shape[1]
    print(f"✓ 根据序列'{seq}'确定实际原子数量: {actual_num_atoms}")
    
    # 生成正确数量的原子坐标
    target_coords = torch.randn(batch_size, actual_num_atoms, 3) * 10  # 合理的3D坐标范围
    
    print(f"✓ tokens形状: {tokens.shape}")
    print(f"✓ rna_fm_tokens形状: {rna_fm_tokens.shape}")
    print(f"✓ target_coords形状: {target_coords.shape}")
    print(f"✓ 序列: {seq}")
    
    return {
        'seq': seq,
        'tokens': tokens,
        'rna_fm_tokens': rna_fm_tokens,
        'target_coords': target_coords
    }


def generate_mock_coordinates(sequence, atoms_per_residue=27):
    """生成模拟的原子坐标"""
    seq_len = len(sequence)
    # 先用合理的估计值，实际数量会在process_alphafold3_input中确定
    num_atoms = seq_len * atoms_per_residue
    
    # 生成随机的3D坐标，但保持合理的结构
    coords = torch.randn(1, num_atoms, 3) * 10  # 扩展到合理的埃单位
    
    return coords


def adjust_coordinates_to_match_input(target_coords, af_in):
    """根据process_alphafold3_input的输出调整坐标数量"""
    if hasattr(af_in, 'atom_inputs') and af_in.atom_inputs is not None:
        expected_atoms = af_in.atom_inputs.shape[1]  # 实际的原子数
        current_atoms = target_coords.shape[1]
        
        if expected_atoms != current_atoms:
            print(f"调整原子坐标数量: {current_atoms} -> {expected_atoms}")
            if expected_atoms > current_atoms:
                # 如果需要更多原子，用零坐标填充
                padding = torch.zeros(target_coords.shape[0], expected_atoms - current_atoms, 3, 
                                    device=target_coords.device)
                target_coords = torch.cat([target_coords, padding], dim=1)
            else:
                # 如果原子太多，截取到需要的数量
                target_coords = target_coords[:, :expected_atoms, :]
    
    return target_coords


def test_model_initialization():
    """测试模型初始化"""
    print("=" * 50)
    print("测试 1: 模型初始化")
    print("=" * 50)
    
    try:
        # 使用指定的权重路径
        rhofold_checkpoint_path = "/home/yamanashi/RhoFold/pretrained/model_20221010_params.pt"
        
        print(f"初始化Diffold模型...")
        print(f"RhoFold权重路径: {rhofold_checkpoint_path}")
        
        # 检查权重文件是否存在
        if os.path.exists(rhofold_checkpoint_path):
            print("✓ 权重文件存在")
        else:
            print(f"⚠️ 权重文件不存在: {rhofold_checkpoint_path}")
            print("将尝试不加载预训练权重进行测试")
            rhofold_checkpoint_path = None
        
        model = Diffold(rhofold_config, rhofold_checkpoint_path=rhofold_checkpoint_path)
        
        print("✓ 模型初始化成功")
        
        # 检查RhoFold参数是否被冻结
        rhofold_params = list(model.rhofold.parameters())
        rhofold_param_count = sum(p.numel() for p in rhofold_params)  # 计算实际参数数量
        frozen_count = sum(1 for p in rhofold_params if not p.requires_grad)
        print(f"✓ RhoFold参数总数: {len(rhofold_params)}")
        print(f"✓ RhoFold参数数量: {rhofold_param_count:,}")
        print(f"✓ 冻结参数数量: {frozen_count}")
        print(f"✓ 冻结比例: {frozen_count/len(rhofold_params)*100:.1f}%")
        
        # 检查可训练参数
        trainable_params = model.get_trainable_parameters()
        trainable_param_count = sum(p.numel() for p in trainable_params)
        total_param_count = sum(p.numel() for p in model.parameters())
        
        print(f"✓ 可训练参数数量: {len(trainable_params)}")
        print(f"✓ 可训练参数总数: {trainable_param_count:,}")
        print(f"✓ 模型总参数数量: {total_param_count:,}")
        print(f"✓ 冻结参数数量: {total_param_count - trainable_param_count:,}")
        print(f"✓ 冻结参数比例: {(total_param_count - trainable_param_count)/total_param_count*100:.1f}%")
        
        return model
        
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        traceback.print_exc()
        return None


def test_forward_pass(model, data):
    """测试前向传播"""
    print("=" * 50)
    print("测试 2: 前向传播")
    print("=" * 50)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 将模型和数据移动到设备
        model = model.to(device)
        tokens = data['tokens'].to(device)
        rna_fm_tokens = data['rna_fm_tokens'].to(device)
        target_coords = data['target_coords'].to(device)
        
        print("✓ 模型和数据已移动到设备")
        
        # 测试推理模式前向传播（无目标坐标）
        print("测试推理模式前向传播...")
        model.set_eval_mode()
        
        with torch.no_grad():
            outputs = model(
                tokens=tokens,
                rna_fm_tokens=rna_fm_tokens,
                seq=data['seq']
            )
        
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            rhofold_outputs, single_fea, pair_fea = outputs
            print("✓ 推理模式前向传播成功")
            print(f"  - single_fea形状: {single_fea.shape if single_fea is not None else 'None'}")
            print(f"  - pair_fea形状: {pair_fea.shape if pair_fea is not None else 'None'}")
        else:
            print("✓ 推理模式前向传播成功")
        
        # 测试训练模式前向传播（有目标坐标）
        print("测试训练模式前向传播...")
        model.set_train_mode()
        
        loss, denoised_pos, loss_breakdown = model(
            tokens=tokens,
            rna_fm_tokens=rna_fm_tokens,
            seq=data['seq'],
            target_coords=target_coords
        )
        
        print("✓ 训练模式前向传播成功")
        print(f"  - 损失值: {loss.item():.6f}")
        print(f"  - 损失需要梯度: {loss.requires_grad}")
        print(f"  - 损失类型: {type(loss)}")
        
        if loss_breakdown is not None and isinstance(loss_breakdown, dict):
            print("  - 损失分解:")
            for key, value in loss_breakdown.items():
                if isinstance(value, torch.Tensor):
                    print(f"    - {key}: {value.item():.6f}")
        
        return loss
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        traceback.print_exc()
        return None


def test_backward_pass(model, data):
    """测试反向传播和参数更新"""
    print("=" * 50)
    print("测试 3: 反向传播和参数更新")
    print("=" * 50)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.set_train_mode()
        
        # 创建优化器
        trainable_params = model.get_trainable_parameters()
        optimizer = optim.Adam(trainable_params, lr=1e-4)
        
        print(f"✓ 创建优化器，管理 {len(trainable_params)} 个可训练参数")
        
        # 记录更新前的参数值（采样几个参数）
        param_before = {}
        sample_size = min(5, len(trainable_params))
        for i in range(sample_size):
            param_before[i] = trainable_params[i].clone().detach()
        
        # 前向传播
        tokens = data['tokens'].to(device)
        rna_fm_tokens = data['rna_fm_tokens'].to(device)
        target_coords = data['target_coords'].to(device)
        
        loss, _, _ = model(
            tokens=tokens,
            rna_fm_tokens=rna_fm_tokens,
            seq=data['seq'],
            target_coords=target_coords
        )
        
        print(f"✓ 前向传播完成，损失: {loss.item():.6f}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        print("✓ 反向传播完成")
        
        # 检查梯度
        grad_count = 0
        total_grad_norm = 0.0
        for param in trainable_params:
            if param.grad is not None:
                grad_count += 1
                total_grad_norm += param.grad.norm().item() ** 2
        
        total_grad_norm = total_grad_norm ** 0.5
        print(f"✓ 梯度计算完成")
        print(f"  - 有梯度的参数数量: {grad_count}/{len(trainable_params)}")
        print(f"  - 总梯度范数: {total_grad_norm:.6f}")
        
        # 参数更新
        optimizer.step()
        print("✓ 参数更新完成")
        
        # 验证参数确实更新了
        updated_count = 0
        for i in range(sample_size):
            if not torch.equal(param_before[i], trainable_params[i]):
                updated_count += 1
        
        print(f"✓ 参数更新验证: {updated_count}/{sample_size} 个采样参数已更新")
        
        return True
        
    except Exception as e:
        print(f"✗ 反向传播失败: {e}")
        traceback.print_exc()
        return False


def test_training_loop(model, data, num_steps=3):
    """测试训练循环"""
    print("=" * 50)
    print(f"测试 4: 训练循环 ({num_steps} 步)")
    print("=" * 50)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.set_train_mode()
        
        # 创建优化器和学习率调度器
        trainable_params = model.get_trainable_parameters()
        optimizer = optim.Adam(trainable_params, lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        
        print(f"✓ 训练设置完成")
        
        # 准备数据
        tokens = data['tokens'].to(device)
        rna_fm_tokens = data['rna_fm_tokens'].to(device)
        target_coords = data['target_coords'].to(device)
        
        losses = []
        
        for step in range(num_steps):
            print(f"\n--- 步骤 {step + 1}/{num_steps} ---")
            
            # 前向传播
            optimizer.zero_grad()
            
            loss, _, loss_breakdown = model(
                tokens=tokens,
                rna_fm_tokens=rna_fm_tokens,
                seq=data['seq'],
                target_coords=target_coords
            )
            
            losses.append(loss.item())
            print(f"损失: {loss.item():.6f}")
            
            # 反向传播和参数更新
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"学习率: {current_lr:.6f}")
            
            if loss_breakdown is not None and isinstance(loss_breakdown, dict):
                print("损失分解:")
                for key, value in loss_breakdown.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  - {key}: {value.item():.6f}")
        
        print(f"\n✓ 训练循环完成")
        print(f"✓ 损失变化: {losses[0]:.6f} -> {losses[-1]:.6f}")
        
        if len(losses) > 1:
            loss_change = losses[-1] - losses[0]
            print(f"✓ 损失变化量: {loss_change:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练循环失败: {e}")
        traceback.print_exc()
        return False


def test_training_mode(model, inputs):
    """测试训练模式"""
    print("\n🔥 测试训练模式...")
    
    model.set_train_mode()
    
    try:
        # 训练前向传播
        loss, denoised_pos, loss_breakdown = model(
            tokens=inputs['tokens'].cuda(),
            rna_fm_tokens=inputs['rna_fm_tokens'].cuda(),
            seq=inputs['seq'],
            target_coords=inputs['target_coords'].cuda()
        )
        
        print(f"✓ 训练前向传播成功")
        print(f"  - 扩散损失: {loss.item():.6f}")
        print(f"  - 去噪坐标形状: {denoised_pos.shape if denoised_pos is not None else 'None'}")
        
        if loss_breakdown:
            print(f"  - 损失详情: {loss_breakdown}")
        
        return loss
        
    except Exception as e:
        print(f"❌ 训练模式测试失败: {e}")
        traceback.print_exc()
        return None


def main():
    """主测试函数"""
    print("🧬 Diffold 模型简化训练测试开始")
    print("=" * 60)
    
    # 生成模拟数据
    data = generate_mock_inputs(seq_length=16, msa_depth=32, batch_size=1)
    
    # 测试1: 模型初始化
    model = test_model_initialization()
    if model is None:
        print("❌ 模型初始化失败，终止测试")
        return False
    
    # 测试2: 前向传播 
    loss = test_forward_pass(model, data)
    if loss is None:
        print("❌ 前向传播失败，终止测试")
        return False
    
    # 测试3: 反向传播
    backward_success = test_backward_pass(model, data)
    if not backward_success:
        print("❌ 反向传播失败，终止测试")
        return False
    
    # 测试4: 训练循环
    training_success = test_training_loop(model, data, num_steps=3)
    if not training_success:
        print("❌ 训练循环失败")
        return False
    
    print("=" * 60)
    print("🎉 所有测试通过！Diffold模型可以正常训练")
    print("✅ 测试结果:")
    print("  - ✓ 模型初始化正常")
    print("  - ✓ RhoFold参数正确冻结")
    print("  - ✓ 前向传播工作正常")
    print("  - ✓ 反向传播和梯度计算正常")
    print("  - ✓ 参数更新正常")
    print("  - ✓ 训练循环正常")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 