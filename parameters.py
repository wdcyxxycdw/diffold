#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffold模型参数统计脚本
按照大模块分组统计参数量
"""

import torch
import torch.nn as nn
from collections import defaultdict, OrderedDict
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from diffold.diffold import Diffold
    from rhofold.config import rhofold_config
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保您在项目根目录下运行此脚本")
    sys.exit(1)


def count_parameters(module):
    """统计模块的参数数量"""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def format_number(num):
    """格式化数字，添加单位"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def get_submodule_stats(module, name_prefix=""):
    """递归获取子模块的参数统计"""
    stats = OrderedDict()
    
    for name, submodule in module.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        total, trainable, frozen = count_parameters(submodule)
        
        if total > 0:  # 只记录有参数的模块
            stats[full_name] = {
                'total': total,
                'trainable': trainable,
                'frozen': frozen,
                'module': submodule
            }
    
    return stats


def analyze_diffold_parameters():
    """分析Diffold模型的参数"""
    
    print("🔍 正在初始化Diffold模型...")
    
    try:
        # 创建简单配置
        config = {
            'model_type': 'diffold',
            'device': 'cpu'
        }
        
        # 初始化模型（不加载预训练权重以避免文件不存在的问题）
        model = Diffold(config, rhofold_checkpoint_path=None)
        
        print("✅ 模型初始化成功！\n")
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return
    
    # 统计总参数
    total_params, trainable_params, frozen_params = count_parameters(model)
    
    print("=" * 80)
    print(f"{'Diffold 模型参数统计报告':^80}")
    print("=" * 80)
    print()
    
    # 总体统计
    print("📊 总体参数统计:")
    print(f"  • 总参数量:     {format_number(total_params):>10} ({total_params:,})")
    print(f"  • 可训练参数:   {format_number(trainable_params):>10} ({trainable_params:,})")
    print(f"  • 冻结参数:     {format_number(frozen_params):>10} ({frozen_params:,})")
    print(f"  • 训练参数比例: {trainable_params/total_params*100:>10.1f}%")
    print()
    
    # 按大模块统计
    print("🏗️  大模块参数分布:")
    print("-" * 80)
    print(f"{'模块名称':<25} {'总参数':<12} {'可训练':<12} {'冻结':<12} {'训练占比':<10}")
    print("-" * 80)
    
    main_modules = [
        ("rhofold", "RhoFold骨干网络"),
        ("relative_position_encoding", "相对位置编码"),
        ("input_embedder", "输入特征嵌入器"),
        ("diffusion", "扩散模块"),
        ("edm", "原子扩散模块"),
        ("single_dim_adapter", "维度适配层"),
        ("confidence_head", "置信度预测头")
    ]
    
    module_stats = {}
    
    for module_name, description in main_modules:
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            total, trainable, frozen = count_parameters(module)
            
            if total > 0:
                module_stats[module_name] = {
                    'description': description,
                    'total': total,
                    'trainable': trainable,
                    'frozen': frozen
                }
                
                train_ratio = trainable / total * 100 if total > 0 else 0
                print(f"{description:<25} {format_number(total):<12} {format_number(trainable):<12} {format_number(frozen):<12} {train_ratio:>8.1f}%")
    
    print("-" * 80)
    print()
    
    # RhoFold子模块详细分析
    if hasattr(model, 'rhofold'):
        print("🧬 RhoFold子模块详细分析:")
        print("-" * 60)
        print(f"{'子模块':<20} {'参数量':<12} {'占RhoFold比例':<15}")
        print("-" * 60)
        
        rhofold_total = module_stats.get('rhofold', {}).get('total', 1)
        rhofold_submodules = get_submodule_stats(model.rhofold)
        
        for name, stats in rhofold_submodules.items():
            ratio = stats['total'] / rhofold_total * 100
            print(f"{name:<20} {format_number(stats['total']):<12} {ratio:>12.1f}%")
        
        print("-" * 60)
        print()
    
    # 扩散模块详细分析
    if hasattr(model, 'diffusion'):
        print("🌊 扩散模块详细分析:")
        print("-" * 60)
        print(f"{'子模块':<20} {'参数量':<12} {'状态':<10}")
        print("-" * 60)
        
        diffusion_submodules = get_submodule_stats(model.diffusion)
        
        for name, stats in diffusion_submodules.items():
            status = "可训练" if stats['trainable'] > 0 else "冻结"
            print(f"{name:<20} {format_number(stats['total']):<12} {status:<10}")
        
        print("-" * 60)
        print()
    
    # 训练效率分析
    print("⚡ 训练效率分析:")
    print("-" * 40)
    
    if frozen_params > 0:
        memory_saved = frozen_params * 4 / (1024**3)  # 假设float32，计算GB
        print(f"  • 冻结参数带来的内存节省: ~{memory_saved:.2f} GB")
    
    if trainable_params > 0:
        training_memory = trainable_params * 12 / (1024**3)  # 考虑梯度和优化器状态
        print(f"  • 训练所需显存估计: ~{training_memory:.2f} GB")
    
    print(f"  • 模型推理显存估计: ~{total_params * 4 / (1024**3):.2f} GB")
    print()
    
    # 按参数量排序的模块
    print("📈 模块参数量排行:")
    print("-" * 50)
    
    all_modules = []
    for name, stats in module_stats.items():
        all_modules.append((stats['description'], stats['total']))
    
    # 添加RhoFold子模块
    if hasattr(model, 'rhofold'):
        rhofold_submodules = get_submodule_stats(model.rhofold)
        for name, stats in rhofold_submodules.items():
            all_modules.append((f"RhoFold.{name}", stats['total']))
    
    # 排序并显示前10
    all_modules.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, params) in enumerate(all_modules[:10], 1):
        percentage = params / total_params * 100
        print(f"{i:2d}. {name:<25} {format_number(params):>10} ({percentage:5.1f}%)")
    
    print()
    print("=" * 80)
    print("✅ 参数统计完成！")


def main():
    """主函数"""
    print("🚀 Diffold模型参数统计工具")
    print()
    
    try:
        analyze_diffold_parameters()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
