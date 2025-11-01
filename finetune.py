"""
Diffold微调脚本
专门用于在预训练模型基础上进行微调，支持LoRA
"""

import argparse
import logging
from pathlib import Path
import torch

# 导入训练相关模块
from train import DiffoldTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_finetune_config(args):
    """创建微调配置"""
    # 加载基础配置
    if args.config:
        config = TrainingConfig(args.config)
    else:
        config = TrainingConfig()
    
    # 🔥 强制启用微调模式
    config.finetune['enable_finetuning'] = True
    
    # 设置预训练检查点
    if args.pretrained:
        config.finetune['pretrained_checkpoint'] = args.pretrained
        logger.info(f"📥 预训练模型: {args.pretrained}")
    elif not config.finetune.get('pretrained_checkpoint'):
        raise ValueError("必须指定预训练模型！使用 --pretrained 或在配置文件中设置")
    
    # 设置冻结策略
    if args.freeze_strategy:
        config.finetune['freeze_strategy'] = args.freeze_strategy
    
    # 设置分层学习率
    if args.backbone_lr_ratio is not None:
        config.finetune['learning_rate_scaling']['enable'] = True
        config.finetune['learning_rate_scaling']['backbone_lr_ratio'] = args.backbone_lr_ratio
    
    # 🎯 LoRA配置
    if args.use_lora:
        config.lora['enable'] = True
        if args.lora_r is not None:
            config.lora['r'] = args.lora_r
        if args.lora_alpha is not None:
            config.lora['alpha'] = args.lora_alpha
        if args.lora_dropout is not None:
            config.lora['dropout'] = args.lora_dropout
        if args.lora_strategy:
            config.lora['strategy'] = args.lora_strategy
        logger.info(f"🎯 启用LoRA: rank={config.lora['r']}, strategy={config.lora['strategy']}")
    
    # 覆盖训练参数
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    # 输出目录
    if args.output_dir:
        config.output_dir = args.output_dir
        config.checkpoint_dir = Path(args.output_dir) / "checkpoints"
    
    return config


def print_finetune_summary(config):
    """打印微调配置摘要"""
    print("\n" + "="*60)
    print("🎯 Diffold 微调配置")
    print("="*60)
    
    # 预训练模型
    print(f"\n📦 预训练模型:")
    print(f"  {config.finetune['pretrained_checkpoint']}")
    
    # 冻结策略
    print(f"\n🔒 冻结策略: {config.finetune['freeze_strategy']}")
    
    # LoRA
    if config.lora['enable']:
        print(f"\n🎯 LoRA配置:")
        print(f"  启用: ✅")
        print(f"  Rank: {config.lora['r']}")
        print(f"  Alpha: {config.lora['alpha']}")
        print(f"  Dropout: {config.lora['dropout']}")
        print(f"  策略: {config.lora['strategy']}")
    else:
        print(f"\n🎯 LoRA: ❌ 未启用")
    
    # 学习率
    print(f"\n📊 学习率配置:")
    print(f"  基础学习率: {config.learning_rate}")
    if config.finetune['learning_rate_scaling']['enable']:
        print(f"  分层学习率: ✅")
        print(f"    骨干网络: {config.finetune['learning_rate_scaling']['backbone_lr_ratio']}x")
        print(f"    头部网络: {config.finetune['learning_rate_scaling']['head_lr_ratio']}x")
    else:
        print(f"  分层学习率: ❌")
    
    # 训练参数
    print(f"\n🏃 训练参数:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  验证频率: 每 {config.validate_every} epoch")
    print(f"  Early Stopping: {config.early_stopping_patience} epochs")
    
    # 数据
    print(f"\n📁 数据:")
    print(f"  数据目录: {config.data_dir}")
    print(f"  Fold: {config.fold}")
    
    # 输出
    print(f"\n💾 输出:")
    print(f"  输出目录: {config.output_dir}")
    print(f"  检查点: {config.checkpoint_dir}")
    
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Diffold微调脚本 - 在预训练模型基础上进行微调',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 基本微调（不使用LoRA）
  python finetune.py --pretrained checkpoints/best_model.pt --config config.yaml

  # 使用LoRA微调（推荐用于小数据集）
  python finetune.py --pretrained checkpoints/best_model.pt --use-lora --lora-r 8

  # 自定义冻结策略
  python finetune.py --pretrained checkpoints/best_model.pt \\
                     --freeze-strategy rhofold_only \\
                     --learning-rate 1e-4

  # 完整LoRA微调配置
  python finetune.py --pretrained checkpoints/best_model.pt \\
                     --use-lora --lora-r 8 --lora-strategy diffusion_confidence \\
                     --freeze-strategy rhofold_only \\
                     --learning-rate 1e-4 --epochs 30

数据量推荐:
  - <100样本: --use-lora --lora-r 4 --freeze-strategy rhofold_only
  - 100-500样本: --use-lora --lora-r 8 --freeze-strategy rhofold_backbone
  - 500-1000样本: --use-lora --lora-r 12 --freeze-strategy rhofold_backbone
  - >1000样本: --freeze-strategy none (标准微调)
        """
    )
    
    # 必需参数
    parser.add_argument('--pretrained', type=str,
                       help='预训练模型路径 (必需)')
    parser.add_argument('--config', type=str,
                       help='配置文件路径 (默认: config.yaml)')
    
    # 微调策略
    finetune_group = parser.add_argument_group('微调策略')
    finetune_group.add_argument('--freeze-strategy', type=str,
                               choices=['none', 'rhofold_only', 'rhofold_backbone', 'rhofold_heads'],
                               help='冻结策略')
    finetune_group.add_argument('--backbone-lr-ratio', type=float,
                               help='骨干网络学习率比例 (默认: 0.1)')
    
    # LoRA参数
    lora_group = parser.add_argument_group('LoRA配置')
    lora_group.add_argument('--use-lora', action='store_true',
                           help='启用LoRA微调（推荐用于小数据集）')
    lora_group.add_argument('--lora-r', type=int,
                           help='LoRA rank (默认: 8)')
    lora_group.add_argument('--lora-alpha', type=int,
                           help='LoRA alpha (默认: 等于rank)')
    lora_group.add_argument('--lora-dropout', type=float,
                           help='LoRA dropout (默认: 0.05)')
    lora_group.add_argument('--lora-strategy', type=str,
                           choices=['diffusion_only', 'diffusion_confidence', 'diffusion_all_heads', 'full_model'],
                           help='LoRA应用策略')
    
    # 训练参数
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--learning-rate', '--lr', type=float,
                            help='学习率 (推荐: 1e-4 ~ 1e-5)')
    train_group.add_argument('--epochs', type=int,
                            help='训练轮数')
    train_group.add_argument('--batch-size', type=int,
                            help='Batch大小')
    
    # 输出
    output_group = parser.add_argument_group('输出配置')
    output_group.add_argument('--output-dir', type=str,
                             help='输出目录')
    
    # 其他
    parser.add_argument('--dry-run', action='store_true',
                       help='仅显示配置，不实际训练')
    
    args = parser.parse_args()
    
    # 创建配置
    try:
        config = create_finetune_config(args)
    except Exception as e:
        logger.error(f"❌ 配置错误: {e}")
        parser.print_help()
        return 1
    
    # 打印配置摘要
    print_finetune_summary(config)
    
    # Dry run模式
    if args.dry_run:
        logger.info("🔍 Dry run模式，不执行训练")
        return 0
    
    # 确认开始训练
    logger.info("🚀 开始微调...")
    
    # 创建训练器并开始训练
    try:
        trainer = DiffoldTrainer(config)
        trainer.train()
        logger.info("✅ 微调完成！")
        return 0
    except KeyboardInterrupt:
        logger.info("\n⚠️ 训练被用户中断")
        return 1
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

