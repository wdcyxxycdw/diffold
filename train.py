#!/usr/bin/env python3
"""
Diffold模型训练脚本
包含：断点保存、进度显示、训练曲线绘制、小规模测试等功能
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 导入模型和数据模块
from diffold.diffold import Diffold
from diffold.dataloader import RNA3DDataLoader
from rhofold.config import rhofold_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """训练配置类"""
    
    def __init__(self):
        # 数据配置
        self.data_dir = "./processed_data"
        self.batch_size = 4
        self.max_sequence_length = 256
        self.num_workers = 4
        self.use_msa = True
        
        # 模型配置
        self.rhofold_checkpoint = "./pretrained/model_20221010_params.pt"
        
        # 训练配置
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.grad_clip_norm = 1.0
        self.warmup_epochs = 5
        
        # 调度器配置
        self.scheduler_type = "cosine"  # "cosine", "plateau"
        self.patience = 10
        
        # 保存配置
        self.output_dir = "./output"
        self.checkpoint_dir = "./checkpoints"
        self.save_every = 5
        self.keep_last_n_checkpoints = 5
        
        # 验证配置
        self.validate_every = 1
        self.early_stopping_patience = 20
        
        # 设备配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mixed_precision = True
        
        # 多GPU配置
        self.use_data_parallel = True  # 是否使用DataParallel
        self.gpu_ids = None  # 指定使用的GPU ID列表，None表示使用所有可用GPU
        
        # 小规模测试配置
        self.test_mode = False
        self.test_samples = 10
        self.test_epochs = 5
        
        # 交叉验证配置
        self.fold = 0
        self.num_folds = 10
        self.use_all_folds = False  # 新增：是否使用所有折数


class TrainingMetrics:
    """训练指标记录类"""
    
    def __init__(self):
        self.train_losses = []
        self.valid_losses = []
        self.learning_rates = []
        self.epoch_times = []
        
        self.best_valid_loss = float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
    
    def update_train(self, loss: float, lr: float, epoch_time: float):
        """更新训练指标"""
        self.train_losses.append(loss)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
    
    def update_valid(self, loss: float, epoch: int):
        """更新验证指标"""
        self.valid_losses.append(loss)
        
        if loss < self.best_valid_loss:
            self.best_valid_loss = loss
            self.best_epoch = epoch
            self.early_stopping_counter = 0
            return True  # 找到更好的模型
        else:
            self.early_stopping_counter += 1
            return False
    
    def to_dict(self):
        """转换为字典"""
        return {
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'best_valid_loss': self.best_valid_loss,
            'best_epoch': self.best_epoch,
            'early_stopping_counter': self.early_stopping_counter
        }


class DiffoldTrainer:
    """Diffold模型训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics = TrainingMetrics()
        
        # 创建输出目录
        self.setup_directories()
        
        # 设置日志
        self.setup_logging()
        
        # 初始化设备
        self.device = torch.device(config.device)
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = self.setup_model()
        
        # 初始化数据加载器
        self.setup_data_loaders()
        
        # 初始化优化器和调度器
        self.setup_optimizer_and_scheduler()
        
        # 初始化tensorboard
        self.writer = SummaryWriter(log_dir=self.config.output_dir / "tensorboard")
        
        # 混合精度训练
        if self.config.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("启用混合精度训练")
        else:
            self.scaler = None
        
        # 记录开始时间
        self.start_time = time.time()
        
    def setup_directories(self):
        """创建必要的目录"""
        self.config.output_dir = Path(self.config.output_dir)
        self.config.checkpoint_dir = Path(self.config.checkpoint_dir)
        
        self.config.output_dir.mkdir(exist_ok=True)
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.config.output_dir / "plots").mkdir(exist_ok=True)
        (self.config.output_dir / "tensorboard").mkdir(exist_ok=True)
        
    def setup_logging(self):
        """设置日志记录"""
        # 添加文件日志处理器
        log_file = self.config.output_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info("="*60)
        logger.info("开始Diffold模型训练")
        logger.info("="*60)
        
    def setup_model(self) -> nn.Module:
        """初始化模型"""
        logger.info("初始化Diffold模型...")
        
        # 使用RhoFold配置
        model = Diffold(
            config=rhofold_config,
            rhofold_checkpoint_path=self.config.rhofold_checkpoint
        )
        
        # 移动到设备
        model = model.to(self.device)
        
        # 设置训练模式
        model.set_train_mode()
        
        # 多GPU设置
        if (self.config.use_data_parallel and 
            self.device.type == 'cuda' and 
            torch.cuda.device_count() > 1):
            
            # 检测可用GPU
            available_gpus = torch.cuda.device_count()
            logger.info(f"检测到 {available_gpus} 个GPU")
            
            # 确定使用的GPU
            if self.config.gpu_ids is None:
                gpu_ids = list(range(available_gpus))
            else:
                gpu_ids = self.config.gpu_ids
                # 验证GPU ID的有效性
                for gpu_id in gpu_ids:
                    if gpu_id >= available_gpus:
                        raise ValueError(f"GPU ID {gpu_id} 不存在，只有 {available_gpus} 个GPU")
            
            logger.info(f"使用GPU: {gpu_ids}")
            
            # 包装模型为DataParallel
            model = nn.DataParallel(model, device_ids=gpu_ids)
            
            # 更新有效batch size
            effective_batch_size = self.config.batch_size * len(gpu_ids)
            logger.info(f"DataParallel启用：每GPU batch_size={self.config.batch_size}, 有效batch_size={effective_batch_size}")
            
            # 标记是否使用了DataParallel
            self.using_data_parallel = True
            self.num_gpus = len(gpu_ids)
        else:
            if self.device.type == 'cuda':
                logger.info("使用单GPU训练")
            else:
                logger.info("使用CPU训练")
            self.using_data_parallel = False
            self.num_gpus = 1
        
        # 获取可训练参数统计
        if self.using_data_parallel:
            trainable_params = model.module.get_trainable_parameters()
        else:
            trainable_params = model.get_trainable_parameters()
        
        total_params = sum(p.numel() for p in trainable_params)
        logger.info(f"可训练参数数量: {total_params:,}")
        
        return model
    
    def setup_data_loaders(self):
        """初始化数据加载器"""
        logger.info("初始化数据加载器...")
        
        if self.config.test_mode:
            # 测试模式：保持用户设置的batch_size，但限制最大序列长度
            batch_size = self.config.batch_size
            max_length = min(128, self.config.max_sequence_length)
        else:
            batch_size = self.config.batch_size
            max_length = self.config.max_sequence_length
        
        data_loader = RNA3DDataLoader(
            data_dir=self.config.data_dir,
            batch_size=batch_size,
            max_length=max_length,
            use_msa=self.config.use_msa,
            num_workers=self.config.num_workers,
            force_reload=False,
            enable_missing_atom_mask=True
        )
        
        if self.config.use_all_folds:
            # 使用所有折数的数据
            logger.info("使用所有折数的训练数据...")
            all_loaders = data_loader.get_all_folds_dataloaders()
            
            # 合并所有折数的训练数据加载器
            from torch.utils.data import ConcatDataset, DataLoader
            from diffold.dataloader import collate_fn
            
            all_train_datasets = []
            all_valid_datasets = []
            
            for fold, loaders in all_loaders.items():
                train_dataset = loaders['train'].dataset
                valid_dataset = loaders['valid'].dataset
                all_train_datasets.append(train_dataset)
                all_valid_datasets.append(valid_dataset)
                
                # 安全获取数据集大小
                if hasattr(train_dataset, '__len__') and hasattr(valid_dataset, '__len__'):
                    train_size = len(train_dataset)  # type: ignore
                    valid_size = len(valid_dataset)  # type: ignore
                    logger.info(f"Fold {fold}: 训练样本 {train_size}, 验证样本 {valid_size}")
                else:
                    logger.info(f"Fold {fold}: 已添加到合并数据集")
            
            # 合并数据集
            combined_train_dataset = ConcatDataset(all_train_datasets)
            combined_valid_dataset = ConcatDataset(all_valid_datasets)
            
            # 创建新的数据加载器
            self.train_loader = DataLoader(
                combined_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
            
            self.valid_loader = DataLoader(
                combined_valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
            
            logger.info(f"合并后训练样本数: {len(combined_train_dataset)}")
            logger.info(f"合并后验证样本数: {len(combined_valid_dataset)}")
            
        else:
            # 使用单个折数的数据
            self.train_loader = data_loader.get_train_dataloader(fold=self.config.fold)
            self.valid_loader = data_loader.get_valid_dataloader(fold=self.config.fold)
            
            # 安全获取数据集大小
            if hasattr(self.train_loader.dataset, '__len__'):
                train_size = len(self.train_loader.dataset)  # type: ignore
                logger.info(f"Fold {self.config.fold} - 训练样本数: {train_size}")
            
            if hasattr(self.valid_loader.dataset, '__len__'):
                valid_size = len(self.valid_loader.dataset)  # type: ignore
                logger.info(f"Fold {self.config.fold} - 验证样本数: {valid_size}")
        
        if self.config.test_mode:
            logger.info(f"测试模式：限制为前{self.config.test_samples}个样本")
    
    def setup_optimizer_and_scheduler(self):
        """初始化优化器和学习率调度器"""
        # 获取可训练参数
        if self.using_data_parallel:
            trainable_params = self.model.module.get_trainable_parameters()
        else:
            trainable_params = self.model.get_trainable_parameters()
        
        # 优化器
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        if self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.patience,
                verbose=True
            )
        else:
            self.scheduler = None
        
        logger.info(f"优化器: AdamW (lr={self.config.learning_rate})")
        logger.info(f"调度器: {self.config.scheduler_type}")
    
    def train_one_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.set_train_mode()
        
        total_loss = 0.0
        num_batches = 0
        
        # 测试模式下限制batch数量
        if self.config.test_mode:
            max_batches = min(self.config.test_samples // self.config.batch_size + 1, 5)
        else:
            max_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=min(max_batches, len(self.train_loader)),
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            leave=False
        )
        
        for batch_idx, batch in progress_bar:
            if self.config.test_mode and batch_idx >= max_batches:
                break
                
            try:
                loss = self.train_step(batch)
                
                if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.6f}",
                        'avg_loss': f"{total_loss/num_batches:.6f}"
                    })
                else:
                    logger.warning(f"Batch {batch_idx}: 无效损失，跳过")
                    
            except Exception as e:
                logger.warning(f"Batch {batch_idx} 训练失败: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def train_step(self, batch: Dict) -> Optional[torch.Tensor]:
        """执行一个训练步骤"""
        # 数据移动到设备
        tokens = batch['tokens'].to(self.device)
        sequences = batch['sequences']
        coordinates = batch.get('coordinates', None)
        missing_atom_masks = batch.get('missing_atom_masks', None)
        
        if coordinates is not None:
            coordinates = coordinates.to(self.device)
        if missing_atom_masks is not None:
            missing_atom_masks = missing_atom_masks.to(self.device)
        
        # rna_fm_tokens处理
        rna_fm_tokens = batch.get('rna_fm_tokens', None)
        if rna_fm_tokens is not None:
            rna_fm_tokens = rna_fm_tokens.to(self.device)
        
        self.optimizer.zero_grad()
        
        # 前向传播
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                result = self.model(
                    tokens=tokens,
                    rna_fm_tokens=rna_fm_tokens,
                    seq=sequences,
                    target_coords=coordinates,
                    missing_atom_mask=missing_atom_masks
                )
        else:
            result = self.model(
                tokens=tokens,
                rna_fm_tokens=rna_fm_tokens,
                seq=sequences,
                target_coords=coordinates,
                missing_atom_mask=missing_atom_masks
            )
        
        # 处理模型输出
        if result is None:
            return None
        
        # 新的字典格式返回值
        if isinstance(result, dict):
            loss = result.get('loss', None)
            if loss is None:
                return None
        elif isinstance(result, tuple):
            # 兼容旧格式
            loss = result[0]
        else:
            loss = result
        
        # 反向传播
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.using_data_parallel:
                torch.nn.utils.clip_grad_norm_(self.model.module.get_trainable_parameters(), self.config.grad_clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.using_data_parallel:
                torch.nn.utils.clip_grad_norm_(self.model.module.get_trainable_parameters(), self.config.grad_clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
        
        return loss
    
    def validate(self, epoch: int) -> float:
        """验证模型"""
        self.model.set_eval_mode()
        
        total_loss = 0.0
        num_batches = 0
        
        # 测试模式下限制batch数量
        if self.config.test_mode:
            max_batches = min(3, len(self.valid_loader))
        else:
            max_batches = len(self.valid_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(self.valid_loader),
                total=min(max_batches, len(self.valid_loader)),
                desc="验证",
                leave=False
            )
            
            for batch_idx, batch in progress_bar:
                if self.config.test_mode and batch_idx >= max_batches:
                    break
                    
                try:
                    # 数据移动到设备
                    tokens = batch['tokens'].to(self.device)
                    sequences = batch['sequences']
                    coordinates = batch.get('coordinates', None)
                    missing_atom_masks = batch.get('missing_atom_masks', None)
                    
                    if coordinates is not None:
                        coordinates = coordinates.to(self.device)
                    if missing_atom_masks is not None:
                        missing_atom_masks = missing_atom_masks.to(self.device)
                    
                    # rna_fm_tokens处理
                    rna_fm_tokens = batch.get('rna_fm_tokens', None)
                    if rna_fm_tokens is not None:
                        rna_fm_tokens = rna_fm_tokens.to(self.device)
                    
                    # 前向传播
                    result = self.model(
                        tokens=tokens,
                        rna_fm_tokens=rna_fm_tokens,
                        seq=sequences,
                        target_coords=coordinates,
                        missing_atom_mask=missing_atom_masks
                    )
                    
                    if result is not None:
                        # 新的字典格式返回值
                        if isinstance(result, dict):
                            loss = result.get('loss', None)
                        elif isinstance(result, tuple):
                            # 兼容旧格式
                            loss = result[0]
                        else:
                            loss = result
                        
                        if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                            total_loss += loss.item()
                            num_batches += 1
                            
                            progress_bar.set_postfix({'val_loss': f"{loss.item():.6f}"})
                
                except Exception as e:
                    logger.warning(f"验证 Batch {batch_idx} 失败: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        # 处理DataParallel的state_dict
        if self.using_data_parallel:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': self.metrics.to_dict(),
            'config': self.config.__dict__,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'using_data_parallel': self.using_data_parallel,
            'num_gpus': self.num_gpus,
        }
        
        # 保存最新检查点
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")
        
        # 清理旧检查点
        self.cleanup_old_checkpoints()
        
        logger.info(f"保存检查点: {checkpoint_path}")
    
    def cleanup_old_checkpoints(self):
        """清理旧的检查点文件"""
        checkpoint_files = list(self.config.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoint_files) > self.config.keep_last_n_checkpoints:
            # 按修改时间排序
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
            # 删除最旧的文件
            for old_file in checkpoint_files[:-self.config.keep_last_n_checkpoints]:
                old_file.unlink()
                logger.debug(f"删除旧检查点: {old_file}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """加载检查点"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return 0
        
        logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型状态（处理DataParallel）
        model_state_dict = checkpoint['model_state_dict']
        if self.using_data_parallel:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 恢复训练指标
        if 'metrics' in checkpoint:
            metrics_dict = checkpoint['metrics']
            self.metrics.train_losses = metrics_dict.get('train_losses', [])
            self.metrics.valid_losses = metrics_dict.get('valid_losses', [])
            self.metrics.learning_rates = metrics_dict.get('learning_rates', [])
            self.metrics.epoch_times = metrics_dict.get('epoch_times', [])
            self.metrics.best_valid_loss = metrics_dict.get('best_valid_loss', float('inf'))
            self.metrics.best_epoch = metrics_dict.get('best_epoch', 0)
            self.metrics.early_stopping_counter = metrics_dict.get('early_stopping_counter', 0)
        
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"从epoch {start_epoch}继续训练")
        return start_epoch
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.metrics.train_losses:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        epochs = range(1, len(self.metrics.train_losses) + 1)
        axes[0, 0].plot(epochs, self.metrics.train_losses, 'b-', label='Training Loss')
        if self.metrics.valid_losses:
            valid_epochs = range(1, len(self.metrics.valid_losses) + 1)
            axes[0, 0].plot(valid_epochs, self.metrics.valid_losses, 'r-', label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 学习率曲线
        if self.metrics.learning_rates:
            axes[0, 1].plot(epochs, self.metrics.learning_rates, 'g-')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].grid(True)
        
        # 训练时间
        if self.metrics.epoch_times:
            axes[1, 0].plot(epochs, self.metrics.epoch_times, 'orange')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].set_title('Epoch Training Time')
            axes[1, 0].grid(True)
        
        # 损失分布（最近10个epoch）
        if len(self.metrics.train_losses) > 1:
            recent_losses = self.metrics.train_losses[-10:]
            axes[1, 1].hist(recent_losses, bins=min(10, len(recent_losses)), alpha=0.7, color='blue')
            axes[1, 1].set_xlabel('Loss')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Recent Training Loss Distribution')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = self.config.output_dir / "plots" / f"training_curves_epoch_{len(self.metrics.train_losses)}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves: {plot_path}")
    
    def train(self, resume_from: Optional[str] = None):
        """主训练循环"""
        logger.info("开始训练...")
        
        # 加载检查点（如果指定）
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        # 训练循环
        num_epochs = self.config.test_epochs if self.config.test_mode else self.config.num_epochs
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss = self.train_one_epoch(epoch)
            
            # 验证
            valid_loss = None
            if epoch % self.config.validate_every == 0:
                valid_loss = self.validate(epoch)
            
            # 更新学习率
            if self.scheduler is not None:
                if self.config.scheduler_type == "plateau" and valid_loss is not None:
                    self.scheduler.step(valid_loss)
                elif self.config.scheduler_type == "cosine":
                    self.scheduler.step()
            
            # 记录指标
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time
            
            self.metrics.update_train(train_loss, current_lr, epoch_time)
            
            is_best = False
            if valid_loss is not None:
                is_best = self.metrics.update_valid(valid_loss, epoch)
            
            # 记录到tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            if valid_loss is not None:
                self.writer.add_scalar('Loss/Valid', valid_loss, epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            self.writer.add_scalar('EpochTime', epoch_time, epoch)
            
            # 计算预计时间
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # 计算平均epoch时间（使用最近的epoch时间来提高预测准确性）
            if len(self.metrics.epoch_times) > 0:
                # 使用最近5个epoch的平均时间，如果不足5个则使用所有的
                recent_times = self.metrics.epoch_times[-5:]
                avg_epoch_time = sum(recent_times) / len(recent_times)
            else:
                avg_epoch_time = epoch_time
            
            # 计算剩余时间
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_remaining_time = remaining_epochs * avg_epoch_time
            estimated_completion_time = current_time + estimated_remaining_time
            
            # 格式化时间显示
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}秒"
                elif seconds < 3600:
                    minutes = seconds / 60
                    return f"{minutes:.1f}分钟"
                else:
                    hours = seconds / 3600
                    return f"{hours:.1f}小时"
            
            def format_datetime(timestamp):
                return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            # 打印进度
            log_msg = f"Epoch {epoch+1}/{num_epochs} - "
            log_msg += f"训练损失: {train_loss:.6f}, "
            if valid_loss is not None:
                log_msg += f"验证损失: {valid_loss:.6f}, "
            log_msg += f"学习率: {current_lr:.2e}, "
            log_msg += f"时间: {epoch_time:.1f}s"
            
            if is_best:
                log_msg += " ⭐ 最佳模型!"
            
            logger.info(log_msg)
            
            # 显示时间统计信息
            time_msg = f"⏱️  已用时间: {format_time(elapsed_time)}, "
            time_msg += f"预计剩余: {format_time(estimated_remaining_time)}, "
            time_msg += f"预计完成: {format_datetime(estimated_completion_time)}"
            logger.info(time_msg)
            
            # 保存检查点
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch, is_best)
            
            # 绘制训练曲线
            if (epoch + 1) % (self.config.save_every * 2) == 0:
                self.plot_training_curves()
            
            # 早停检查
            if (self.metrics.early_stopping_counter >= self.config.early_stopping_patience 
                and not self.config.test_mode):
                logger.info(f"早停触发 (patience={self.config.early_stopping_patience})")
                break
        
        # 训练结束
        total_time = time.time() - self.start_time
        logger.info("="*60)
        logger.info("训练完成!")
        logger.info(f"总训练时间: {total_time/3600:.2f} 小时")
        logger.info(f"最佳验证损失: {self.metrics.best_valid_loss:.6f} (Epoch {self.metrics.best_epoch+1})")
        logger.info("="*60)
        
        # 最终保存
        self.save_checkpoint(epoch, False)
        self.plot_training_curves()
        
        # 保存训练指标
        metrics_path = self.config.output_dir / "training_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics.to_dict(), f, indent=2, ensure_ascii=False)
        
        self.writer.close()


def run_small_scale_test():
    """运行小规模测试"""
    print("🧪 运行小规模测试...")
    
    # 测试配置
    config = TrainingConfig()
    config.test_mode = True
    config.test_epochs = 1
    config.test_samples = 6  # 改回到6个样本
    config.batch_size = 2    # 使用batch_size=2，避免单样本问题
    config.max_sequence_length = 128  # 使用更合理的序列长度
    config.device = "cpu"
    config.num_workers = 0  # 避免多进程问题
    config.output_dir = "./test_output"
    config.checkpoint_dir = "./test_checkpoints"
    config.mixed_precision = False  # 测试时禁用混合精度
    config.use_data_parallel = False  # 测试时禁用多GPU
    
    try:
        # 创建训练器
        trainer = DiffoldTrainer(config)
        
        # 运行训练
        trainer.train()
        
        print("✅ 小规模测试完成!")
        print(f"📁 输出目录: {config.output_dir}")
        print(f"📁 检查点目录: {config.checkpoint_dir}")
        
    except Exception as e:
        print(f"❌ 小规模测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Diffold模型训练")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./processed_data", help="数据目录")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载进程数")
    parser.add_argument("--fold", type=int, default=0, help="交叉验证折数 (0-9)")
    parser.add_argument("--use_all_folds", action="store_true", help="使用所有折数的数据进行训练", default=False)
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    
    # 模型参数
    parser.add_argument("--rhofold_checkpoint", type=str, 
                       default="./pretrained/model_20221010_params.pt", 
                       help="RhoFold预训练权重路径")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点目录")
    parser.add_argument("--save_every", type=int, default=5, help="每N轮保存一次")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda", help="设备 (auto/cpu/cuda)")
    parser.add_argument("--no_mixed_precision", action="store_true", help="禁用混合精度训练")
    parser.add_argument("--no_data_parallel", action="store_true", help="禁用DataParallel多GPU训练")
    parser.add_argument("--gpu_ids", type=int, nargs='+', help="指定使用的GPU ID (例如: --gpu_ids 0 1 2)")
    
    # 其他参数
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--test", action="store_true", help="运行小规模测试", default=True)
    
    args = parser.parse_args()
    
    # 如果是测试模式
    if args.test:
        run_small_scale_test()
        return
    
    # 创建配置
    config = TrainingConfig()
    
    # 更新配置
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.max_sequence_length = args.max_length
    config.num_workers = args.num_workers
    config.fold = args.fold
    config.use_all_folds = args.use_all_folds
    
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.grad_clip_norm = args.grad_clip
    
    config.rhofold_checkpoint = args.rhofold_checkpoint
    config.output_dir = args.output_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.save_every = args.save_every
    
    if args.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        config.device = args.device
    
    config.mixed_precision = not args.no_mixed_precision
    config.use_data_parallel = not args.no_data_parallel
    config.gpu_ids = args.gpu_ids
    
    # 创建训练器
    trainer = DiffoldTrainer(config)
    
    # 开始训练
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
