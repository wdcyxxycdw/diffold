#!/usr/bin/env python3
"""
Diffold模型训练脚本 - 增强版
整合了所有训练保障措施和优化功能
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, cast
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# 导入模型和数据处理
from diffold.diffold import Diffold
from diffold.dataloader import create_data_loaders

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔥 导入增强功能模块
try:
    from diffold.training_monitor import TrainingMonitor
    from diffold.advanced_optimizers import AdaptiveOptimizer, DataLoaderOptimizer, EvaluationMetrics
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ 增强功能不可用: {e}")
    logger.info("💡 安装依赖: pip install psutil matplotlib")
    ENHANCED_FEATURES_AVAILABLE = False

class TrainingConfig:
    """训练配置类 - 兼容原版和增强版"""
    
    def __init__(self):
        # 基础数据配置
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
        self.warmup_epochs = 3  # 修复: 减少预热轮数，针对100轮优化
        
        # 调度器配置
        self.scheduler_type = "cosine"  # "cosine", "plateau", "warmup_cosine"
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
        # torch.compile 配置
        self.use_torch_compile = False
        self.torch_compile_mode = 'default'  # 可选: 'default', 'reduce-overhead', 'max-autotune'
        
        # 多GPU配置
        self.use_data_parallel = True
        self.gpu_ids = None
        
        # 小规模测试配置
        self.test_mode = False
        self.test_samples = 10
        self.test_epochs = 5
        
        # 交叉验证配置
        self.fold = 0
        self.num_folds = 10
        self.use_all_folds = False
        
        # 🔥 增强功能配置
        self.enhanced_features = {
            'enable_enhanced_training': ENHANCED_FEATURES_AVAILABLE,  # 总开关
            'monitoring': {
                'enable_performance_monitoring': True,
                'enable_memory_monitoring': True,
                'enable_health_checking': True,
                'monitoring_interval': 10,
                'save_monitoring_plots': True,
                'memory_cleanup_threshold': 0.85
            },
            'optimizer': {
                'use_advanced_optimizer': True,
                'optimizer_name': 'adamw',  # 'adamw', 'adam', 'sgd', 'lion'
                'gradient_accumulation_steps': 1,
                'scheduler_type': 'warmup_cosine'  # 'warmup_cosine', 'warmup_cosine_restarts', 'plateau'
            },
            'dataloader': {
                'enable_prefetch': True,
                'prefetch_factor': 2,
                'cache_size': 100,
                'pin_memory': True,
                'persistent_workers': True
            },
            'evaluation': {
                'compute_structure_metrics': True,
                'compute_confidence_metrics': True,
                'save_predictions': False
            },
            'error_recovery': {
                'auto_retry_on_oom': True,
                'max_retry_attempts': 3,
                'reduce_batch_size_on_oom': True
            }
        }
    
    def apply_enhanced_preset(self, preset_name: str):
        """应用增强功能预设"""
        if not ENHANCED_FEATURES_AVAILABLE:
            logger.warning("增强功能不可用，跳过预设应用")
            return
        
        presets = {
            'performance': {
                'dataloader': {'enable_prefetch': True, 'prefetch_factor': 3},
                'optimizer': {'gradient_accumulation_steps': 2 if self.batch_size < 8 else 1},
                'monitoring': {'enable_performance_monitoring': True}
            },
            'safety': {
                'monitoring': {'enable_health_checking': True},
                'error_recovery': {'auto_retry_on_oom': True},
                'batch_size': min(self.batch_size, 4)  # 更保守的batch size
            },
            'memory': {
                'batch_size': max(1, self.batch_size // 2),
                'optimizer': {'gradient_accumulation_steps': self.enhanced_features['optimizer']['gradient_accumulation_steps'] * 2},
                'dataloader': {'prefetch_factor': 1},
                'monitoring': {'memory_cleanup_threshold': 0.75}
            },
            'debug': {
                'monitoring': {'monitoring_interval': 1, 'save_monitoring_plots': True},
                'evaluation': {'compute_structure_metrics': True, 'save_predictions': True},
                'test_mode': True
            }
        }
        
        if preset_name in presets:
            preset = presets[preset_name]
            for category, settings in preset.items():
                if category in self.enhanced_features:
                    self.enhanced_features[category].update(settings)
                elif hasattr(self, category):
                    setattr(self, category, settings)
            logger.info(f"✅ 应用预设: {preset_name}")
        else:
            logger.warning(f"未知预设: {preset_name}")


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
    """Diffold模型训练器 - 增强版"""
    
    def __init__(self, config: TrainingConfig, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.metrics = TrainingMetrics()
        
        # 创建输出目录
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main_process = (self.local_rank == 0)
        if self.is_main_process:
            self.setup_directories()
        
        # 设置日志
        if self.is_main_process:
            self.setup_logging()
        
        # 初始化设备
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(config.device)
        if self.is_main_process:
            logger.info(f"使用设备: {self.device}")
        
        # 🔥 初始化增强功能
        self.enhanced_enabled = config.enhanced_features.get('enable_enhanced_training', False)
        self.training_monitor = None
        self.enhanced_optimizer = None
        self.enhanced_metrics = None
        
        if self.enhanced_enabled:
            self._setup_enhanced_features()
        
        # 初始化模型
        self.model = self.setup_model()
        # 移动到设备
        self.model = self.model.to(self.device)
        # ⚡ 可选 torch.compile
        if self.config.use_torch_compile:
            try:
                self.model = torch.compile(self.model, mode=self.config.torch_compile_mode)
                if self.is_main_process:
                    logger.info(f"⚡ 已启用 torch.compile (mode={self.config.torch_compile_mode})")
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"torch.compile 启用失败: {e}")
        # 统计并打印可训练参数总数
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.is_main_process:
            logger.info(f"可训练参数总数: {total_params}")
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            self.using_ddp = True
            self.num_gpus = world_size
            if self.is_main_process:
                logger.info(f"使用DDP，GPU数量: {self.num_gpus}")
        else:
            # 单卡已在上面 to(device)
            self.using_ddp = False
            self.num_gpus = 1
        
        # 初始化数据加载器
        self.setup_data_loaders()
        
        # 初始化优化器和调度器
        self.setup_optimizer_and_scheduler()
        
        # 初始化tensorboard
        if self.is_main_process:
            self.writer = SummaryWriter(log_dir=str(self.config.output_dir) + "/tensorboard")
        
        # 混合精度训练
        if self.config.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            if self.is_main_process:
                logger.info("启用混合精度训练")
        else:
            self.scaler = None
        
        # 记录开始时间
        self.start_time = time.time()
    
    def _setup_enhanced_features(self):
        """设置增强功能"""
        logger.info("🔥 启用增强训练功能")
        
        # 训练监控
        if self.config.enhanced_features['monitoring']['enable_performance_monitoring']:
            self.training_monitor = TrainingMonitor(self.config.output_dir)
            logger.info("✅ 训练监控已启用")
        
        # 评估指标
        if self.config.enhanced_features['evaluation']['compute_structure_metrics']:
            self.enhanced_metrics = {
                'train': EvaluationMetrics(),
                'val': EvaluationMetrics()
            }
            logger.info("✅ 增强评估指标已启用")
    
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
        """设置日志"""
        log_file = self.config.output_dir / "training.log"
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加到logger
        logger.addHandler(file_handler)
        logger.info("日志系统已初始化")
    
    def setup_model(self):
        """设置模型"""
        logger.info("初始化Diffold模型...")
        model = Diffold(self.config, rhofold_checkpoint_path=self.config.rhofold_checkpoint)
        logger.info("模型初始化完成")
        return model
    
    def setup_data_loaders(self):
        """设置数据加载器"""
        if self.is_main_process:
            logger.info("设置数据加载器...")
        
        # 创建基础数据加载器，传递分布式信息
        train_loader, valid_loader = create_data_loaders(
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            max_length=self.config.max_sequence_length,
            num_workers=self.config.num_workers,
            fold=self.config.fold,
            use_msa=self.config.use_msa,
            use_all_folds=self.config.use_all_folds,
            world_size=self.world_size,
            local_rank=self.local_rank
        )
        # 🔥 应用数据加载优化
        if (self.enhanced_enabled and 
            self.config.enhanced_features['dataloader']['enable_prefetch']):
            if self.is_main_process:
                logger.info("🚀 启用数据预取优化")
            
            self.train_loader = DataLoaderOptimizer(
                train_loader,
                prefetch_factor=self.config.enhanced_features['dataloader']['prefetch_factor'],
                cache_size=self.config.enhanced_features['dataloader']['cache_size'],
                enable_prefetch=True
            )
            self.valid_loader = DataLoaderOptimizer(
                valid_loader,
                prefetch_factor=1,  # 验证时使用较小的预取
                enable_prefetch=True
            )
        else:
            self.train_loader = train_loader
            self.valid_loader = valid_loader
        
        if self.is_main_process:
            logger.info(f"训练集大小: {len(train_loader)}")
            logger.info(f"验证集大小: {len(valid_loader)}")
    
    def setup_optimizer_and_scheduler(self):
        """设置优化器和调度器"""
        logger.info("设置优化器和调度器...")
        
        # 🔥 使用增强优化器
        if (self.enhanced_enabled and 
            self.config.enhanced_features['optimizer']['use_advanced_optimizer']):
            logger.info("🎯 使用高级优化器")
            
            self.enhanced_optimizer = AdaptiveOptimizer(  # type: ignore[arg-type]
                model=cast(nn.Module, self.model),
                optimizer_name=self.config.enhanced_features['optimizer']['optimizer_name'],
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                scheduler_config={
                    'type': self.config.enhanced_features['optimizer']['scheduler_type'],
                    'warmup_epochs': self.config.warmup_epochs,
                    'T_max': self.config.num_epochs,
                    'eta_min': 1e-6
                },
                gradient_accumulation_steps=self.config.enhanced_features['optimizer']['gradient_accumulation_steps'],
                max_grad_norm=self.config.grad_clip_norm
            )
            
            # 包装原接口
            self.optimizer = self.enhanced_optimizer.optimizer
            self.scheduler = self.enhanced_optimizer.scheduler
            
            # 检查是否需要使用plateau调度器（来自学习率修复）
            if hasattr(self, 'checkpoint_data') and self.checkpoint_data.get('use_plateau_scheduler', False):
                logger.info("🔄 检测到学习率修复标记，切换到plateau调度器")
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.8, patience=5, verbose=True
                )
            
        else:
            # 使用原版优化器
            if hasattr(self.model, 'get_trainable_parameters'):
                trainable_params = self.model.get_trainable_parameters()
            else:
                trainable_params = self.model.parameters()
            
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # 创建调度器
            if self.config.scheduler_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.config.num_epochs, eta_min=1e-6
                )
            elif self.config.scheduler_type == "plateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=self.config.patience, verbose=True
                )
            else:
                self.scheduler = None
        
        logger.info("优化器和调度器设置完成")
    
    def train_one_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        # 兼容DataParallel
        if self.using_ddp:
            self.model.module.set_train_mode()
        else:
            self.model.set_train_mode()
        
        total_loss = 0.0
        num_batches = 0
        
        # 🔥 重置增强指标
        if self.enhanced_metrics:
            self.enhanced_metrics['train'].reset()
        
        # 测试模式下限制batch数量
        if self.config.test_mode:
            max_batches = min(self.config.test_samples // self.config.batch_size + 1, 5)
        else:
            max_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=min(max_batches, len(self.train_loader)),
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            leave=False,
            disable=not self.is_main_process
        )
        
        for batch_idx, batch in progress_bar:
            if self.config.test_mode and batch_idx >= max_batches:
                break
            
            batch_start_time = time.time()
            
            try:
                loss = self.train_step(batch, batch_idx, epoch)
                
                if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    num_batches += 1
                    
                    batch_time = time.time() - batch_start_time
                    
                    # 🔥 记录监控数据
                    if (self.training_monitor and 
                        batch_idx % self.config.enhanced_features['monitoring']['monitoring_interval'] == 0):
                        self.training_monitor.log_training_step(
                            step=batch_idx + epoch * len(self.train_loader),
                            epoch=epoch,
                            loss_value=loss.item(),
                            learning_rate=self.optimizer.param_groups[0]['lr'],
                            batch_time=batch_time,
                            model=self.model
                        )
                    
                    # 更新进度条
                    postfix_dict = {
                        'loss': loss.item(),
                        'avg_loss': total_loss / (batch_idx + 1),
                        'lr': self.optimizer.param_groups[0]['lr']
                    }
                    if self.device.type == 'cuda':
                        memory_reserved_gb = torch.cuda.memory_reserved(self.device) / 1024**3
                        postfix_dict['mem_gb'] = f"{memory_reserved_gb:.2f}"
                    
                    progress_bar.set_postfix(**postfix_dict)
                else:
                    logger.warning(f"Batch {batch_idx}: 无效损失，跳过")
                    
            except RuntimeError as e:
                # 🔥 OOM错误处理
                if ('out of memory' in str(e) and 
                    self.config.enhanced_features['error_recovery']['auto_retry_on_oom']):
                    logger.warning(f"检测到OOM错误，尝试恢复: {e}")
                    
                    # 清理内存
                    if self.training_monitor:
                        self.training_monitor.memory_manager.cleanup_memory(aggressive=True)
                    else:
                        torch.cuda.empty_cache()
                    
                    # 减少batch size（如果启用）
                    if self.config.enhanced_features['error_recovery']['reduce_batch_size_on_oom']:
                        logger.warning("考虑减少batch_size以避免OOM")
                    
                    continue
                else:
                    logger.error(f"Batch {batch_idx} 训练失败: {e}")
                    raise
            
            except Exception as e:
                logger.warning(f"Batch {batch_idx} 训练失败: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def train_step(self, batch: Dict, batch_idx: int, epoch: int) -> Optional[torch.Tensor]:
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
        
        # 🔥 使用增强优化器或原版优化器
        if self.enhanced_optimizer:
            # 增强优化器自动处理梯度累积
            self.enhanced_optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        
        # 前向传播
        if self.scaler is not None:
            with torch.autocast('cuda', dtype=torch.bfloat16):
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
        
        # 提取损失
        if isinstance(result, dict):
            loss = result.get('loss', None)
        elif isinstance(result, tuple):
            loss = result[0]
        else:
            loss = result
        
        if loss is None:
            return None
        
        # 🔥 更新增强评估指标
        if self.enhanced_metrics and isinstance(result, dict):
            loss_breakdown = {}
            if 'loss_breakdown' in result:
                breakdown = result['loss_breakdown']
                if hasattr(breakdown, 'total_diffusion'):
                    loss_breakdown['total_diffusion'] = breakdown.total_diffusion.item()
                if hasattr(breakdown, 'confidence'):
                    loss_breakdown['confidence'] = breakdown.confidence.item()
            
            self.enhanced_metrics['train'].update(
                loss=loss.item(),
                batch_size=batch['tokens'].size(0),
                loss_breakdown=loss_breakdown,
                predicted_coords=result.get('predicted_coords'),
                target_coords=coordinates
            )
        
        # 反向传播
        if self.enhanced_optimizer:
            # 使用增强优化器（自动处理梯度累积）
            self.enhanced_optimizer.backward(loss)
        else:
            # 使用原版优化器
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.using_ddp:
                    torch.nn.utils.clip_grad_norm_(self.model.module.get_trainable_parameters(), self.config.grad_clip_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), self.config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.using_ddp:
                    torch.nn.utils.clip_grad_norm_(self.model.module.get_trainable_parameters(), self.config.grad_clip_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), self.config.grad_clip_norm)
                self.optimizer.step()
        
        return loss
    
    def validate(self, epoch: int) -> float:
        """验证模型"""
        # 兼容DataParallel
        if self.using_ddp:
            self.model.module.set_eval_mode()
        else:
            self.model.set_eval_mode()
        
        total_loss = 0.0
        num_batches = 0
        
        # 🔥 重置增强指标
        if self.enhanced_metrics:
            self.enhanced_metrics['val'].reset()
        
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
                leave=False,
                disable=not self.is_main_process
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
                        # 提取损失
                        if isinstance(result, dict):
                            loss = result.get('loss', None)
                        elif isinstance(result, tuple):
                            loss = result[0]
                        else:
                            loss = result
                        
                        if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                            total_loss += loss.item()
                            num_batches += 1
                            
                            # 🔥 更新增强评估指标
                            if self.enhanced_metrics and isinstance(result, dict):
                                self.enhanced_metrics['val'].update(
                                    loss=loss.item(),
                                    batch_size=batch['tokens'].size(0),
                                    predicted_coords=result.get('predicted_coords'),
                                    target_coords=coordinates,
                                    confidence_scores=result.get('confidence_logits')
                                )
                            
                            postfix_dict = {
                                'val_loss': loss.item(),
                                'avg_val_loss': total_loss / (batch_idx + 1)
                            }
                            if self.device.type == 'cuda':
                                memory_reserved_gb = torch.cuda.memory_reserved(self.device) / 1024**3
                                postfix_dict['mem_gb'] = f"{memory_reserved_gb:.2f}"
                            progress_bar.set_postfix(**postfix_dict)
                
                except Exception as e:
                    logger.warning(f"验证 Batch {batch_idx} 失败: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        # 处理DataParallel的state_dict
        if self.using_ddp:
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
            'using_data_parallel': self.using_ddp,
            'num_gpus': self.num_gpus,
            'enhanced_enabled': self.enhanced_enabled
        }
        
        # 🔥 保存增强功能状态
        if self.enhanced_optimizer:
            checkpoint['enhanced_optimizer_stats'] = self.enhanced_optimizer.get_stats()
        
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
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.checkpoint_data = checkpoint  # 保存检查点数据供调度器使用
        
        # 加载模型状态（处理DataParallel）
        model_state_dict = checkpoint['model_state_dict']
        if self.using_ddp:
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
        if self.is_main_process:
            logger.info("开始训练...")
        
        # 🔥 打印增强功能状态
        if self.enhanced_enabled:
            if self.is_main_process:
                logger.info("🔥 增强功能已启用:")
                for category, features in self.config.enhanced_features.items():
                    if isinstance(features, dict):
                        enabled_features = [k for k, v in features.items() if v]
                        if enabled_features:
                            logger.info(f"  {category}: {', '.join(enabled_features)}")
        
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
            if self.enhanced_optimizer:
                # 使用增强优化器
                self.enhanced_optimizer.scheduler_step(valid_loss)
            else:
                # 使用原版调度器
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
            
            # 🔥 记录增强评估指标
            if self.enhanced_metrics:
                train_metrics = self.enhanced_metrics['train'].compute_metrics()
                val_metrics = self.enhanced_metrics['val'].compute_metrics() if valid_loss is not None else {}
                
                if 'avg_rmsd' in val_metrics and self.is_main_process:
                    logger.info(f"验证RMSD: {val_metrics['avg_rmsd']:.4f}")
            
            # 只在主进程写tensorboard
            if self.is_main_process:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                if valid_loss is not None:
                    self.writer.add_scalar('Loss/Valid', valid_loss, epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
                self.writer.add_scalar('EpochTime', epoch_time, epoch)
            
            # 计算预计时间
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # 计算平均epoch时间
            if len(self.metrics.epoch_times) > 0:
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
            if self.is_main_process:
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
            if self.is_main_process:
                if (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(epoch, is_best)
                # 绘制训练曲线
                if (epoch + 1) % (self.config.save_every * 2) == 0:
                    self.plot_training_curves()
                # 🔥 保存监控报告
                if (self.training_monitor and 
                    self.config.enhanced_features['monitoring']['save_monitoring_plots'] and
                    (epoch + 1) % 5 == 0):
                    self.training_monitor.save_monitoring_report()
                    self.training_monitor.generate_performance_plots()
            
            # 早停检查
            if (self.metrics.early_stopping_counter >= self.config.early_stopping_patience 
                and not self.config.test_mode):
                if self.is_main_process:
                    logger.info(f"早停触发 (patience={self.config.early_stopping_patience})")
                break
        
        # 训练结束
        total_time = time.time() - self.start_time
        if self.is_main_process:
            logger.info("="*60)
            logger.info("训练完成!")
            logger.info(f"总训练时间: {total_time/3600:.2f} 小时")
            logger.info(f"最佳验证损失: {self.metrics.best_valid_loss:.6f} (Epoch {self.metrics.best_epoch+1})")
            # 🔥 显示增强功能统计
            if self.enhanced_optimizer:
                stats = self.enhanced_optimizer.get_stats()
                logger.info(f"优化器统计: 更新次数={stats.get('update_count', 0)}, "
                           f"平均梯度范数={stats.get('avg_grad_norm', 0):.4f}")
            if self.training_monitor:
                logger.info("性能监控报告:")
                summary = self.training_monitor.performance_monitor.get_performance_summary()
                logger.info(f"  总batch数: {summary.get('total_batches', 0)}")
                logger.info(f"  OOM次数: {summary.get('oom_count', 0)}")
                logger.info(f"  NaN损失次数: {summary.get('nan_loss_count', 0)}")
                if 'avg_batch_time' in summary:
                    logger.info(f"  平均batch时间: {summary['avg_batch_time']:.2f}s")
            logger.info("="*60)
            # 最终保存
            self.save_checkpoint(epoch, False)
            self.plot_training_curves()
            # 🔥 最终报告
            if self.training_monitor:
                self.training_monitor.save_monitoring_report()
                self.training_monitor.generate_performance_plots()
            # 保存训练指标
            metrics_path = str(self.config.output_dir) + "/training_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics.to_dict(), f, indent=2, ensure_ascii=False)
            self.writer.close()
        # DDP结束
        if self.world_size > 1:
            dist.destroy_process_group()


def run_small_scale_test():
    """运行小规模测试"""
    logger.info("🧪 运行小规模测试...")
    
    # 测试配置
    config = TrainingConfig()
    config.test_mode = True
    config.test_epochs = 1
    config.test_samples = 6
    config.batch_size = 2
    config.max_sequence_length = 128
    config.device = "cuda"
    config.num_workers = 0
    config.output_dir = "./test_output"
    config.checkpoint_dir = "./test_checkpoints"
    config.mixed_precision = False
    config.use_data_parallel = False
    
    # 🔥 启用增强功能进行测试
    if ENHANCED_FEATURES_AVAILABLE:
        config.apply_enhanced_preset('debug')
    
    try:
        # 创建训练器
        trainer = DiffoldTrainer(config)
        
        # 运行训练
        trainer.train()
        
        logger.info("✅ 小规模测试完成!")
        logger.info(f"📁 输出目录: {config.output_dir}")
        logger.info(f"📁 检查点目录: {config.checkpoint_dir}")
        
        exit()
    except Exception as e:
        logger.error(f"❌ 小规模测试失败: {e}")
        import traceback
        traceback.print_exc()


def fix_checkpoint_learning_rate(checkpoint_path, new_lr):
    """修复检查点中的学习率 - 针对预热bug的快速修复"""
    import shutil
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"检查点不存在: {checkpoint_path}")
        return False
    
    # 备份
    backup_path = f"{checkpoint_path}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(checkpoint_path, backup_path)
        logger.info(f"已备份检查点至: {backup_path}")
    
    # 修复学习率 - 兼容PyTorch 2.6的安全机制
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer_state = checkpoint['optimizer_state_dict']
        if 'param_groups' in optimizer_state:
            for group in optimizer_state['param_groups']:
                old_lr = group['lr']
                group['lr'] = new_lr
                logger.info(f"🔧 修复学习率: {old_lr:.2e} → {new_lr:.2e}")
    
    # 重置调度器状态，使用plateau调度器避免继续下降
    if 'scheduler_state_dict' in checkpoint:
        logger.info("🔄 重置调度器状态，改用plateau调度器")
        del checkpoint['scheduler_state_dict']
        # 添加配置信息，让恢复时使用plateau调度器
        checkpoint['use_plateau_scheduler'] = True
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"✅ 学习率修复完成: {checkpoint_path}")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Diffold模型训练 - 增强版")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./processed_data", help="数据目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--max_length", type=int, default=256, help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载进程数")
    parser.add_argument("--fold", type=int, default=0, help="交叉验证折数 (0-9)")
    parser.add_argument("--use_all_folds", action="store_true", help="使用所有折数的数据进行训练")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
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
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cpu/cuda)")
    parser.add_argument("--no_mixed_precision", action="store_true", help="禁用混合精度训练")
    parser.add_argument("--no_data_parallel", action="store_true", help="禁用DataParallel多GPU训练")
    parser.add_argument("--gpu_ids", type=int, nargs='+', help="指定使用的GPU ID")
    parser.add_argument('--local_rank', type=int, default=0, help='DDP local rank')
    
    # 🔥 增强功能参数
    parser.add_argument("--enhanced_preset", type=str, default=None,
                       choices=['performance', 'safety', 'memory', 'debug'],
                       help="增强功能预设")
    parser.add_argument("--disable_enhanced", action="store_true", 
                       help="禁用所有增强功能")
    parser.add_argument("--disable_monitoring", action="store_true",
                       help="禁用性能监控")
    parser.add_argument("--disable_prefetch", action="store_true",
                       help="禁用数据预取")
    parser.add_argument("--disable_advanced_optimizer", action="store_true",
                       help="禁用高级优化器")
    # torch.compile 相关参数
    parser.add_argument("--torch_compile", action="store_true", help="启用 torch.compile (PyTorch>=2.0)")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile 优化模式")
    parser.add_argument("--grad_accum", type=int, default=None,
                       help="梯度累积步数 (gradient_accumulation_steps)，默认根据预设或1")
    
    # 其他参数
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--test", action="store_true", help="运行小规模测试")
    
    # 🔧 学习率修复参数
    parser.add_argument("--fix_lr", type=float, default=None, 
                       help="修复检查点中的学习率 (配合--resume使用)")
    parser.add_argument("--fix_lr_only", action="store_true",
                       help="仅修复学习率，不开始训练")
    
    args = parser.parse_args()
    
    # 如果是测试模式
    if args.test:
        run_small_scale_test()
        return
    
    # 🔧 处理学习率修复
    if args.fix_lr is not None:
        if args.resume is None:
            logger.error("❌ 使用 --fix_lr 需要指定 --resume 检查点路径")
            return
        
        logger.info(f"🔧 开始修复学习率: {args.fix_lr}")
        success = fix_checkpoint_learning_rate(args.resume, args.fix_lr)
        
        if success:
            logger.info("✅ 学习率修复完成!")
            if args.fix_lr_only:
                logger.info("💡 使用以下命令恢复训练:")
                logger.info(f"python train.py --resume {args.resume}")
                return
            else:
                logger.info("🚀 继续开始训练...")
        else:
            logger.error("❌ 学习率修复失败")
            return
    
    # 创建配置
    config = TrainingConfig()
    
    # 🔥 应用增强功能设置
    if args.disable_enhanced or not ENHANCED_FEATURES_AVAILABLE:
        config.enhanced_features['enable_enhanced_training'] = False
        if args.disable_enhanced:
            logger.warning("⚠️ 增强功能已手动禁用")
    else:
        # 应用预设
        if args.enhanced_preset:
            config.apply_enhanced_preset(args.enhanced_preset)
            logger.info(f"🔥 应用增强预设: {args.enhanced_preset}")
        
        # 应用具体禁用选项
        if args.disable_monitoring:
            config.enhanced_features['monitoring']['enable_performance_monitoring'] = False
        if args.disable_prefetch:
            config.enhanced_features['dataloader']['enable_prefetch'] = False
        if args.disable_advanced_optimizer:
            config.enhanced_features['optimizer']['use_advanced_optimizer'] = False
    
    # 如果指定了梯度累积步数
    if args.grad_accum is not None:
        config.enhanced_features['optimizer']['gradient_accumulation_steps'] = max(1, args.grad_accum)
        logger.info(f"✅ 设置梯度累积步数为: {config.enhanced_features['optimizer']['gradient_accumulation_steps']}")

    # torch.compile 设置
    if args.torch_compile:
        config.use_torch_compile = True
        config.torch_compile_mode = args.compile_mode
        logger.info(f"⚡ 计划启用 torch.compile (mode={config.torch_compile_mode})")
    
    # 更新基础配置
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
    
    # DDP初始化
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(config.device)
    
    # 打印配置信息
    logger.info("🎯 Diffold训练 - 增强版")
    logger.info("="*50)
    logger.info(f"📁 数据目录: {config.data_dir}")
    logger.info(f"📦 批次大小: {config.batch_size}")
    logger.info(f"📏 最大序列长度: {config.max_sequence_length}")
    logger.info(f"🖥️  设备: {config.device}")
    logger.info(f"⏱️  训练轮数: {config.num_epochs}")
    logger.info(f"📊 学习率: {config.learning_rate}")
    
    # 🔥 显示增强功能状态
    if config.enhanced_features.get('enable_enhanced_training', False):
        logger.info("🔥 增强功能: 已启用")
        enabled_features = []
        if config.enhanced_features['monitoring']['enable_performance_monitoring']:
            enabled_features.append("性能监控")
        if config.enhanced_features['dataloader']['enable_prefetch']:
            enabled_features.append("数据预取")
        if config.enhanced_features['optimizer']['use_advanced_optimizer']:
            enabled_features.append("高级优化器")
        if config.enhanced_features['evaluation']['compute_structure_metrics']:
            enabled_features.append("结构评估")
        if enabled_features:
            logger.info(f"   • {', '.join(enabled_features)}")
    else:
        logger.info("⚪ 增强功能: 已禁用（使用原版功能）")
    
    logger.info("="*50)
    
    # 创建训练器
    trainer = DiffoldTrainer(config, local_rank=local_rank, world_size=world_size)
    
    # 开始训练
    trainer.train(resume_from=args.resume)
    # DDP结束
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
