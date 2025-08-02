#!/usr/bin/env python3
"""
Diffold模型训练脚本 - 增强版
整合了所有训练保障措施和优化功能
"""

import argparse
import json
import logging
import time
import yaml
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
def setup_logging(log_level: str = "INFO"):
    """设置全局日志配置"""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    log_level = log_level.upper()
    if log_level not in level_map:
        log_level = "INFO"
    
    logging.basicConfig(
        level=level_map[log_level],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 控制台输出
        ]
    )
    
    # 设置所有模块的日志级别
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(level_map[log_level])
    
    return logging.getLogger(__name__)

# 默认设置
logger = setup_logging()

# 🔥 导入增强功能模块
try:
    from diffold.training_monitor import TrainingMonitor
    from diffold.advanced_optimizers import AdaptiveOptimizer, DataLoaderOptimizer
    from diffold.rna_metrics import RNAEvaluationMetrics
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ 增强功能不可用: {e}")
    logger.info("💡 安装依赖: pip install psutil matplotlib")
    ENHANCED_FEATURES_AVAILABLE = False

class TrainingConfig:
    """训练配置类 - 兼容原版和增强版"""
    
    def __init__(self, config_file: Optional[str] = None):
        # 基础数据配置
        self.data_dir = "./processed_data"
        self.batch_size = 8
        self.max_sequence_length = 256
        self.num_workers = 4
        self.use_msa = True
        
        # 模型配置
        self.rhofold_checkpoint = "./pretrained/model_20221010_params.pt"
        
        # 训练配置
        self.num_epochs = 100
        self.learning_rate = 1.2e-4
        self.weight_decay = 1e-5
        self.grad_clip_norm = 1.0
        self.warmup_steps = 1000  # 预热步数（基于step而不是epoch）
        
        # 调度器配置
        self.scheduler_type = "warmup_cosine"  # "cosine", "plateau", "warmup_cosine", "warmup_cosine_restarts"
        self.patience = 10
        
        # 保存配置
        self.output_dir = "./output"
        self.checkpoint_dir = "./checkpoints"
        self.save_every = 1  # 检查点保存频率
        self.plot_every = 1  # 训练曲线保存频率
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
                'monitoring_interval': 1,
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
        
        # 日志配置
        self.log_level = "INFO"
        
        # 学习率修改配置
        self.learning_rate_modification = {
            'enable_runtime_modification': True,
            'save_modification_history': True,
            'log_lr_changes': True,
            'validation_checks': {
                'min_lr': 1e-7,
                'max_lr': 1e-1,
                'warn_on_large_changes': True,
                'large_change_threshold': 10.0
            }
        }
        
        # 如果提供了配置文件，则加载配置
        if config_file:
            self.load_from_yaml(config_file)
    
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
    
    def load_from_yaml(self, config_file: str):
        """从YAML文件加载配置"""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        logger.info(f"📄 加载配置文件: {config_file}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 加载数据配置
        if 'data' in config_data:
            data_config = config_data['data']
            self.data_dir = data_config.get('data_dir', self.data_dir)
            self.batch_size = data_config.get('batch_size', self.batch_size)
            self.max_sequence_length = data_config.get('max_sequence_length', self.max_sequence_length)
            self.num_workers = data_config.get('num_workers', self.num_workers)
            self.use_msa = data_config.get('use_msa', self.use_msa)
            self.fold = data_config.get('fold', self.fold)
        
        # 加载模型配置
        if 'model' in config_data:
            model_config = config_data['model']
            self.rhofold_checkpoint = model_config.get('rhofold_checkpoint', self.rhofold_checkpoint)
        
        # 加载训练配置
        if 'training' in config_data:
            training_config = config_data['training']
            self.num_epochs = training_config.get('num_epochs', self.num_epochs)
            self.learning_rate = training_config.get('learning_rate', self.learning_rate)
            self.weight_decay = training_config.get('weight_decay', self.weight_decay)
            self.grad_clip_norm = training_config.get('grad_clip_norm', self.grad_clip_norm)
            self.warmup_steps = training_config.get('warmup_steps', self.warmup_steps)
            self.scheduler_type = training_config.get('scheduler_type', self.scheduler_type)
            self.patience = training_config.get('patience', self.patience)
            self.validate_every = training_config.get('validate_every', self.validate_every)
            self.early_stopping_patience = training_config.get('early_stopping_patience', self.early_stopping_patience)
            
            # 加载学习率修改配置
            if 'learning_rate_modification' in training_config:
                lr_mod_config = training_config['learning_rate_modification']
                self.learning_rate_modification.update(lr_mod_config)
                if 'validation_checks' in lr_mod_config:
                    self.learning_rate_modification['validation_checks'].update(
                        lr_mod_config['validation_checks']
                    )
        
        # 加载输出配置
        if 'output' in config_data:
            output_config = config_data['output']
            self.output_dir = output_config.get('output_dir', self.output_dir)
            self.checkpoint_dir = output_config.get('checkpoint_dir', self.checkpoint_dir)
            self.save_every = output_config.get('save_every', self.save_every)
            self.keep_last_n_checkpoints = output_config.get('keep_last_n_checkpoints', self.keep_last_n_checkpoints)
        
        # 加载设备配置
        if 'device' in config_data:
            device_config = config_data['device']
            device_str = device_config.get('device', 'auto')
            if device_str == 'auto':
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device_str
            self.mixed_precision = device_config.get('mixed_precision', self.mixed_precision)
            self.use_torch_compile = device_config.get('use_torch_compile', self.use_torch_compile)
            self.torch_compile_mode = device_config.get('torch_compile_mode', self.torch_compile_mode)
        
        # 加载多GPU配置
        if 'multi_gpu' in config_data:
            multi_gpu_config = config_data['multi_gpu']
            self.use_data_parallel = multi_gpu_config.get('use_data_parallel', self.use_data_parallel)
            self.gpu_ids = multi_gpu_config.get('gpu_ids', self.gpu_ids)
        
        # 加载交叉验证配置
        if 'cross_validation' in config_data:
            cv_config = config_data['cross_validation']
            self.fold = cv_config.get('fold', self.fold)
            self.num_folds = cv_config.get('num_folds', self.num_folds)
            self.use_all_folds = cv_config.get('use_all_folds', self.use_all_folds)
        
        # 加载测试配置
        if 'test' in config_data:
            test_config = config_data['test']
            self.test_mode = test_config.get('test_mode', self.test_mode)
            self.test_samples = test_config.get('test_samples', self.test_samples)
            self.test_epochs = test_config.get('test_epochs', self.test_epochs)
        
        # 加载日志配置
        if 'logging' in config_data:
            logging_config = config_data['logging']
            self.log_level = logging_config.get('log_level', 'INFO')
        
        # 加载增强功能配置
        if 'enhanced_features' in config_data:
            enhanced_config = config_data['enhanced_features']
            self.enhanced_features.update(enhanced_config)
        
        logger.info("✅ 配置文件加载完成")


class TrainingMetrics:
    """训练指标记录类"""
    
    def __init__(self):
        # 基于epoch的记录（向后兼容）
        self.train_losses = []
        self.valid_losses = []
        self.learning_rates = []
        self.epoch_times = []
        
        # 基于step的记录（新增）
        self.step_losses = []
        self.step_learning_rates = []
        self.steps = []
        
        self.best_valid_loss = float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
    
    def update_train(self, loss: float, lr: float, epoch_time: float):
        """更新训练指标（基于epoch）"""
        self.train_losses.append(loss)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
    
    def update_train_step(self, loss: float, lr: float, step: int):
        """更新训练指标（基于step）"""
        self.step_losses.append(loss)
        self.step_learning_rates.append(lr)
        self.steps.append(step)
    
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
            'step_losses': self.step_losses,
            'step_learning_rates': self.step_learning_rates,
            'steps': self.steps,
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
        
        # RNA评估指标
        if self.config.enhanced_features['evaluation']['compute_structure_metrics']:
            self.enhanced_metrics = {
                'train': RNAEvaluationMetrics(),
                'val': RNAEvaluationMetrics()
            }
            logger.info("✅ RNA专用评估指标已启用")
    
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
        file_handler.setLevel(logging.DEBUG)  # 改为DEBUG级别
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加到logger
        logger.addHandler(file_handler)
        
        # 确保diffold模块的logger也使用DEBUG级别
        diffold_logger = logging.getLogger('diffold')
        diffold_logger.setLevel(logging.DEBUG)
        
        logger.info("日志系统已初始化")
        logger.debug("🐛 DEBUG级别日志已启用，文件和控制台都会记录DEBUG信息")
    
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
            
            # 使用真实数据计算总步数
            steps_per_epoch = len(self.train_loader)
            total_steps = self.config.num_epochs * steps_per_epoch
            
            if self.is_main_process:
                logger.info(f"📊 使用真实数据配置调度器:")
                logger.info(f"  每epoch步数: {steps_per_epoch}")
                logger.info(f"  总训练步数: {total_steps}")
                logger.info(f"  预热步数: {self.config.warmup_steps}")
            
            self.enhanced_optimizer = AdaptiveOptimizer(  # type: ignore[arg-type]
                model=cast(nn.Module, self.model),
                optimizer_name=self.config.enhanced_features['optimizer']['optimizer_name'],
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                scheduler_config={
                    'type': self.config.enhanced_features['optimizer']['scheduler_type'],
                    'warmup_steps': self.config.warmup_steps,
                    'T_max': total_steps,  # 使用真实总步数
                    'eta_min': 1e-6
                },
                gradient_accumulation_steps=self.config.enhanced_features['optimizer']['gradient_accumulation_steps'],
                max_grad_norm=self.config.grad_clip_norm
            )
            
            # 包装原接口
            self.optimizer = self.enhanced_optimizer.optimizer
            self.scheduler = self.enhanced_optimizer.scheduler
            
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
            desc=f"🚀 训练 {epoch+1}/{self.config.num_epochs}",
            leave=False,
            disable=not self.is_main_process,
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        
        for batch_idx, batch in progress_bar:
            if self.config.test_mode and batch_idx >= max_batches:
                break
            
            # 在测试模式下输出样本名称
            if self.config.test_mode and self.local_rank == 0:
                sample_names = batch.get('names', ['unknown'])
                logger.info(f"🔍 当前训练样本: {sample_names}")
            
            batch_start_time = time.time()
            
            try:
                loss = self.train_step(batch, batch_idx, epoch)
                
                if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    num_batches += 1
                    
                    batch_time = time.time() - batch_start_time
                    
                    # 计算当前step
                    current_step = batch_idx + epoch * len(self.train_loader)
                    
                    # 获取当前学习率
                    if self.enhanced_optimizer:
                        current_lr = self.enhanced_optimizer.get_lr()
                    else:
                        current_lr = self.optimizer.param_groups[0]['lr']
                    
                    # 记录基于step的训练指标
                    self.metrics.update_train_step(loss.item(), current_lr, current_step)
                    
                    # 🔥 记录监控数据
                    if (self.training_monitor and 
                        batch_idx % self.config.enhanced_features['monitoring']['monitoring_interval'] == 0):
                        self.training_monitor.log_training_step(
                            step=current_step,
                            epoch=epoch,
                            loss_value=loss.item(),
                            learning_rate=current_lr,
                            batch_time=batch_time,
                            model=self.model
                        )
                    
                    # 美化进度条显示
                    postfix_dict = {
                        '损失': f'{loss.item():.3f}',
                        '平均': f'{total_loss / (batch_idx + 1):.3f}',
                        '学习率': f'{current_lr:.2e}'
                    }
                    if self.device.type == 'cuda':
                        memory_reserved_gb = torch.cuda.memory_reserved(self.device) / 1024**3
                        postfix_dict['显存'] = f"{memory_reserved_gb:.1f}GB"
                    
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
                target_coords=coordinates,
                sequences=sequences
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
                desc="🔍 验证中",
                leave=False,
                disable=not self.is_main_process,
                ncols=120,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
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
                                    confidence_scores=result.get('confidence_logits'),
                                    sequences=sequences
                                )
                            
                            # 美化验证进度条显示
                            postfix_dict = {
                                '验证损失': f'{loss.item():.3f}',
                                '平均': f'{total_loss / (batch_idx + 1):.3f}'
                            }
                            if self.device.type == 'cuda':
                                memory_reserved_gb = torch.cuda.memory_reserved(self.device) / 1024**3
                                postfix_dict['显存'] = f"{memory_reserved_gb:.1f}GB"
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
            'enhanced_enabled': self.enhanced_enabled,
            'current_lr': self.get_current_lr(),  # 保存当前学习率
            'lr_modification_history': getattr(self, 'lr_modification_history', [])  # 保存学习率修改历史
        }
        
        # 🔥 保存增强功能状态
        if self.enhanced_optimizer:
            checkpoint['enhanced_optimizer_stats'] = self.enhanced_optimizer.get_stats()
            # 保存增强优化器的调度器配置
            if hasattr(self.enhanced_optimizer, 'scheduler_config'):
                checkpoint['enhanced_scheduler_config'] = self.enhanced_optimizer.scheduler_config
        
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
        
        current_lr = self.get_current_lr()
        lr_history_len = len(getattr(self, 'lr_modification_history', []))
        logger.info(f"💾 保存检查点: {checkpoint_path}")
        logger.info(f"📊 当前状态: 学习率={current_lr:.6f}, 修改历史={lr_history_len}条")
    
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
    
    def modify_learning_rate(self, new_lr: float, reason: str = "Manual adjustment", force_scheduler_reset: bool = True):
        """智能修改学习率，正确处理调度器状态
        
        Args:
            new_lr: 新的学习率
            reason: 修改原因，用于日志记录
            force_scheduler_reset: 是否强制重置调度器状态以从新学习率开始
        """
        # 检查是否允许运行时修改
        if not self.config.learning_rate_modification.get('enable_runtime_modification', True):
            logger.warning("⚠️ 运行时学习率修改已禁用")
            return
        
        old_lr = self.get_current_lr()
        
        # 验证新学习率
        validation_checks = self.config.learning_rate_modification.get('validation_checks', {})
        min_lr = validation_checks.get('min_lr', 1e-7)
        max_lr = validation_checks.get('max_lr', 1e-1)
        
        if new_lr < min_lr:
            logger.warning(f"⚠️ 新学习率 {new_lr:.6f} 低于最小限制 {min_lr:.6f}")
            new_lr = min_lr
            logger.info(f"📈 学习率已调整至最小值: {new_lr:.6f}")
        
        if new_lr > max_lr:
            logger.warning(f"⚠️ 新学习率 {new_lr:.6f} 高于最大限制 {max_lr:.6f}")
            new_lr = max_lr
            logger.info(f"📉 学习率已调整至最大值: {new_lr:.6f}")
        
        # 检查大幅度变化
        if validation_checks.get('warn_on_large_changes', True):
            threshold = validation_checks.get('large_change_threshold', 10.0)
            if old_lr > 0 and (new_lr / old_lr > threshold or old_lr / new_lr > threshold):
                logger.warning(f"⚠️ 学习率变化过大: {old_lr:.6f} → {new_lr:.6f} "
                              f"(变化倍数: {new_lr/old_lr:.2f})")
        
        # 检测当前调度器状态
        scheduler_info = self._analyze_scheduler_state()
        logger.info(f"📊 调度器状态分析: {scheduler_info}")
        
        # 应用新学习率
        if self.enhanced_optimizer:
            # 使用增强优化器
            for param_group in self.enhanced_optimizer.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            # 使用标准优化器
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        # 智能更新调度器状态
        if force_scheduler_reset and self.scheduler:
            self._update_scheduler_for_new_lr(new_lr, scheduler_info)
        
        if self.config.learning_rate_modification.get('log_lr_changes', True):
            logger.info(f"🔧 学习率修改: {old_lr:.6f} → {new_lr:.6f} ({reason})")
            if force_scheduler_reset:
                logger.info("🔄 调度器状态已同步，将从新学习率开始调度")
        
        # 记录学习率修改历史
        if self.config.learning_rate_modification.get('save_modification_history', True):
            if not hasattr(self, 'lr_modification_history'):
                self.lr_modification_history = []
            
            self.lr_modification_history.append({
                'timestamp': time.time(),
                'old_lr': old_lr,
                'new_lr': new_lr,
                'reason': reason,
                'scheduler_info': scheduler_info,
                'scheduler_reset': force_scheduler_reset
            })
    
    def _analyze_scheduler_state(self) -> Dict[str, Any]:
        """分析当前调度器状态"""
        if not self.scheduler:
            return {'type': 'none', 'phase': 'none'}
        
        scheduler_type = type(self.scheduler).__name__
        info = {
            'type': scheduler_type,
            'last_epoch': getattr(self.scheduler, 'last_epoch', -1),
            'phase': 'unknown'
        }
        
        # 检测预热调度器
        if hasattr(self.scheduler, 'warmup_steps'):
            warmup_steps = self.scheduler.warmup_steps
            current_step = self.scheduler.last_epoch
            
            if current_step < warmup_steps:
                info['phase'] = 'warmup'
                info['warmup_progress'] = f"{current_step}/{warmup_steps}"
            else:
                info['phase'] = 'post_warmup'
                info['post_warmup_steps'] = current_step - warmup_steps
            
            info['warmup_steps'] = warmup_steps
        
        # 检测余弦退火调度器
        if hasattr(self.scheduler, 'T_cur'):
            info['T_cur'] = self.scheduler.T_cur
            info['T_i'] = getattr(self.scheduler, 'T_i', 'unknown')
            info['cycle'] = getattr(self.scheduler, 'cycle', 0)
        
        # 检测基础调度器（嵌套调度器）
        if hasattr(self.scheduler, 'base_scheduler') and self.scheduler.base_scheduler:
            info['has_base_scheduler'] = True
            info['base_scheduler_type'] = type(self.scheduler.base_scheduler).__name__
        
        return info
    
    def _update_scheduler_for_new_lr(self, new_lr: float, scheduler_info: Dict[str, Any]):
        """根据调度器类型智能更新状态"""
        
        # 更新基础学习率
        if hasattr(self.scheduler, 'base_lrs'):
            self.scheduler.base_lrs = [new_lr] * len(self.scheduler.base_lrs)
            logger.info(f"📝 更新调度器基础学习率: {self.scheduler.base_lrs}")
        
        # 处理WarmupLRScheduler
        if 'Warmup' in scheduler_info['type']:
            if scheduler_info['phase'] == 'post_warmup':
                # 如果已过预热期，需要更新基础调度器并重置其状态
                if hasattr(self.scheduler, 'base_scheduler') and self.scheduler.base_scheduler:
                    base_sched = self.scheduler.base_scheduler
                    
                    # 更新基础调度器的base_lrs
                    if hasattr(base_sched, 'base_lrs'):
                        base_sched.base_lrs = [new_lr] * len(base_sched.base_lrs)
                        logger.info("📝 更新基础调度器的base_lrs")
                    
                    # 重置基础调度器的状态，使其从新学习率开始调度
                    base_sched_type = type(base_sched).__name__
                    if 'CosineAnnealing' in base_sched_type:
                        # 重置余弦退火，从新学习率开始下降
                        base_sched.last_epoch = -1  # 重新开始
                        logger.info("🔄 重置基础余弦退火调度器，将从新学习率开始下降")
                    elif 'Plateau' in base_sched_type:
                        # 重置plateau计数器
                        if hasattr(base_sched, 'num_bad_epochs'):
                            base_sched.num_bad_epochs = 0
                        if hasattr(base_sched, 'best'):
                            base_sched.best = float('inf')
                        logger.info("🔄 重置ReduceLROnPlateau状态")
                else:
                    logger.info("📝 预热已结束，但没有基础调度器，学习率将保持恒定")
            else:
                # 在预热阶段，更新预热的目标学习率
                logger.info(f"📝 当前在预热阶段 ({scheduler_info.get('warmup_progress', 'unknown')})，"
                           f"调度器将在预热结束后达到新的目标学习率 {new_lr:.6f}")
        
        # 处理CosineAnnealingWarmRestarts
        elif 'CosineAnnealing' in scheduler_info['type']:
            # 重置当前周期，从新学习率开始下一个退火周期
            if hasattr(self.scheduler, 'T_cur'):
                logger.info(f"🔄 重置余弦退火周期，当前T_cur: {self.scheduler.T_cur} → 0")
                self.scheduler.T_cur = 0
        
        # 处理ReduceLROnPlateau
        elif 'Plateau' in scheduler_info['type']:
            # 重置patience计数器
            if hasattr(self.scheduler, 'num_bad_epochs'):
                self.scheduler.num_bad_epochs = 0
                logger.info("🔄 重置ReduceLROnPlateau的patience计数器")
        
        # 处理增强优化器中的调度器
        if self.enhanced_optimizer and hasattr(self.enhanced_optimizer, 'scheduler'):
            enhanced_scheduler = self.enhanced_optimizer.scheduler
            if enhanced_scheduler and hasattr(enhanced_scheduler, 'base_lrs'):
                enhanced_scheduler.base_lrs = [new_lr] * len(enhanced_scheduler.base_lrs)
                logger.info("📝 更新增强优化器调度器的base_lrs")
    
    def get_current_lr(self) -> float:
        """获取当前学习率"""
        if self.enhanced_optimizer:
            return self.enhanced_optimizer.get_lr()
        else:
            return self.optimizer.param_groups[0]['lr']
    
    def modify_lr_schedule(self, new_schedule_config: Dict[str, Any]):
        """修改学习率调度策略
        
        Args:
            new_schedule_config: 新的调度器配置
        """
        logger.info("🔄 重新配置学习率调度器...")
        
        if self.enhanced_optimizer:
            # 重新创建调度器
            old_scheduler = self.scheduler
            self.enhanced_optimizer.scheduler = self.enhanced_optimizer._create_scheduler(new_schedule_config)
            self.scheduler = self.enhanced_optimizer.scheduler
            
            if old_scheduler and hasattr(old_scheduler, 'last_epoch'):
                # 保持当前的调度器状态
                if self.scheduler and hasattr(self.scheduler, 'last_epoch'):
                    self.scheduler.last_epoch = old_scheduler.last_epoch
            
            logger.info(f"✅ 增强优化器调度器已更新: {new_schedule_config.get('type', 'unknown')}")
        else:
            # 为标准优化器重新创建调度器
            scheduler_type = new_schedule_config.get('type', 'cosine')
            
            if scheduler_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=new_schedule_config.get('T_max', self.config.num_epochs), 
                    eta_min=new_schedule_config.get('eta_min', 1e-6)
                )
            elif scheduler_type == "plateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 
                    mode='min', 
                    factor=new_schedule_config.get('factor', 0.5), 
                    patience=new_schedule_config.get('patience', 10), 
                    verbose=True
                )
            else:
                self.scheduler = None
            
            logger.info(f"✅ 标准调度器已更新: {scheduler_type}")

    def load_checkpoint(self, checkpoint_path: str, 
                       skip_optimizer_state: bool = False,
                       skip_scheduler_state: bool = False) -> int:
        """加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            skip_optimizer_state: 是否跳过优化器状态加载（用于学习率修改）
            skip_scheduler_state: 是否跳过调度器状态加载（用于调度器修改）
        """
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
        
        # 加载优化器状态（可选跳过）
        if not skip_optimizer_state:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("✅ 优化器状态加载成功")
            except Exception as e:
                logger.warning(f"⚠️ 优化器状态加载失败: {e}")
                logger.info("继续使用当前优化器配置")
        else:
            logger.info("🔄 跳过优化器状态加载（准备修改学习率）")
        
        # 加载调度器状态（可选跳过）
        if not skip_scheduler_state and self.scheduler and checkpoint.get('scheduler_state_dict'):
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("✅ 学习率调度器状态加载成功")
            except Exception as e:
                logger.warning(f"⚠️ 调度器状态加载失败: {e}")
                logger.info("使用当前调度器配置继续训练")
        elif skip_scheduler_state:
            logger.info("🔄 跳过调度器状态加载（准备修改调度策略）")
        else:
            logger.info("📝 没有调度器状态需要加载")
        
        # 加载GradScaler状态
        if self.scaler and checkpoint.get('scaler_state_dict'):
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info("✅ GradScaler状态加载成功")
            except Exception as e:
                logger.warning(f"⚠️ GradScaler状态加载失败: {e}")
                logger.info("使用默认GradScaler配置")
        
        # 恢复训练指标
        if 'metrics' in checkpoint:
            metrics_dict = checkpoint['metrics']
            self.metrics.train_losses = metrics_dict.get('train_losses', [])
            self.metrics.valid_losses = metrics_dict.get('valid_losses', [])
            self.metrics.learning_rates = metrics_dict.get('learning_rates', [])
            self.metrics.epoch_times = metrics_dict.get('epoch_times', [])
            # 恢复step-based指标
            self.metrics.step_losses = metrics_dict.get('step_losses', [])
            self.metrics.step_learning_rates = metrics_dict.get('step_learning_rates', [])
            self.metrics.steps = metrics_dict.get('steps', [])
            self.metrics.best_valid_loss = metrics_dict.get('best_valid_loss', float('inf'))
            self.metrics.best_epoch = metrics_dict.get('best_epoch', 0)
            self.metrics.early_stopping_counter = metrics_dict.get('early_stopping_counter', 0)
        
        # 恢复增强优化器状态（如果存在）
        if self.enhanced_optimizer and checkpoint.get('enhanced_optimizer_stats'):
            try:
                self.enhanced_optimizer.load_stats(checkpoint['enhanced_optimizer_stats'])
                logger.info("✅ 增强优化器统计数据加载成功")
            except Exception as e:
                logger.warning(f"⚠️ 增强优化器统计数据加载失败: {e}")
        
        # 恢复学习率修改历史（如果存在）
        if 'lr_modification_history' in checkpoint:
            self.lr_modification_history = checkpoint['lr_modification_history']
            logger.info(f"✅ 学习率修改历史已恢复 ({len(self.lr_modification_history)} 条记录)")
        
        start_epoch = checkpoint['epoch'] + 1
        current_lr = self.get_current_lr()
        logger.info(f"🚀 从epoch {start_epoch}继续训练，当前学习率: {current_lr:.6f}")
        return start_epoch
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.metrics.train_losses and not self.metrics.step_losses:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线 - 基于epoch（保持原有逻辑）
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
        
        # 学习率曲线 - 基于step（修改为使用step）
        if self.metrics.step_learning_rates:
            axes[0, 1].plot(self.metrics.steps, self.metrics.step_learning_rates, 'g-')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule (Step-based)')
            axes[0, 1].grid(True)
        else:
            # 回退到基于epoch的绘图
            if self.metrics.learning_rates:
                epochs = range(1, len(self.metrics.learning_rates) + 1)
                axes[0, 1].plot(epochs, self.metrics.learning_rates, 'g-')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Learning Rate')
                axes[0, 1].set_title('Learning Rate Schedule')
                axes[0, 1].grid(True)
        
        # 训练时间（基于epoch）
        if self.metrics.epoch_times:
            epochs = range(1, len(self.metrics.epoch_times) + 1)
            axes[1, 0].plot(epochs, self.metrics.epoch_times, 'orange')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].set_title('Epoch Training Time')
            axes[1, 0].grid(True)
        
        # 损失分布（基于epoch）
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
    
    def train(self, resume_from: Optional[str] = None, 
              modify_lr: Optional[float] = None,
              lr_schedule_override: Optional[str] = None,
              reset_lr_history: bool = False,
              keep_scheduler_progress: bool = False):
        """主训练循环"""
        if self.is_main_process:
            logger.info("🚀 开始训练...")
            logger.info("=" * 60)
        
        # 🔥 美化增强功能状态显示
        if self.enhanced_enabled and self.is_main_process:
            logger.info("🔥 增强功能状态:")
            feature_icons = {
                'monitoring': '📊',
                'optimizer': '🎯', 
                'dataloader': '⚡',
                'evaluation': '📏',
                'error_recovery': '🛡️'
            }
            for category, features in self.config.enhanced_features.items():
                if isinstance(features, dict) and category != 'enable_enhanced_training':
                    enabled_features = [k for k, v in features.items() if v and k != 'enable_enhanced_training']
                    if enabled_features:
                        icon = feature_icons.get(category, '🔧')
                        logger.info(f"   {icon} {category}: {len(enabled_features)} 项功能已启用")
        
        # 加载检查点（如果指定）
        start_epoch = 0
        if resume_from:
            # 判断是否需要跳过优化器/调度器状态加载
            skip_optimizer = modify_lr is not None
            skip_scheduler = lr_schedule_override is not None or (modify_lr is not None and not keep_scheduler_progress)
            
            if skip_optimizer or skip_scheduler:
                logger.info("🔄 智能加载模式:")
                if skip_optimizer:
                    logger.info("  - 跳过优化器状态加载（准备修改学习率）")
                if skip_scheduler:
                    logger.info("  - 跳过调度器状态加载（准备重置调度策略）")
            
            start_epoch = self.load_checkpoint(
                resume_from, 
                skip_optimizer_state=skip_optimizer,
                skip_scheduler_state=skip_scheduler
            )
            
            # 在跳过状态加载后处理学习率修改选项
            if modify_lr is not None:
                self.modify_learning_rate(
                    modify_lr, 
                    "Command line override after resume",
                    force_scheduler_reset=not keep_scheduler_progress
                )
            
            if lr_schedule_override is not None:
                # 构建新的调度器配置
                new_schedule_config = {
                    'type': lr_schedule_override,
                    'T_max': self.config.num_epochs * len(self.train_loader),
                    'eta_min': 1e-6,
                    'warmup_steps': self.config.warmup_steps,
                    'patience': self.config.patience
                }
                self.modify_lr_schedule(new_schedule_config)
            
            if reset_lr_history:
                self.lr_modification_history = []
                logger.info("🗑️ 学习率修改历史已重置")
        
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
                # 对于step-based调度器，已经在train_step中处理
                # 对于epoch-based调度器（如ReduceLROnPlateau），在这里处理
                if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if valid_loss is not None:
                        self.enhanced_optimizer.scheduler_step(valid_loss)
            else:
                # 使用原版调度器（基于epoch的调度器）
                if self.scheduler is not None:
                    if self.config.scheduler_type == "plateau" and valid_loss is not None:
                        self.scheduler.step(valid_loss)
                    elif self.config.scheduler_type == "cosine":
                        self.scheduler.step()
            
            # 记录指标
            if self.enhanced_optimizer:
                current_lr = self.enhanced_optimizer.get_lr()
            else:
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
                
                if val_metrics and self.is_main_process:
                    # RNA结构评估指标输出
                    metrics_log = "🧬 RNA结构评估:"
                    if 'avg_rmsd' in val_metrics:
                        metrics_log += f" RMSD={val_metrics['avg_rmsd']:.4f}Å"
                    if 'avg_tm_score' in val_metrics:
                        metrics_log += f" TM-score={val_metrics['avg_tm_score']:.3f}"
                        if 'tm_score_good_ratio' in val_metrics:
                            metrics_log += f"({val_metrics['tm_score_good_ratio']:.1%}≥0.45)"
                    if 'avg_lddt' in val_metrics:
                        metrics_log += f" lDDT={val_metrics['avg_lddt']:.1f}"
                        if 'lddt_high_quality_ratio' in val_metrics:
                            metrics_log += f"({val_metrics['lddt_high_quality_ratio']:.1%}≥70)"
                    if 'avg_clash_score' in val_metrics:
                        metrics_log += f" Clash={val_metrics['avg_clash_score']:.1f}%"
                    
                    logger.info(metrics_log)
            
            # 只在主进程写tensorboard
            if self.is_main_process:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                if valid_loss is not None:
                    self.writer.add_scalar('Loss/Valid', valid_loss, epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
                self.writer.add_scalar('EpochTime', epoch_time, epoch)
                
                # 🔥 记录RNA结构评估指标到TensorBoard
                if val_metrics:
                    if 'avg_rmsd' in val_metrics:
                        self.writer.add_scalar('RNA_Metrics/RMSD', val_metrics['avg_rmsd'], epoch)
                    if 'avg_tm_score' in val_metrics:
                        self.writer.add_scalar('RNA_Metrics/TM_Score', val_metrics['avg_tm_score'], epoch)
                        if 'tm_score_good_ratio' in val_metrics:
                            self.writer.add_scalar('RNA_Metrics/TM_Score_Good_Ratio', 
                                                 val_metrics['tm_score_good_ratio'], epoch)
                    if 'avg_lddt' in val_metrics:
                        self.writer.add_scalar('RNA_Metrics/lDDT', val_metrics['avg_lddt'], epoch)
                        if 'lddt_high_quality_ratio' in val_metrics:
                            self.writer.add_scalar('RNA_Metrics/lDDT_High_Quality_Ratio', 
                                                 val_metrics['lddt_high_quality_ratio'], epoch)
                    if 'avg_clash_score' in val_metrics:
                        self.writer.add_scalar('RNA_Metrics/Clash_Score', val_metrics['avg_clash_score'], epoch)
            
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
            
            # 美化Epoch结果输出
            if self.is_main_process:
                # 构建状态图标
                progress_percent = (epoch + 1) / num_epochs * 100
                status_icon = "⭐" if is_best else "📈" if valid_loss and train_loss > valid_loss else "🚀"
                
                # 主要信息行
                log_msg = f"🎯 Epoch {epoch+1:3d}/{num_epochs} [{progress_percent:5.1f}%] "
                log_msg += f"| 训练: {train_loss:.4f}"
                if valid_loss is not None:
                    log_msg += f" | 验证: {valid_loss:.4f}"
                log_msg += f" | LR: {current_lr:.2e} | {epoch_time:.0f}s {status_icon}"
                if is_best:
                    log_msg += " 🏆 NEW BEST"
                
                logger.info(log_msg)
                
                # 时间统计行（简化显示）
                if (epoch + 1) % 5 == 0 or is_best:  # 每5轮或最佳模型时显示详细时间
                    time_msg = f"⏰ 已训练: {format_time(elapsed_time)} | 预计剩余: {format_time(estimated_remaining_time)} | 完成时间: {format_datetime(estimated_completion_time)}"
                    logger.info(time_msg)
            
            # 保存检查点
            if self.is_main_process:
                if (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(epoch, is_best)
                # 绘制训练曲线
                if (epoch + 1) % self.config.plot_every == 0:
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


def run_small_scale_test(fixed_sample_name=None):
    """运行小规模测试 - 包含多GPU环境测试，可指定固定样本"""
    logger.info("🧪 启动多GPU环境小规模测试...")
    
    # 基础测试配置
    config = TrainingConfig()
    config.test_mode = True
    config.test_epochs = 2  # 测试2轮，验证完整流程
    config.test_samples = 1  # 稍微增加样本数测试批次处理
    config.max_sequence_length = 20
    config.num_workers = 2   # 测试数据加载
    config.output_dir = "./test_output"
    config.checkpoint_dir = "./test_checkpoints"
    
    # 🔍 GPU环境检测和配置
    gpu_count = torch.cuda.device_count()
    
    logger.info(f"🖥️  将使用 {gpu_count} 个GPU进行测试")
    
    if gpu_count == 0:
        logger.warning("⚠️  未检测到GPU，使用CPU模式")
        config.device = "cpu"
        config.batch_size = 1
        config.mixed_precision = False
        config.use_data_parallel = False
    elif gpu_count == 1:
        logger.info("📱 单GPU模式测试")
        config.device = "cuda"
        config.batch_size = 1
        config.mixed_precision = True
        config.use_data_parallel = False
    else:
        logger.info(f"🚀 多GPU模式测试 ({gpu_count} GPUs)")
        config.device = "cuda"
        config.batch_size = 1
        config.mixed_precision = True
        config.use_data_parallel = True
        
        # 显示GPU信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # 🔥 启用增强功能进行测试
    if ENHANCED_FEATURES_AVAILABLE:
        config.apply_enhanced_preset('debug')
        logger.info("✅ 增强功能已启用用于测试")
    
    # 📊 测试步骤
    test_results = {
        'gpu_count': gpu_count,
        'config': config.device,
        'batch_size': config.batch_size,
        'steps': []
    }
    
    try:
        # ========== 新增：只用特定样本 ===========
        fixed_train_loader = None
        fixed_valid_loader = None
        if fixed_sample_name is not None:
            logger.info(f"🔒 仅使用指定样本进行测试: {fixed_sample_name}")
            from diffold.dataloader import RNA3DDataset, collate_fn
            dataset = RNA3DDataset(
                data_dir=config.data_dir,
                fold=0,
                split="train",
                max_length=config.max_sequence_length,
                use_msa=config.use_msa,
                cache_dir=None,
                force_reload=False,
                enable_missing_atom_mask=True
            )
            sample = dataset.get_sample_by_name(fixed_sample_name)
            if sample is None:
                raise RuntimeError(f"未找到指定样本: {fixed_sample_name}")
            # 构造只包含该样本的 DataLoader
            from torch.utils.data import DataLoader
            fixed_dataset = [sample]
            fixed_train_loader = DataLoader(fixed_dataset, batch_size=1, collate_fn=collate_fn)
            fixed_valid_loader = fixed_train_loader  # 验证也用同一个
            logger.info(f"✅ 已构造只包含样本 {fixed_sample_name} 的DataLoader")
        # ========== 新增结束 ===========

        trainer = DiffoldTrainer(config)
        test_results['steps'].append('✅ 模型初始化成功')
        logger.info("✅ 模型初始化成功")
        
        # 如果指定了固定样本，替换trainer的数据加载器
        if fixed_train_loader is not None:
            trainer.train_loader = fixed_train_loader
            trainer.valid_loader = fixed_valid_loader
            logger.info(f"✅ 训练/验证均只用样本: {fixed_sample_name}")
        
        # 测试模型设备分布
        if hasattr(trainer.model, 'module'):
            model_devices = set()
            for param in trainer.model.module.parameters():
                model_devices.add(param.device)
            logger.info(f"📍 模型参数分布在设备: {model_devices}")
        
        logger.info("🧪 完整训练流程测试")
        
        # 运行完整训练流程
        trainer.train()
        test_results['steps'].append('✅ 完整训练流程成功')
        
        logger.info("=" * 60)
        logger.info("🎉 多GPU环境测试完成!")
        logger.info("📊 测试结果总结:")
        logger.info(f"   🖥️  GPU数量: {test_results['gpu_count']}")
        logger.info(f"   ⚙️  配置: {test_results['config']}")
        logger.info(f"   📦 批次大小: {test_results['batch_size']}")
        logger.info("   📋 完成步骤:")
        for step in test_results['steps']:
            logger.info(f"      {step}")
        
        logger.info(f"📁 测试输出: {config.output_dir}")
        logger.info(f"📁 测试检查点: {config.checkpoint_dir}")
        
        # 🧹 清理测试文件（可选）
        cleanup_choice = input("\n🧹 是否清理测试文件? (y/n): ").lower().strip()
        if cleanup_choice == 'y':
            import shutil
            import os
            if os.path.exists(config.output_dir):
                shutil.rmtree(config.output_dir)
                logger.info(f"🗑️  已删除: {config.output_dir}")
            if os.path.exists(config.checkpoint_dir):
                shutil.rmtree(config.checkpoint_dir)
                logger.info(f"🗑️  已删除: {config.checkpoint_dir}")
        
        logger.info("✅ 测试完成，环境准备就绪！")
        
    except Exception as e:
        logger.error(f"❌ 多GPU测试失败: {e}")
        logger.error("📋 失败信息:")
        import traceback
        traceback.print_exc()
    
    exit()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Diffold模型训练 - 增强版")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录 (默认使用配置文件中的设置)")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小 (默认使用配置文件中的设置)")
    parser.add_argument("--max_length", type=int, default=None, help="最大序列长度 (默认使用配置文件中的设置)")
    parser.add_argument("--num_workers", type=int, default=None, help="数据加载进程数 (默认使用配置文件中的设置)")
    parser.add_argument("--fold", type=int, default=None, help="交叉验证折数 (0-9) (默认使用配置文件中的设置)")
    parser.add_argument("--use_all_folds", action="store_true", help="使用所有折数的数据进行训练")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数 (默认使用配置文件中的设置)")
    parser.add_argument("--learning_rate", type=float, default=None, help="学习率 (默认使用配置文件中的设置)")
    parser.add_argument("--weight_decay", type=float, default=None, help="权重衰减 (默认使用配置文件中的设置)")
    parser.add_argument("--grad_clip", type=float, default=None, help="梯度裁剪阈值 (默认使用配置文件中的设置)")
    
    # 模型参数
    parser.add_argument("--rhofold_checkpoint", type=str, default=None,
                       help="RhoFold预训练权重路径 (默认使用配置文件中的设置)")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录 (默认使用配置文件中的设置)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="检查点目录 (默认使用配置文件中的设置)")
    parser.add_argument("--save_every", type=int, default=None, help="每N轮保存检查点 (默认使用配置文件中的设置)")
    parser.add_argument("--plot_every", type=int, default=None, help="每N轮保存训练曲线 (默认使用配置文件中的设置)")
    
    # 设备参数
    parser.add_argument("--device", type=str, default=None, help="设备 (auto/cpu/cuda) (默认使用配置文件中的设置)")
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
    
    # 配置文件参数
    parser.add_argument("--config", type=str, default='./config.yaml', help="配置文件路径")
    
    # 日志参数
    parser.add_argument("--log_level", type=str, default=None,
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="日志级别 (默认使用配置文件中的设置)")
    
    # 其他参数
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--test", action="store_true", help="运行多GPU环境小规模测试")
    parser.add_argument("--fixed_sample_name", type=str, default=None, help="指定用于测试的固定样本名称")
    
    # 学习率修改参数
    parser.add_argument("--modify_lr", type=float, default=None, help="在恢复训练时修改学习率")
    parser.add_argument("--lr_schedule_override", type=str, default=None, 
                       help="覆盖调度器类型 (cosine/plateau/warmup_cosine/warmup_cosine_restarts)")
    parser.add_argument("--reset_lr_history", action="store_true", help="重置学习率修改历史")
    parser.add_argument("--keep_scheduler_progress", action="store_true", 
                       help="保持调度器当前进度，不重置调度器状态（默认会重置以从新学习率开始）")

    
    args = parser.parse_args()
    
    # 创建配置（从配置文件或使用默认值）
    config = TrainingConfig(args.config)
    
    # 重新设置日志级别（配置文件优先级更高，只有明确指定命令行参数时才覆盖）
    if args.log_level is not None:
        log_level = args.log_level
        print(f"使用命令行指定的日志级别: {log_level}")
    else:
        log_level = config.log_level
        print(f"使用配置文件中的日志级别: {log_level}")
    
    logger = setup_logging(log_level)
    
    # 打印初始配置信息
    logger.info(f"📋 初始配置信息:")
    logger.info(f"  输出目录: {config.output_dir}")
    logger.info(f"  批次大小: {config.batch_size}")
    logger.info(f"  学习率: {config.learning_rate}")
    logger.info(f"  设备: {config.device}")
    
    # 应用命令行参数到配置（只在明确指定时覆盖配置文件）
    if args.data_dir is not None:
        config.data_dir = args.data_dir
        logger.info(f"使用命令行指定的数据目录: {args.data_dir}")
    
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        logger.info(f"使用命令行指定的批次大小: {args.batch_size}")
    
    if args.max_length is not None:
        config.max_sequence_length = args.max_length
        logger.info(f"使用命令行指定的最大序列长度: {args.max_length}")
    
    if args.num_workers is not None:
        config.num_workers = args.num_workers
        logger.info(f"使用命令行指定的数据加载进程数: {args.num_workers}")
    
    if args.fold is not None:
        config.fold = args.fold
        logger.info(f"使用命令行指定的交叉验证折数: {args.fold}")
    
    if args.epochs is not None:
        config.num_epochs = args.epochs
        logger.info(f"使用命令行指定的训练轮数: {args.epochs}")
    
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
        logger.info(f"使用命令行指定的学习率: {args.learning_rate}")
    
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
        logger.info(f"使用命令行指定的权重衰减: {args.weight_decay}")
    
    if args.grad_clip is not None:
        config.grad_clip_norm = args.grad_clip
        logger.info(f"使用命令行指定的梯度裁剪阈值: {args.grad_clip}")
    
    if args.rhofold_checkpoint is not None:
        config.rhofold_checkpoint = args.rhofold_checkpoint
        logger.info(f"使用命令行指定的模型路径: {args.rhofold_checkpoint}")
    
    if args.output_dir is not None:
        config.output_dir = args.output_dir
        logger.info(f"使用命令行指定的输出目录: {args.output_dir}")
    
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
        logger.info(f"使用命令行指定的检查点目录: {args.checkpoint_dir}")
    
    if args.save_every is not None:
        config.save_every = args.save_every
        logger.info(f"使用命令行指定的保存频率: {args.save_every}")
    
    if args.plot_every is not None:
        config.plot_every = args.plot_every
        logger.info(f"使用命令行指定的绘图频率: {args.plot_every}")
    
    if args.device is not None:
        config.device = args.device
        logger.info(f"使用命令行指定的设备: {args.device}")
    
    # 处理布尔参数
    if args.no_mixed_precision:
        config.mixed_precision = False
        logger.info("禁用混合精度训练")
    
    if args.no_data_parallel:
        config.use_data_parallel = False
        logger.info("禁用数据并行")
    
    if args.use_all_folds:
        config.use_all_folds = True
        logger.info("使用所有折数的数据进行训练")
    
    # 如果是测试模式
    if args.test:
        run_small_scale_test(fixed_sample_name=args.fixed_sample_name)
        return
    
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
    
    # 更新基础配置（只在明确指定时覆盖）
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_length is not None:
        config.max_sequence_length = args.max_length
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.fold is not None:
        config.fold = args.fold
    
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.grad_clip is not None:
        config.grad_clip_norm = args.grad_clip
    
    if args.rhofold_checkpoint is not None:
        config.rhofold_checkpoint = args.rhofold_checkpoint
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.save_every is not None:
        config.save_every = args.save_every
    if args.plot_every is not None:
        config.plot_every = args.plot_every
    
    if args.device is not None:
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
    
    # 处理学习率修改参数（仅在恢复训练时有效）
    if args.modify_lr is not None:
        if args.resume:
            logger.info(f"🔧 准备在恢复训练后修改学习率至: {args.modify_lr}")
        else:
            logger.warning("--modify_lr 仅在使用 --resume 时有效")
    
    if args.lr_schedule_override is not None:
        if args.resume:
            logger.info(f"🔄 准备在恢复训练后修改调度器至: {args.lr_schedule_override}")
        else:
            logger.warning("--lr_schedule_override 仅在使用 --resume 时有效")
    
    if args.reset_lr_history:
        if args.resume:
            logger.info("🗑️ 准备重置学习率修改历史")
        else:
            logger.warning("--reset_lr_history 仅在使用 --resume 时有效")
    
    # 开始训练（传递学习率修改参数）
    trainer.train(
        resume_from=args.resume,
        modify_lr=args.modify_lr if args.resume else None,
        lr_schedule_override=args.lr_schedule_override if args.resume else None,
        reset_lr_history=args.reset_lr_history if args.resume else False,
        keep_scheduler_progress=args.keep_scheduler_progress if args.resume else False
    )
    # DDP结束
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
