"""
增强的训练配置
集成所有新的保障措施和优化功能
"""

import torch
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EnhancedTrainingConfig:
    """增强的训练配置类"""
    
    def __init__(self):
        # 基础配置
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
        
        # 🔥 新增：高级优化器配置
        self.optimizer_config = {
            'name': 'adamw',  # 'adamw', 'adam', 'sgd', 'lion'
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0
        }
        
        # 🔥 新增：增强的学习率调度器配置
        self.scheduler_config = {
            'type': 'warmup_cosine',  # 'warmup_cosine', 'warmup_cosine_restarts', 'plateau', 'cosine'
            'warmup_epochs': 5,
            'warmup_start_lr': 1e-7,
            'T_max': 100,
            'eta_min': 1e-6,
            # 余弦重启参数
            'T_0': 50,
            'T_mult': 2,
            # plateau参数
            'factor': 0.5,
            'patience': 10
        }
        
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
        self.use_data_parallel = True
        self.gpu_ids = None
        
        # 🔥 新增：性能监控配置
        self.monitoring_config = {
            'enable_performance_monitoring': True,
            'enable_memory_monitoring': True,
            'enable_health_checking': True,
            'monitoring_interval': 10,  # 每10个batch记录一次
            'save_monitoring_plots': True,
            'memory_cleanup_threshold': 0.85
        }
        
        # 🔥 新增：数据加载优化配置
        self.dataloader_config = {
            'enable_prefetch': True,
            'prefetch_factor': 2,
            'cache_size': 100,
            'pin_memory': True,
            'persistent_workers': True
        }
        
        # 🔥 新增：自适应批次大小配置
        self.adaptive_batch_config = {
            'enable_adaptive_batch_size': False,  # 默认关闭，需要时启用
            'min_batch_size': 1,
            'max_batch_size': 16,
            'oom_detection': True
        }
        
        # 🔥 新增：评估指标配置
        self.evaluation_config = {
            'compute_structure_metrics': True,
            'compute_confidence_metrics': True,
            'save_predictions': False,  # 是否保存预测结果
            'evaluation_frequency': 1  # 每N个epoch评估一次
        }
        
        # 🔥 新增：错误恢复配置
        self.error_recovery_config = {
            'auto_retry_on_oom': True,
            'max_retry_attempts': 3,
            'reduce_batch_size_on_oom': True,
            'fallback_to_cpu_on_gpu_error': False
        }
        
        # 小规模测试配置
        self.test_mode = False
        self.test_samples = 10
        self.test_epochs = 5
        
        # 交叉验证配置
        self.fold = 0
        self.num_folds = 10
        self.use_all_folds = False
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """获取优化器配置"""
        return {
            'optimizer_name': self.optimizer_config['name'],
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'scheduler_config': self.scheduler_config,
            'gradient_accumulation_steps': self.optimizer_config['gradient_accumulation_steps'],
            'max_grad_norm': self.optimizer_config['max_grad_norm']
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        return self.monitoring_config.copy()
    
    def get_dataloader_config(self) -> Dict[str, Any]:
        """获取数据加载器配置"""
        config = self.dataloader_config.copy()
        config.update({
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'shuffle': True
        })
        return config
    
    def enable_performance_mode(self):
        """启用性能优化模式"""
        logger.info("🚀 启用性能优化模式")
        
        # 启用所有优化功能
        self.dataloader_config['enable_prefetch'] = True
        self.dataloader_config['prefetch_factor'] = 3
        self.dataloader_config['pin_memory'] = True
        self.dataloader_config['persistent_workers'] = True
        
        # 启用梯度累积
        if self.batch_size < 8:
            self.optimizer_config['gradient_accumulation_steps'] = max(1, 8 // self.batch_size)
            logger.info(f"启用梯度累积，步数: {self.optimizer_config['gradient_accumulation_steps']}")
        
        # 使用更好的调度器
        self.scheduler_config['type'] = 'warmup_cosine'
        
        # 启用监控
        self.monitoring_config['enable_performance_monitoring'] = True
        self.monitoring_config['enable_memory_monitoring'] = True
        
        logger.info("✅ 性能优化模式已启用")
    
    def enable_safety_mode(self):
        """启用安全模式"""
        logger.info("🛡️ 启用安全模式")
        
        # 启用所有安全检查
        self.monitoring_config['enable_health_checking'] = True
        self.error_recovery_config['auto_retry_on_oom'] = True
        self.error_recovery_config['reduce_batch_size_on_oom'] = True
        
        # 更保守的批次大小
        if self.batch_size > 4:
            self.batch_size = 4
            logger.info("降低批次大小为4以提高稳定性")
        
        # 启用自适应批次大小
        self.adaptive_batch_config['enable_adaptive_batch_size'] = True
        
        # 更频繁的检查点保存
        self.save_every = min(self.save_every, 3)
        
        logger.info("✅ 安全模式已启用")
    
    def enable_memory_efficient_mode(self):
        """启用内存高效模式"""
        logger.info("💾 启用内存高效模式")
        
        # 降低批次大小
        self.batch_size = max(1, self.batch_size // 2)
        
        # 启用梯度累积补偿
        self.optimizer_config['gradient_accumulation_steps'] *= 2
        
        # 降低序列长度
        self.max_sequence_length = min(self.max_sequence_length, 200)
        
        # 启用更激进的内存清理
        self.monitoring_config['memory_cleanup_threshold'] = 0.75
        
        # 减少预取
        self.dataloader_config['prefetch_factor'] = 1
        
        logger.info(f"批次大小调整为: {self.batch_size}")
        logger.info(f"梯度累积步数: {self.optimizer_config['gradient_accumulation_steps']}")
        logger.info("✅ 内存高效模式已启用")
    
    def enable_debug_mode(self):
        """启用调试模式"""
        logger.info("🐛 启用调试模式")
        
        # 启用详细监控
        self.monitoring_config['monitoring_interval'] = 1
        self.monitoring_config['save_monitoring_plots'] = True
        
        # 更频繁的保存
        self.save_every = 1
        
        # 启用所有评估指标
        self.evaluation_config['compute_structure_metrics'] = True
        self.evaluation_config['compute_confidence_metrics'] = True
        self.evaluation_config['save_predictions'] = True
        
        # 小批次测试
        if not self.test_mode:
            self.test_mode = True
            self.test_epochs = 3
            self.test_samples = 12
        
        logger.info("✅ 调试模式已启用")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_dir': self.data_dir,
            'batch_size': self.batch_size,
            'max_sequence_length': self.max_sequence_length,
            'num_workers': self.num_workers,
            'use_msa': self.use_msa,
            'rhofold_checkpoint': self.rhofold_checkpoint,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'grad_clip_norm': self.grad_clip_norm,
            'optimizer_config': self.optimizer_config,
            'scheduler_config': self.scheduler_config,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'save_every': self.save_every,
            'keep_last_n_checkpoints': self.keep_last_n_checkpoints,
            'validate_every': self.validate_every,
            'early_stopping_patience': self.early_stopping_patience,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'use_data_parallel': self.use_data_parallel,
            'gpu_ids': self.gpu_ids,
            'monitoring_config': self.monitoring_config,
            'dataloader_config': self.dataloader_config,
            'adaptive_batch_config': self.adaptive_batch_config,
            'evaluation_config': self.evaluation_config,
            'error_recovery_config': self.error_recovery_config,
            'test_mode': self.test_mode,
            'test_samples': self.test_samples,
            'test_epochs': self.test_epochs,
            'fold': self.fold,
            'num_folds': self.num_folds,
            'use_all_folds': self.use_all_folds
        }


def get_recommended_config(scenario: str = "balanced") -> EnhancedTrainingConfig:
    """获取推荐配置
    
    Args:
        scenario: 配置场景
            - "balanced": 平衡模式（推荐）
            - "performance": 性能优先
            - "safety": 安全优先
            - "memory": 内存高效
            - "debug": 调试模式
    """
    config = EnhancedTrainingConfig()
    
    if scenario == "performance":
        config.enable_performance_mode()
    elif scenario == "safety":
        config.enable_safety_mode()
    elif scenario == "memory":
        config.enable_memory_efficient_mode()
    elif scenario == "debug":
        config.enable_debug_mode()
    elif scenario == "balanced":
        # 平衡模式：启用部分优化
        config.dataloader_config['enable_prefetch'] = True
        config.monitoring_config['enable_performance_monitoring'] = True
        config.monitoring_config['enable_memory_monitoring'] = True
        config.error_recovery_config['auto_retry_on_oom'] = True
        logger.info("✅ 使用平衡配置模式")
    else:
        logger.warning(f"未知的配置场景: {scenario}，使用默认配置")
    
    return config


# 预定义配置模板
PRESET_CONFIGS = {
    "small_model": {
        "batch_size": 8,
        "max_sequence_length": 128,
        "learning_rate": 2e-4,
        "scheduler_config": {"type": "warmup_cosine", "warmup_epochs": 3}
    },
    
    "large_model": {
        "batch_size": 2,
        "max_sequence_length": 512,
        "learning_rate": 5e-5,
        "optimizer_config": {"gradient_accumulation_steps": 4},
        "scheduler_config": {"type": "warmup_cosine_restarts", "T_0": 30}
    },
    
    "production": {
        "save_every": 10,
        "keep_last_n_checkpoints": 3,
        "early_stopping_patience": 30,
        "monitoring_config": {"monitoring_interval": 50}
    },
    
    "research": {
        "save_every": 1,
        "evaluation_config": {
            "compute_structure_metrics": True,
            "save_predictions": True
        },
        "monitoring_config": {"save_monitoring_plots": True}
    }
}


def create_config_from_preset(preset_name: str, 
                            base_scenario: str = "balanced") -> EnhancedTrainingConfig:
    """从预设创建配置
    
    Args:
        preset_name: 预设名称
        base_scenario: 基础场景
    """
    if preset_name not in PRESET_CONFIGS:
        logger.warning(f"未知的预设: {preset_name}，使用默认配置")
        return get_recommended_config(base_scenario)
    
    config = get_recommended_config(base_scenario)
    preset = PRESET_CONFIGS[preset_name]
    
    # 应用预设配置
    for key, value in preset.items():
        if hasattr(config, key):
            if isinstance(value, dict) and isinstance(getattr(config, key), dict):
                getattr(config, key).update(value)
            else:
                setattr(config, key, value)
    
    logger.info(f"✅ 应用预设配置: {preset_name}")
    return config 