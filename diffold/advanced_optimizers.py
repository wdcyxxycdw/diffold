"""
高级优化器和调度器模块
包含学习率预热、改进的调度器、数据预加载优化等
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import threading
import queue
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class WarmupLRScheduler(_LRScheduler):
    """学习率预热调度器"""
    
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 warmup_steps: int,
                 base_scheduler: _LRScheduler = None,
                 warmup_start_lr: float = 1e-7,
                 last_epoch: int = -1,
                 verbose: bool = False):
        """
        Args:
            optimizer: 优化器
            warmup_steps: 预热步数
            base_scheduler: 预热后使用的基础调度器
            warmup_start_lr: 预热起始学习率
            last_epoch: 上次epoch索引
            verbose: 是否详细输出
        """
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.warmup_start_lr = warmup_start_lr
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # 兼容不同版本的PyTorch
        try:
            super().__init__(optimizer, last_epoch, verbose)
        except TypeError:
            # 新版本PyTorch不支持verbose参数
            super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 预热阶段：线性增长 - 修复: 使用 (last_epoch + 1) 确保达到目标学习率
            progress = (self.last_epoch + 1) / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
                    for base_lr in self.base_lrs]
        else:
            # 预热结束后使用基础调度器
            if self.base_scheduler is not None:
                # 调整基础调度器的步数（从0开始）
                effective_step = self.last_epoch - self.warmup_steps
                self.base_scheduler.last_epoch = effective_step
                return self.base_scheduler.get_lr()
            else:
                return self.base_lrs
    
    def step(self, epoch=None):
        super().step(epoch)
        
        # 如果有基础调度器且过了预热期，同步更新
        if (self.base_scheduler is not None and 
            self.last_epoch >= self.warmup_steps):
            self.base_scheduler.step()


class CosineAnnealingWarmRestarts(_LRScheduler):
    """带预热的余弦退火重启调度器"""
    
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 T_0: int,
                 T_mult: int = 1,
                 eta_min: float = 0,
                 warmup_steps: int = 0,
                 warmup_start_lr: float = 1e-7,
                 last_epoch: int = -1):
        """
        Args:
            T_0: 第一次重启前的epoch数
            T_mult: 重启后周期的乘数
            eta_min: 最小学习率
            warmup_steps: 每次重启后的预热步数
            warmup_start_lr: 预热起始学习率
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        
        # 跟踪当前周期
        self.T_cur = 0
        self.T_i = T_0
        self.cycle = 0
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur < self.warmup_steps:
            # 预热阶段 - 修复: 使用 (T_cur + 1) 确保达到目标学习率
            progress = (self.T_cur + 1) / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
                    for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            effective_t = self.T_cur - self.warmup_steps
            effective_T_i = self.T_i - self.warmup_steps
            
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * effective_t / effective_T_i)) / 2
                    for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.T_cur += 1
        
        if self.T_cur >= self.T_i:
            # 重启
            self.cycle += 1
            self.T_cur = 0
            self.T_i = self.T_0 * (self.T_mult ** self.cycle)
            
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class AdaptiveOptimizer:
    """自适应优化器包装器 - 支持标准训练和微调（参数组）"""

    def __init__(self, 
                 model: Optional[nn.Module] = None,
                 param_groups: Optional[List[Dict]] = None,
                 optimizer_name: str = "adamw",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 scheduler_config: Dict[str, Any] = None,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """
        初始化自适应优化器
        
        Args:
            model: 模型（与param_groups二选一）
            param_groups: 参数组列表，用于微调时的分层学习率（与model二选一）
            optimizer_name: 优化器名称
            learning_rate: 学习率（仅在使用model时生效）
            weight_decay: 权重衰减
            scheduler_config: 调度器配置
            gradient_accumulation_steps: 梯度累积步数
            max_grad_norm: 最大梯度范数（梯度裁剪）
            scaler: 混合精度训练的GradScaler
        
        Note:
            - 标准训练模式: 传入 model 参数
            - 微调模式（分层学习率）: 传入 param_groups 参数
        """
        # 验证参数
        if model is None and param_groups is None:
            raise ValueError("必须提供 model 或 param_groups 之一")
        if model is not None and param_groups is not None:
            logger.warning("同时提供了 model 和 param_groups，将使用 param_groups")
        
        self.model = model
        self.param_groups = param_groups
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scaler = scaler
        self.accumulated_steps = 0
        
        # 🔥 根据输入类型确定模式
        if param_groups is not None:
            # 微调模式：使用参数组
            logger.info(f"🎯 使用微调模式（{len(param_groups)}个参数组）")
            self.optimizer = self._create_optimizer_from_param_groups(
                optimizer_name, param_groups, weight_decay
            )
        else:
            # 标准模式：从模型获取参数
            logger.info("📦 使用标准训练模式")
            # 获取可训练参数
            if hasattr(model, 'get_trainable_parameters'):
                trainable_params = model.get_trainable_parameters()
            else:
                trainable_params = model.parameters()
            
            # 创建优化器
            self.optimizer = self._create_optimizer(
                optimizer_name, trainable_params, learning_rate, weight_decay
            )
        
        # 创建调度器
        self.scheduler = self._create_scheduler(scheduler_config)
        
        # 统计信息
        self.stats = {
            'grad_norm_history': [],
            'lr_history': [],
            'param_norm_history': [],
            'update_count': 0
        }
    
    def _create_optimizer_from_param_groups(self,
                                           optimizer_name: str,
                                           param_groups: List[Dict],
                                           weight_decay: float) -> optim.Optimizer:
        """从参数组创建优化器（用于微调）"""
        optimizer_name = optimizer_name.lower()
        
        # 确保每个参数组都有weight_decay（如果没有指定的话）
        for group in param_groups:
            if 'weight_decay' not in group:
                group['weight_decay'] = weight_decay
        
        if optimizer_name == "adamw":
            optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        elif optimizer_name == "adam":
            optimizer = optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(param_groups, momentum=0.9, nesterov=True)
        elif optimizer_name == "lion":
            try:
                from lion_pytorch import Lion
                optimizer = Lion(param_groups)
            except ImportError:
                logger.warning("Lion optimizer不可用，回退到AdamW")
                optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        # 打印各参数组的学习率
        logger.info("📊 参数组学习率配置:")
        for i, group in enumerate(optimizer.param_groups):
            group_name = group.get('name', f'group_{i}')
            param_count = sum(p.numel() for p in group['params'])
            logger.info(f"  {group_name}: LR={group['lr']:.2e}, "
                       f"参数={param_count:,}, WD={group.get('weight_decay', 0):.2e}")
        
        return optimizer
    
    def _create_optimizer(self, 
                         optimizer_name: str, 
                         params, 
                         lr: float, 
                         weight_decay: float) -> optim.Optimizer:
        """创建优化器（标准模式）"""
        optimizer_name = optimizer_name.lower()
        
        if optimizer_name == "adamw":
            return optim.AdamW(
                params, lr=lr, weight_decay=weight_decay,
                betas=(0.9, 0.999), eps=1e-8
            )
        elif optimizer_name == "adam":
            return optim.Adam(
                params, lr=lr, weight_decay=weight_decay,
                betas=(0.9, 0.999), eps=1e-8
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                params, lr=lr, weight_decay=weight_decay,
                momentum=0.9, nesterov=True
            )
        elif optimizer_name == "lion":
            try:
                from lion_pytorch import Lion
                return Lion(params, lr=lr, weight_decay=weight_decay)
            except ImportError:
                logger.warning("Lion optimizer不可用，回退到AdamW")
                return optim.AdamW(
                    params, lr=lr, weight_decay=weight_decay,
                    betas=(0.9, 0.999), eps=1e-8
                )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    def _create_scheduler(self, config: Dict[str, Any] = None) -> Optional[_LRScheduler]:
        """创建学习率调度器"""
        if config is None:
            return None
        
        scheduler_type = config.get('type', 'cosine')
        
        if scheduler_type == 'warmup_cosine':
            # 计算预热后的总步数
            total_steps = config.get('T_max', 100)
            warmup_steps = config.get('warmup_steps', 100)
            remaining_steps = total_steps - warmup_steps
            
            base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,  # 预热后的剩余步数
                eta_min=config.get('eta_min', 1e-6)
            )
            return WarmupLRScheduler(
                self.optimizer,
                warmup_steps=warmup_steps,
                base_scheduler=base_scheduler,
                warmup_start_lr=config.get('warmup_start_lr', 1e-7)
            )
        
        elif scheduler_type == 'warmup_cosine_restarts':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.get('T_0', 50),
                T_mult=config.get('T_mult', 2),
                eta_min=config.get('eta_min', 1e-6),
                warmup_steps=config.get('warmup_steps', 100),
                warmup_start_lr=config.get('warmup_start_lr', 1e-6)
            )
        
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.get('factor', 0.5),
                patience=config.get('patience', 10),
                verbose=True
            )
        
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('T_max', 100),
                eta_min=config.get('eta_min', 1e-6)
            )
        
        else:
            logger.warning(f"未知的调度器类型: {scheduler_type}")
            return None
    
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()
    
    def backward(self, loss: torch.Tensor):
        """反向传播"""
        # 缩放损失以考虑梯度累积
        scaled_loss = loss / self.gradient_accumulation_steps
        
        # 混合精度训练处理
        if self.scaler is not None:
            # 使用scaler进行反向传播
            self.scaler.scale(scaled_loss).backward()
        else:
            # 标准反向传播
            scaled_loss.backward()
        
        self.accumulated_steps += 1
        
        # 梯度累积
        if self.accumulated_steps >= self.gradient_accumulation_steps:
            self.step()
            self.accumulated_steps = 0
    
    def step(self):
        """执行优化步骤"""
        # 计算梯度范数
        grad_norm = self._compute_grad_norm()
        self.stats['grad_norm_history'].append(grad_norm)
        
        # 梯度裁剪
        if self.max_grad_norm > 0:
            # 在混合精度训练中，需要先unscale梯度
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            # 🔥 获取所有可训练参数（兼容参数组模式）
            all_params = self._get_all_trainable_params()
            torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
        
        # 优化器步骤
        if self.scaler is not None:
            # 混合精度训练：通过scaler执行优化器步骤
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 标准训练：直接执行优化器步骤
            self.optimizer.step()
        
        self.stats['update_count'] += 1
        
        # 调度器步骤（对于step-based调度器）
        if self.scheduler is not None:
            # 对于step-based调度器，在每个优化器步骤后调用
            if isinstance(self.scheduler, (WarmupLRScheduler, CosineAnnealingWarmRestarts)):
                self.scheduler.step()
            # 对于epoch-based调度器（如ReduceLROnPlateau），不在这里调用
        
        # 记录学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.stats['lr_history'].append(current_lr)
        
        # 记录参数范数
        param_norm = self._compute_param_norm()
        self.stats['param_norm_history'].append(param_norm)
    
    def scheduler_step(self, metrics: float = None):
        """调度器步骤"""
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if metrics is not None:
                    self.scheduler.step(metrics)
            else:
                self.scheduler.step()
    
    def _get_all_trainable_params(self):
        """获取所有可训练参数（兼容标准模式和参数组模式）"""
        if self.param_groups is not None:
            # 参数组模式：从optimizer的param_groups获取
            all_params = []
            for group in self.optimizer.param_groups:
                all_params.extend([p for p in group['params'] if p.requires_grad])
            return all_params
        else:
            # 标准模式：从模型获取
            if hasattr(self.model, 'get_trainable_parameters'):
                return list(self.model.get_trainable_parameters())
            else:
                return [p for p in self.model.parameters() if p.requires_grad]
    
    def _compute_grad_norm(self) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        
        parameters = self._get_all_trainable_params()
        
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        return total_norm ** (1. / 2)
    
    def _compute_param_norm(self) -> float:
        """计算参数范数"""
        total_norm = 0.0
        
        parameters = self._get_all_trainable_params()
        
        for param in parameters:
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
        
        return total_norm ** (1. / 2)
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def get_stats(self) -> Dict[str, Any]:
        """获取优化器统计信息"""
        stats = self.stats.copy()
        
        if self.stats['grad_norm_history']:
            stats['avg_grad_norm'] = sum(self.stats['grad_norm_history']) / len(self.stats['grad_norm_history'])
            stats['max_grad_norm'] = max(self.stats['grad_norm_history'])
        
        if self.stats['param_norm_history']:
            stats['avg_param_norm'] = sum(self.stats['param_norm_history']) / len(self.stats['param_norm_history'])
        
        return stats
    
    def load_stats(self, stats_dict: Dict[str, Any]):
        """加载优化器统计信息
        
        Args:
            stats_dict: 统计信息字典
        """
        # 恢复基本统计信息
        for key in ['update_count', 'grad_norm_history', 'lr_history', 'param_norm_history']:
            if key in stats_dict:
                self.stats[key] = stats_dict[key]
        
        # 恢复累积步数
        self.accumulated_steps = stats_dict.get('accumulated_steps', 0)
        
        logger.info(f"📊 加载优化器统计: 更新次数={self.stats['update_count']}, "
                   f"梯度历史={len(self.stats['grad_norm_history'])}, "
                   f"学习率历史={len(self.stats['lr_history'])}")
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'update_count': 0,
            'grad_norm_history': [],
            'lr_history': [],
            'param_norm_history': []
        }
        self.accumulated_steps = 0
        logger.info("📊 优化器统计信息已重置")


class DataLoaderOptimizer:
    """数据加载器优化器"""
    
    def __init__(self, 
                 base_dataloader,
                 prefetch_factor: int = 2,
                 cache_size: int = 100,
                 enable_prefetch: bool = True):
        """
        Args:
            base_dataloader: 基础数据加载器
            prefetch_factor: 预取因子
            cache_size: 缓存大小
            enable_prefetch: 是否启用预取
        """
        self.base_dataloader = base_dataloader
        self.prefetch_factor = prefetch_factor
        self.cache_size = cache_size
        self.enable_prefetch = enable_prefetch
        
        # 缓存
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 预取队列
        self.prefetch_queue = queue.Queue(maxsize=prefetch_factor * 2)
        self.prefetch_thread = None
        self.stop_prefetch = threading.Event()
    
    def __iter__(self):
        if self.enable_prefetch:
            return self._prefetch_iter()
        else:
            return iter(self.base_dataloader)
    
    def __len__(self):
        return len(self.base_dataloader)
    
    def _prefetch_iter(self):
        """预取迭代器"""
        # 启动预取线程
        self.stop_prefetch.clear()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.start()
        
        try:
            while True:
                try:
                    batch = self.prefetch_queue.get(timeout=30)  # 30秒超时
                    if batch is None:  # 结束标志
                        break
                    yield batch
                except queue.Empty:
                    logger.warning("预取队列超时，可能出现性能问题")
                    break
        finally:
            # 停止预取线程
            self.stop_prefetch.set()
            if self.prefetch_thread and self.prefetch_thread.is_alive():
                self.prefetch_thread.join(timeout=5)
    
    def _prefetch_worker(self):
        """预取工作线程"""
        try:
            for batch in self.base_dataloader:
                if self.stop_prefetch.is_set():
                    break
                
                # 数据预处理（如移动到GPU）
                processed_batch = self._preprocess_batch(batch)
                
                # 放入队列
                self.prefetch_queue.put(processed_batch)
            
            # 发送结束信号
            self.prefetch_queue.put(None)
            
        except Exception as e:
            logger.error(f"预取线程出错: {e}")
            self.prefetch_queue.put(None)
    
    def _preprocess_batch(self, batch):
        """预处理batch数据"""
        # 这里可以添加数据预处理逻辑
        # 比如移动到GPU、数据格式转换等
        return batch
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }