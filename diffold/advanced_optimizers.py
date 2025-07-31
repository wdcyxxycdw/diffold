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
    """自适应优化器包装器"""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer_name: str = "adamw",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 scheduler_config: Dict[str, Any] = None,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0):
        """
        Args:
            model: 要优化的模型
            optimizer_name: 优化器名称 ("adamw", "adam", "sgd", "lion")
            learning_rate: 学习率
            weight_decay: 权重衰减
            scheduler_config: 调度器配置
            gradient_accumulation_steps: 梯度累积步数
            max_grad_norm: 最大梯度范数
        """
        self.model = model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.accumulated_steps = 0
        
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
    
    def _create_optimizer(self, 
                         optimizer_name: str, 
                         params, 
                         lr: float, 
                         weight_decay: float) -> optim.Optimizer:
        """创建优化器"""
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
            if hasattr(self.model, 'get_trainable_parameters'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_parameters(), 
                    self.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
        
        # 优化器步骤
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
    
    def _compute_grad_norm(self) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        
        if hasattr(self.model, 'get_trainable_parameters'):
            parameters = self.model.get_trainable_parameters()
        else:
            parameters = self.model.parameters()
        
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        return total_norm ** (1. / 2)
    
    def _compute_param_norm(self) -> float:
        """计算参数范数"""
        total_norm = 0.0
        
        if hasattr(self.model, 'get_trainable_parameters'):
            parameters = self.model.get_trainable_parameters()
        else:
            parameters = self.model.parameters()
        
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


class RNAEvaluationMetrics:
    """RNA结构评估指标计算器 - 专为RNA结构预测设计"""
    
    def __init__(self):
        """初始化RNA评估指标"""
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.total_loss = 0.0
        self.total_samples = 0
        self.batch_count = 0
        
        # 损失分解
        self.loss_components = defaultdict(float)
        
        # RNA核心结构评估指标
        self.rmsd_values = []
        self.tm_scores = []  # RNA TM-score
        self.lddt_scores = []  # RNA lDDT
        self.clash_scores = []  # RNA clash score
        
        # 置信度指标
        self.confidence_scores = []
    
    def update(self, 
              loss: float, 
              batch_size: int,
              loss_breakdown: Dict[str, float] = None,
              predicted_coords: torch.Tensor = None,
              target_coords: torch.Tensor = None,
              confidence_scores: torch.Tensor = None):
        """更新指标"""
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        self.batch_count += 1
        
        # 更新损失分解
        if loss_breakdown:
            for component, value in loss_breakdown.items():
                self.loss_components[component] += value * batch_size
        
        # 更新RNA结构指标
        if predicted_coords is not None and target_coords is not None:
            self._update_rna_structure_metrics(predicted_coords, target_coords)
        
        # 更新置信度指标
        if confidence_scores is not None:
            # 检查 confidence_scores 的类型
            if hasattr(confidence_scores, 'plddt') and confidence_scores.plddt is not None:
                # ConfidenceHeadLogits 对象，提取 plddt 分数
                plddt_scores = confidence_scores.plddt
                if torch.is_tensor(plddt_scores):
                    self.confidence_scores.extend(plddt_scores.flatten().tolist())
            elif torch.is_tensor(confidence_scores):
                # 普通张量
                self.confidence_scores.extend(confidence_scores.flatten().tolist())
    
    def _update_rna_structure_metrics(self, 
                                    predicted_coords: torch.Tensor, 
                                    target_coords: torch.Tensor):
        """更新RNA结构评估指标"""
        try:
            # 计算RMSD
            rmsd = self._compute_rmsd(predicted_coords, target_coords)
            self.rmsd_values.extend(rmsd.tolist())
            
            # 计算RNA TM-score
            tm_scores = self._compute_rna_tm_score(predicted_coords, target_coords)
            self.tm_scores.extend(tm_scores.tolist())
            
            # 计算RNA lDDT
            lddt_scores = self._compute_rna_lddt(predicted_coords, target_coords)
            self.lddt_scores.extend(lddt_scores.tolist())
            
            # 计算RNA clash score
            clash_scores = self._compute_rna_clash_score(predicted_coords)
            self.clash_scores.extend(clash_scores.tolist())
            
        except Exception as e:
            logger.warning(f"RNA结构指标计算失败: {e}")
    
    def _compute_rmsd(self, 
                     pred_coords: torch.Tensor, 
                     target_coords: torch.Tensor) -> torch.Tensor:
        """计算结构对齐后的RMSD (使用Kabsch算法)"""
        # 确保坐标形状匹配
        if pred_coords.shape != target_coords.shape:
            min_len = min(pred_coords.shape[1], target_coords.shape[1])
            pred_coords = pred_coords[:, :min_len]
            target_coords = target_coords[:, :min_len]
        
        # 检查坐标维度和有效性
        if pred_coords.dim() != 3 or target_coords.dim() != 3:
            logger.warning(f"坐标维度错误: pred {pred_coords.shape}, target {target_coords.shape}")
            # 回退到原始计算
            diff = pred_coords - target_coords
            squared_diff = torch.sum(diff ** 2, dim=-1)
            return torch.sqrt(torch.mean(squared_diff, dim=-1))
        
        if pred_coords.shape[-1] != 3 or target_coords.shape[-1] != 3:
            logger.warning(f"坐标不是3D: pred {pred_coords.shape}, target {target_coords.shape}")
            # 回退到原始计算
            diff = pred_coords - target_coords
            squared_diff = torch.sum(diff ** 2, dim=-1)
            return torch.sqrt(torch.mean(squared_diff, dim=-1))
        
        # 快速检查：如果坐标完全相同，直接返回0
        if torch.allclose(pred_coords, target_coords, atol=1e-6):
            return torch.zeros(pred_coords.shape[0], device=pred_coords.device)
        
        try:
            return self._compute_aligned_rmsd_kabsch(pred_coords, target_coords)
        except Exception as e:
            logger.warning(f"Kabsch对齐失败，使用未对齐RMSD: {e}")
            # 回退到原始计算
            diff = pred_coords - target_coords
            squared_diff = torch.sum(diff ** 2, dim=-1)
            return torch.sqrt(torch.mean(squared_diff, dim=-1))
    
    def _compute_aligned_rmsd_kabsch(self, 
                                   pred_coords: torch.Tensor, 
                                   target_coords: torch.Tensor) -> torch.Tensor:
        """使用标准Kabsch算法计算对齐后的RMSD
        
        标准Kabsch算法实现，参考经典论文和可靠实现
        W. Kabsch (1976) "A solution for the best rotation to relate two sets of vectors"
        
        Args:
            pred_coords: [batch_size, n_atoms, 3] 预测坐标 (要被对齐的点集)
            target_coords: [batch_size, n_atoms, 3] 目标坐标 (参考点集)
            
        Returns:
            rmsd: [batch_size] 每个样本的对齐RMSD
        """
        batch_size, n_atoms, _ = pred_coords.shape
        device = pred_coords.device
        
        # 快速检查：如果坐标完全相同，直接返回0
        if torch.allclose(pred_coords, target_coords, atol=1e-6):
            return torch.zeros(batch_size, device=device)
        
        # 1. 质心对齐
        pred_centroid = torch.mean(pred_coords, dim=1, keepdim=True)  # [B, 1, 3]
        target_centroid = torch.mean(target_coords, dim=1, keepdim=True)  # [B, 1, 3]
        
        pred_centered = pred_coords - pred_centroid  # [B, N, 3]
        target_centered = target_coords - target_centroid  # [B, N, 3]
        
        # 2. 计算交叉协方差矩阵 H = pred_centered^T @ target_centered
        H = torch.bmm(pred_centered.transpose(-2, -1), target_centered)  # [B, 3, 3]
        
        # 3. SVD分解: H = U @ S @ V^T
        try:
            # 使用稳定的SVD分解
            U, S, Vt = torch.linalg.svd(H) if hasattr(torch.linalg, 'svd') else torch.svd(H)
        except Exception as e:
            logger.warning(f"SVD分解失败: {e}")
            # 回退到未对齐RMSD
            diff = pred_coords - target_coords
            return torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1), dim=-1))
        
        # 4. 计算旋转矩阵 R = V @ U^T
        # 注意：这里V是Vt的转置
        V = Vt.transpose(-2, -1)  # [B, 3, 3]
        R = torch.bmm(U, Vt)  # [B, 3, 3]
        
        # 5. 处理反射情况：确保det(R) = +1（右手系）
        det_R = torch.det(R)  # [B]
        
        # 对于det(R) < 0的情况，修正V的最后一列
        det_R = torch.det(R)
        neg_mask = det_R < 0
        if neg_mask.any():
            # 在 U 的最后一列乘 -1 再重算 R
            U_fix = U.clone()
            U_fix[neg_mask, :, -1] *= -1
            R[neg_mask] = torch.bmm(U_fix[neg_mask], Vt[neg_mask])
        
        # 6. 应用最优旋转：pred_aligned = pred_centered @ R
        pred_aligned = torch.bmm(pred_centered, R)  # [B, N, 3]
        
        # 7. 计算对齐后的RMSD
        diff = pred_aligned - target_centered  # [B, N, 3]
        squared_distances = torch.sum(diff ** 2, dim=-1)  # [B, N]
        rmsd = torch.sqrt(torch.mean(squared_distances, dim=-1))  # [B]
        
        # 8. 验证结果的合理性
        # 计算未对齐RMSD作为上界
        unaligned_diff = pred_coords - target_coords
        unaligned_rmsd = torch.sqrt(torch.mean(torch.sum(unaligned_diff ** 2, dim=-1), dim=-1))
        
        # 正常情况下，对齐后的RMSD应该小于等于未对齐的RMSD
        # 如果不是，说明算法有问题，但我们仍然返回结果并记录警告
        worse_alignment = rmsd > unaligned_rmsd * 1.01  # 允许1%的数值误差
        if worse_alignment.any():
            logger.warning(f"警告: {worse_alignment.sum()}/{batch_size} 个样本的对齐RMSD大于未对齐RMSD")
        
        return torch.clamp(rmsd, min=0.0)
    
    def _compute_rna_tm_score(self, 
                         pred_coords: torch.Tensor, 
                         target_coords: torch.Tensor) -> torch.Tensor:
        """计算RNA TM-score (Template Modeling score)
        
        RNA TM-score是衡量两个RNA结构相似性的指标，范围0-1，值越高越好
        参考Zhang Lab的RNA-align方法，使用适合RNA的归一化因子
        
        Args:
            pred_coords: [batch_size, n_atoms, 3] 预测坐标
            target_coords: [batch_size, n_atoms, 3] 目标坐标
            
        Returns:
            tm_score: [batch_size] 每个样本的RNA TM-score
        """
        batch_size, n_atoms, _ = pred_coords.shape
        device = pred_coords.device
        
        # 确保坐标形状匹配
        if pred_coords.shape != target_coords.shape:
            min_len = min(pred_coords.shape[1], target_coords.shape[1])
            pred_coords = pred_coords[:, :min_len]
            target_coords = target_coords[:, :min_len]
            n_atoms = min_len
        
        try:
            # 1. 获取最优对齐（使用与RMSD相同的对齐方法）
            pred_centroid = torch.mean(pred_coords, dim=1, keepdim=True)
            target_centroid = torch.mean(target_coords, dim=1, keepdim=True)
            
            pred_centered = pred_coords - pred_centroid
            target_centered = target_coords - target_centroid
            
            # 计算旋转矩阵
            H = torch.bmm(pred_centered.transpose(-2, -1), target_centered)
            
            # SVD分解获得最优旋转
            U, S, Vt = torch.linalg.svd(H) if hasattr(torch.linalg, 'svd') else torch.svd(H)
            V = Vt.transpose(-2, -1)
            R = torch.bmm(U, Vt)
            
            # 处理反射
            det_R = torch.det(R)
            neg_mask = det_R < 0
            if neg_mask.any():
                U_fix = U.clone()
                U_fix[neg_mask, :, -1] *= -1
                R[neg_mask] = torch.bmm(U_fix[neg_mask], Vt[neg_mask])
            
            # 应用旋转
            pred_aligned = torch.bmm(pred_centered, R)
            
            # 2. 计算距离
            distances = torch.sqrt(torch.sum((pred_aligned - target_centered) ** 2, dim=-1))  # [B, N]
            
            # 3. RNA TM-score计算 - 使用RNA优化的d0公式
            L = n_atoms
            
            # RNA专用的d0计算（基于RNA-align方法）
            if L > 30:
                d0 = 1.24 * ((L - 15) ** (1/3)) - 1.8 + 0.3  # 增加0.3埃的偏移
                d0 = max(d0, 0.5)  # 最小值保持为0.5
            else:
                d0 = 0.5 + 0.3 * (L / 30.0)  # 短序列的平滑过渡
            
            # TM-score = 1/L * sum(1 / (1 + (di/d0)^2))
            d0_squared = d0 ** 2
            tm_scores = torch.mean(1.0 / (1.0 + distances ** 2 / d0_squared), dim=-1)  # [B]
            
            return torch.clamp(tm_scores, min=0.0, max=1.0)
            
        except Exception as e:
            logger.warning(f"RNA TM-score计算失败: {e}")
            # 返回默认值
            return torch.zeros(batch_size, device=device)
    
    def _compute_rna_lddt(self, 
                     pred_coords: torch.Tensor, 
                     target_coords: torch.Tensor,
                     cutoff_distances: List[float] = None,
                     inclusion_radius: float = None) -> torch.Tensor:
        """计算RNA lDDT (local Distance Difference Test)
        
        RNA lDDT是基于距离差异的本地结构质量指标，范围0-100，值越高越好
        使用针对RNA调整的inclusion_radius和距离阈值
        
        Args:
            pred_coords: [batch_size, n_atoms, 3] 预测坐标
            target_coords: [batch_size, n_atoms, 3] 目标坐标
            cutoff_distances: 距离差异的阈值列表（默认RNA优化值）
            inclusion_radius: 考虑的原子对距离半径（默认RNA优化值）
            
        Returns:
            lddt_score: [batch_size] 每个样本的RNA lDDT分数
        """
        batch_size, n_atoms, _ = pred_coords.shape
        device = pred_coords.device
        
        # RNA专用的默认参数
        if cutoff_distances is None:
            cutoff_distances = [1.0, 2.0, 4.0, 8.0]  # RNA使用更大的阈值
        
        if inclusion_radius is None:
            inclusion_radius = 20.0  # RNA使用更大的inclusion radius
        
        # 确保坐标形状匹配
        if pred_coords.shape != target_coords.shape:
            min_len = min(pred_coords.shape[1], target_coords.shape[1])
            pred_coords = pred_coords[:, :min_len]
            target_coords = target_coords[:, :min_len]
            n_atoms = min_len
        
        try:
            # 1. 计算所有原子对之间的距离
            # pred_distances: [B, N, N]
            pred_diff = pred_coords.unsqueeze(2) - pred_coords.unsqueeze(1)  # [B, N, N, 3]
            pred_distances = torch.sqrt(torch.sum(pred_diff ** 2, dim=-1))  # [B, N, N]
            
            target_diff = target_coords.unsqueeze(2) - target_coords.unsqueeze(1)  # [B, N, N, 3]
            target_distances = torch.sqrt(torch.sum(target_diff ** 2, dim=-1))  # [B, N, N]
            
            # 2. 创建mask：只考虑inclusion_radius范围内的原子对
            inclusion_mask = target_distances <= inclusion_radius  # [B, N, N]
            
            # 排除对角线（自己与自己的距离）
            diag_mask = torch.eye(n_atoms, device=device).bool().unsqueeze(0).expand(batch_size, -1, -1)
            inclusion_mask = inclusion_mask & (~diag_mask)
            
            # 3. 计算距离差异
            distance_diff = torch.abs(pred_distances - target_distances)  # [B, N, N]
            
            # 4. 对每个阈值计算保存的接触数
            lddt_scores = []
            for cutoff in cutoff_distances:
                preserved = (distance_diff <= cutoff) & inclusion_mask  # [B, N, N]
                preserved_count = torch.sum(preserved.float(), dim=(1, 2))  # [B]
                total_count = torch.sum(inclusion_mask.float(), dim=(1, 2))  # [B]
                
                # 避免除零
                score = torch.where(total_count > 0, 
                                  preserved_count / total_count, 
                                  torch.zeros_like(preserved_count))
                lddt_scores.append(score)
            
            # 5. lDDT是所有阈值的平均值，乘以100
            lddt_final = torch.stack(lddt_scores, dim=0).mean(dim=0) * 100.0  # [B]
            
            return torch.clamp(lddt_final, min=0.0, max=100.0)
            
        except Exception as e:
            logger.warning(f"RNA lDDT计算失败: {e}")
            return torch.zeros(batch_size, device=device)
    
    def _compute_rna_clash_score(self, 
                           pred_coords: torch.Tensor,
                           clash_threshold: float = None,
                           vdw_radii: Dict[str, float] = None) -> torch.Tensor:
        """计算RNA clash score（原子冲突分数）
        
        检测RNA结构中过于接近的原子对，值越低越好
        使用针对RNA调整的冲突阈值
        
        Args:
            pred_coords: [batch_size, n_atoms, 3] 预测坐标
            clash_threshold: 冲突距离阈值（埃），默认RNA优化值
            vdw_radii: 原子的范德华半径字典（暂未使用）
            
        Returns:
            clash_score: [batch_size] 每个样本的冲突分数
        """
        batch_size, n_atoms, _ = pred_coords.shape
        device = pred_coords.device
        
        # RNA专用的冲突阈值
        if clash_threshold is None:
            clash_threshold = 2.5  # RNA使用更大的阈值
        
        try:
            # 计算所有原子对之间的距离
            coord_diff = pred_coords.unsqueeze(2) - pred_coords.unsqueeze(1)  # [B, N, N, 3]
            distances = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))  # [B, N, N]
            
            # 排除对角线（自己与自己）
            mask = torch.eye(n_atoms, device=device).bool().unsqueeze(0).expand(batch_size, -1, -1)
            distances = distances.masked_fill(mask, float('inf'))
            
            # 检测冲突
            clashes = distances < clash_threshold  # [B, N, N]
            
            # 计算每个样本的冲突数量（除以2因为每个冲突被计算了两次）
            clash_count = torch.sum(clashes.float(), dim=(1, 2)) / 2.0  # [B]
            
            # 归一化：除以可能的原子对数量
            total_pairs = n_atoms * (n_atoms - 1) / 2.0
            clash_score = clash_count / total_pairs * 100.0  # 转换为百分比
            
            return clash_score
            
        except Exception as e:
            logger.warning(f"RNA Clash score计算失败: {e}")
            return torch.zeros(batch_size, device=device)
    
    def compute_metrics(self) -> Dict[str, float]:
        """计算最终RNA评估指标"""
        if self.total_samples == 0:
            return {}
        
        metrics = {
            'avg_loss': self.total_loss / self.total_samples,
            'batch_count': self.batch_count,
            'total_samples': self.total_samples,
            'structure_type': 'RNA'  # 专门为RNA设计
        }
        
        # 添加损失分解
        for component, total_value in self.loss_components.items():
            metrics[f'avg_{component}'] = total_value / self.total_samples
        
        # 添加RNA结构指标
        if self.rmsd_values:
            import numpy as np
            metrics['avg_rmsd'] = np.mean(self.rmsd_values)
            metrics['median_rmsd'] = np.median(self.rmsd_values)
            metrics['std_rmsd'] = np.std(self.rmsd_values)
        
        # 添加RNA TM-score指标
        if self.tm_scores:
            import numpy as np
            metrics['avg_tm_score'] = np.mean(self.tm_scores)
            metrics['median_tm_score'] = np.median(self.tm_scores)
            metrics['std_tm_score'] = np.std(self.tm_scores)
            
            # RNA特有的阈值评估
            # RNA family similarity threshold (≥0.45 for same Rfam family)
            good_predictions = np.sum(np.array(self.tm_scores) >= 0.45)
            metrics['tm_score_good_ratio'] = good_predictions / len(self.tm_scores)
            
            # Excellent predictions (≥0.6)
            excellent_predictions = np.sum(np.array(self.tm_scores) >= 0.6)
            metrics['tm_score_excellent_ratio'] = excellent_predictions / len(self.tm_scores)
        
        # 添加RNA lDDT指标
        if self.lddt_scores:
            import numpy as np
            metrics['avg_lddt'] = np.mean(self.lddt_scores)
            metrics['median_lddt'] = np.median(self.lddt_scores)
            metrics['std_lddt'] = np.std(self.lddt_scores)
            
            # lDDT质量分级
            lddt_array = np.array(self.lddt_scores)
            metrics['lddt_high_quality_ratio'] = np.sum(lddt_array >= 70.0) / len(self.lddt_scores)  # 高质量
            metrics['lddt_good_quality_ratio'] = np.sum(lddt_array >= 50.0) / len(self.lddt_scores)   # 良好质量
        
        # 添加RNA clash score指标
        if self.clash_scores:
            import numpy as np
            metrics['avg_clash_score'] = np.mean(self.clash_scores)
            metrics['median_clash_score'] = np.median(self.clash_scores)
            metrics['std_clash_score'] = np.std(self.clash_scores)
            
            # Clash score质量评估 (值越低越好)
            clash_array = np.array(self.clash_scores)
            metrics['clash_low_ratio'] = np.sum(clash_array <= 5.0) / len(self.clash_scores)  # 低冲突比例
        
        # 添加置信度指标
        if self.confidence_scores:
            import numpy as np
            metrics['avg_confidence'] = np.mean(self.confidence_scores)
            metrics['median_confidence'] = np.median(self.confidence_scores)
        
        return metrics


# 为了向后兼容，保持EvaluationMetrics别名
EvaluationMetrics = RNAEvaluationMetrics 