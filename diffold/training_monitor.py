"""
训练监控和优化模块
提供性能监控、内存管理、健康检查等功能来节省训练时间
"""

import torch
import psutil
import time
import gc
import warnings
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import json
import threading
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        
        # 性能指标历史
        self.gpu_memory_usage = deque(maxlen=history_length)
        self.gpu_utilization = deque(maxlen=history_length)
        self.cpu_usage = deque(maxlen=history_length)
        self.ram_usage = deque(maxlen=history_length)
        self.batch_times = deque(maxlen=history_length)
        self.forward_times = deque(maxlen=history_length)
        self.backward_times = deque(maxlen=history_length)
        
        # 统计信息
        self.total_batches = 0
        self.oom_count = 0
        self.nan_loss_count = 0
        self.slow_batch_count = 0
        
        # 阈值设置
        self.gpu_memory_warning_threshold = 0.9  # 90%
        self.batch_time_warning_threshold = 300  # 5分钟
        
    def log_batch_performance(self, 
                             batch_time: float,
                             forward_time: float = None,
                             backward_time: float = None,
                             loss_value: float = None):
        """记录batch性能数据"""
        self.total_batches += 1
        self.batch_times.append(batch_time)
        
        if forward_time is not None:
            self.forward_times.append(forward_time)
        if backward_time is not None:
            self.backward_times.append(backward_time)
            
        # 检查异常
        if loss_value is not None and (torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value))):
            self.nan_loss_count += 1
            logger.warning(f"检测到NaN/Inf损失 (累计: {self.nan_loss_count})")
            
        if batch_time > self.batch_time_warning_threshold:
            self.slow_batch_count += 1
            logger.warning(f"检测到慢batch: {batch_time:.1f}s (累计: {self.slow_batch_count})")
    
    def log_system_metrics(self):
        """记录系统指标"""
        # CPU和内存使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        self.cpu_usage.append(cpu_percent)
        self.ram_usage.append(memory.percent)
        
        # GPU指标
        if torch.cuda.is_available():
            try:
                # GPU内存使用
                gpu_memory = torch.cuda.memory_stats()
                allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                reserved = gpu_memory.get('reserved_bytes.all.current', 0)
                max_allocated = torch.cuda.max_memory_allocated()
                
                memory_usage = allocated / max_allocated if max_allocated > 0 else 0
                self.gpu_memory_usage.append(memory_usage)
                
                # 检查GPU内存警告
                if memory_usage > self.gpu_memory_warning_threshold:
                    logger.warning(f"GPU内存使用率过高: {memory_usage:.1%}")
                    
            except Exception as e:
                logger.debug(f"获取GPU指标失败: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            'total_batches': self.total_batches,
            'oom_count': self.oom_count,
            'nan_loss_count': self.nan_loss_count,
            'slow_batch_count': self.slow_batch_count,
        }
        
        if self.batch_times:
            summary.update({
                'avg_batch_time': np.mean(self.batch_times),
                'median_batch_time': np.median(self.batch_times),
                'max_batch_time': np.max(self.batch_times),
                'min_batch_time': np.min(self.batch_times),
            })
        
        if self.gpu_memory_usage:
            summary.update({
                'avg_gpu_memory': np.mean(self.gpu_memory_usage),
                'max_gpu_memory': np.max(self.gpu_memory_usage),
            })
        
        if self.cpu_usage:
            summary.update({
                'avg_cpu_usage': np.mean(self.cpu_usage),
                'avg_ram_usage': np.mean(self.ram_usage),
            })
            
        return summary
    
    def plot_performance_metrics(self, save_path: str):
        """绘制性能指标图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Batch时间
        if self.batch_times:
            axes[0, 0].plot(list(self.batch_times))
            axes[0, 0].set_title('Batch Processing Time')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].grid(True)
        
        # GPU内存使用
        if self.gpu_memory_usage:
            axes[0, 1].plot(list(self.gpu_memory_usage))
            axes[0, 1].set_title('GPU Memory Usage')
            axes[0, 1].set_ylabel('Usage Ratio')
            axes[0, 1].grid(True)
        
        # CPU使用率
        if self.cpu_usage:
            axes[1, 0].plot(list(self.cpu_usage), label='CPU')
            axes[1, 0].plot(list(self.ram_usage), label='RAM')
            axes[1, 0].set_title('System Resource Usage')
            axes[1, 0].set_ylabel('Usage (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 前向/反向传播时间对比
        if self.forward_times and self.backward_times:
            axes[1, 1].plot(list(self.forward_times), label='Forward')
            axes[1, 1].plot(list(self.backward_times), label='Backward')
            axes[1, 1].set_title('Forward vs Backward Time')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class MemoryManager:
    """内存管理器"""
    
    def __init__(self):
        self.peak_memory = 0
        self.memory_cleanup_threshold = 0.85  # 85%时触发清理
        
    def check_memory_status(self) -> Dict[str, Any]:
        """检查内存状态"""
        status = {}
        
        # 系统内存
        memory = psutil.virtual_memory()
        status['system_memory'] = {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'needs_cleanup': memory.percent > self.memory_cleanup_threshold * 100
        }
        
        # GPU内存
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(device_id)
                reserved = torch.cuda.memory_reserved(device_id)
                total = torch.cuda.get_device_properties(device_id).total_memory
                
                usage_percent = (allocated / total) * 100
                
                status[f'gpu_{device_id}'] = {
                    'allocated': allocated,
                    'reserved': reserved,
                    'total': total,
                    'percent': usage_percent,
                    'needs_cleanup': usage_percent > self.memory_cleanup_threshold * 100
                }
                
                if allocated > self.peak_memory:
                    self.peak_memory = allocated
        
        return status
    
    def cleanup_memory(self, aggressive: bool = False):
        """清理内存"""
        logger.info("执行内存清理...")
        
        # Python垃圾回收
        collected = gc.collect()
        logger.info(f"Python GC回收了 {collected} 个对象")
        
        # PyTorch缓存清理
        if torch.cuda.is_available():
            try:
                # 清空CUDA缓存
                torch.cuda.empty_cache()
                logger.info("已清空CUDA缓存")
                
                if aggressive:
                    # 强制垃圾回收所有GPU内存
                    torch.cuda.ipc_collect()
                    logger.info("执行了强制GPU内存回收")
                    
            except Exception as e:
                logger.warning(f"GPU内存清理失败: {e}")
    
    def auto_cleanup_if_needed(self):
        """根据需要自动清理内存"""
        status = self.check_memory_status()
        
        needs_cleanup = False
        for device_name, device_status in status.items():
            if isinstance(device_status, dict) and device_status.get('needs_cleanup', False):
                needs_cleanup = True
                logger.warning(f"{device_name} 内存使用率过高: {device_status.get('percent', 0):.1f}%")
        
        if needs_cleanup:
            self.cleanup_memory()
            
    def get_memory_recommendations(self) -> List[str]:
        """获取内存优化建议"""
        recommendations = []
        status = self.check_memory_status()
        
        # 系统内存建议
        sys_mem = status.get('system_memory', {})
        if sys_mem.get('percent', 0) > 80:
            recommendations.append("系统内存使用率过高，考虑减少batch_size或num_workers")
        
        # GPU内存建议
        for device_name, device_status in status.items():
            if device_name.startswith('gpu_') and isinstance(device_status, dict):
                percent = device_status.get('percent', 0)
                if percent > 90:
                    recommendations.append(f"{device_name}: 内存使用率{percent:.1f}%，建议减少batch_size")
                elif percent > 80:
                    recommendations.append(f"{device_name}: 内存使用率{percent:.1f}%，建议启用梯度累积")
        
        return recommendations


class TrainingHealthChecker:
    """训练健康检查器"""
    
    def __init__(self):
        self.loss_history = deque(maxlen=50)  # 保存最近50个loss
        self.lr_history = deque(maxlen=50)
        self.gradient_norm_history = deque(maxlen=50)
        
        # 健康检查阈值
        self.loss_explosion_threshold = 10.0  # 损失爆炸阈值
        self.loss_stagnation_threshold = 0.001  # 损失停滞阈值
        self.gradient_explosion_threshold = 100.0  # 梯度爆炸阈值
        
    def check_loss_health(self, loss_value: float) -> Dict[str, Any]:
        """检查损失健康状况"""
        health_status = {'status': 'healthy', 'warnings': [], 'recommendations': []}
        
        self.loss_history.append(loss_value)
        
        # 检查损失爆炸
        if loss_value > self.loss_explosion_threshold:
            health_status['status'] = 'critical'
            health_status['warnings'].append(f"损失爆炸: {loss_value:.6f}")
            health_status['recommendations'].append("降低学习率或检查数据质量")
        
        # 检查损失停滞（需要至少20个历史值）
        if len(self.loss_history) >= 20:
            recent_losses = list(self.loss_history)[-20:]
            loss_variance = np.var(recent_losses)
            
            if loss_variance < self.loss_stagnation_threshold:
                health_status['status'] = 'warning' if health_status['status'] == 'healthy' else health_status['status']
                health_status['warnings'].append(f"损失停滞，方差: {loss_variance:.6f}")
                health_status['recommendations'].append("考虑调整学习率或检查模型容量")
        
        # 检查NaN/Inf
        if torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value)):
            health_status['status'] = 'critical'
            health_status['warnings'].append("检测到NaN/Inf损失")
            health_status['recommendations'].append("检查输入数据、降低学习率或使用梯度裁剪")
        
        return health_status
    
    def check_gradient_health(self, model) -> Dict[str, Any]:
        """检查梯度健康状况"""
        health_status = {'status': 'healthy', 'warnings': [], 'recommendations': []}
        
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.gradient_norm_history.append(total_norm)
            
            # 检查梯度爆炸
            if total_norm > self.gradient_explosion_threshold:
                health_status['status'] = 'critical'
                health_status['warnings'].append(f"梯度爆炸: {total_norm:.6f}")
                health_status['recommendations'].append("降低学习率或调整梯度裁剪阈值")
            
            # 检查梯度消失
            if total_norm < 1e-6:
                health_status['status'] = 'warning'
                health_status['warnings'].append(f"梯度过小: {total_norm:.6f}")
                health_status['recommendations'].append("检查学习率是否过小或模型是否过深")
        
        return health_status
    
    def check_learning_rate_health(self, current_lr: float) -> Dict[str, Any]:
        """检查学习率健康状况"""
        health_status = {'status': 'healthy', 'warnings': [], 'recommendations': []}
        
        self.lr_history.append(current_lr)
        
        # 检查学习率是否过大
        if current_lr > 1e-2:
            health_status['warnings'].append(f"学习率可能过大: {current_lr:.6f}")
            health_status['recommendations'].append("考虑降低学习率")
        
        # 检查学习率是否过小
        if current_lr < 1e-6:
            health_status['warnings'].append(f"学习率可能过小: {current_lr:.6f}")
            health_status['recommendations'].append("考虑提高学习率或检查调度器设置")
        
        return health_status
    
    def generate_health_report(self, model, loss_value: float, current_lr: float) -> Dict[str, Any]:
        """生成综合健康报告"""
        report = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'checks': {},
            'all_warnings': [],
            'all_recommendations': []
        }
        
        # 执行各项检查
        loss_health = self.check_loss_health(loss_value)
        gradient_health = self.check_gradient_health(model)
        lr_health = self.check_learning_rate_health(current_lr)
        
        report['checks'] = {
            'loss': loss_health,
            'gradient': gradient_health,
            'learning_rate': lr_health
        }
        
        # 汇总状态
        all_statuses = [loss_health['status'], gradient_health['status'], lr_health['status']]
        if 'critical' in all_statuses:
            report['overall_status'] = 'critical'
        elif 'warning' in all_statuses:
            report['overall_status'] = 'warning'
        
        # 汇总警告和建议
        for check in report['checks'].values():
            report['all_warnings'].extend(check['warnings'])
            report['all_recommendations'].extend(check['recommendations'])
        
        return report


class AdaptiveBatchSize:
    """自适应批次大小管理器"""
    
    def __init__(self, initial_batch_size: int, min_batch_size: int = 1, max_batch_size: int = 64):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        self.oom_count = 0
        self.success_count = 0
        self.last_adjustment_step = 0
        
        # 调整策略参数
        self.increase_threshold = 10  # 连续成功10次后尝试增加
        self.adjustment_cooldown = 50  # 调整后的冷却期
        
    def should_adjust_batch_size(self, current_step: int) -> Tuple[bool, int]:
        """判断是否应该调整批次大小"""
        if current_step - self.last_adjustment_step < self.adjustment_cooldown:
            return False, self.current_batch_size
        
        # OOM后减少批次大小
        if self.oom_count > 0:
            new_size = max(self.min_batch_size, self.current_batch_size // 2)
            if new_size != self.current_batch_size:
                logger.info(f"由于OOM，批次大小从{self.current_batch_size}减少到{new_size}")
                self.current_batch_size = new_size
                self.last_adjustment_step = current_step
                self.oom_count = 0
                self.success_count = 0
                return True, new_size
        
        # 连续成功后尝试增加批次大小
        elif self.success_count >= self.increase_threshold and self.current_batch_size < self.max_batch_size:
            new_size = min(self.max_batch_size, int(self.current_batch_size * 1.5))
            if new_size != self.current_batch_size:
                logger.info(f"尝试增加批次大小从{self.current_batch_size}到{new_size}")
                self.current_batch_size = new_size
                self.last_adjustment_step = current_step
                self.success_count = 0
                return True, new_size
        
        return False, self.current_batch_size
    
    def report_oom(self):
        """报告OOM事件"""
        self.oom_count += 1
        self.success_count = 0
        logger.warning(f"检测到OOM事件 (累计: {self.oom_count})")
    
    def report_success(self):
        """报告成功执行"""
        self.success_count += 1
        if self.oom_count > 0:
            self.oom_count = max(0, self.oom_count - 1)


class TrainingMonitor:
    """训练监控主类"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化各个监控组件
        self.performance_monitor = PerformanceMonitor()
        self.memory_manager = MemoryManager()
        self.health_checker = TrainingHealthChecker()
        
        # 监控数据保存
        self.monitoring_data = []
        
    def log_training_step(self, 
                         step: int,
                         epoch: int,
                         loss_value: float,
                         learning_rate: float,
                         batch_time: float,
                         model,
                         forward_time: float = None,
                         backward_time: float = None):
        """记录训练步骤的所有指标"""
        
        # 记录性能指标
        self.performance_monitor.log_batch_performance(
            batch_time=batch_time,
            forward_time=forward_time,
            backward_time=backward_time,
            loss_value=loss_value
        )
        
        # 记录系统指标
        self.performance_monitor.log_system_metrics()
        
        # 健康检查
        health_report = self.health_checker.generate_health_report(
            model=model,
            loss_value=loss_value,
            current_lr=learning_rate
        )
        
        # 内存管理
        memory_status = self.memory_manager.check_memory_status()
        self.memory_manager.auto_cleanup_if_needed()
        
        # 汇总数据
        monitoring_entry = {
            'step': step,
            'epoch': epoch,
            'loss': loss_value,
            'learning_rate': learning_rate,
            'batch_time': batch_time,
            'health_status': health_report['overall_status'],
            'memory_status': memory_status,
            'timestamp': time.time()
        }
        
        self.monitoring_data.append(monitoring_entry)
        
        # 输出重要警告
        if health_report['overall_status'] != 'healthy':
            logger.warning(f"训练健康状态: {health_report['overall_status']}")
            for warning in health_report['all_warnings']:
                logger.warning(f"  - {warning}")
    
    def save_monitoring_report(self, filename: str = "training_monitor_report.json"):
        """保存监控报告"""
        report_path = self.output_dir / filename
        
        # 生成综合报告
        report = {
            'performance_summary': self.performance_monitor.get_performance_summary(),
            'memory_recommendations': self.memory_manager.get_memory_recommendations(),
            'monitoring_data': self.monitoring_data[-100:],  # 最近100条记录
            'timestamp': time.time()
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"监控报告已保存到: {report_path}")
    
    def generate_performance_plots(self):
        """生成性能图表"""
        plot_path = self.output_dir / "performance_metrics.png"
        self.performance_monitor.plot_performance_metrics(str(plot_path))
        logger.info(f"性能图表已保存到: {plot_path}") 