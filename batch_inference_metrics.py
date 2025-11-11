#!/usr/bin/env python3
"""
批量推理和指标计算脚本
对验证集样本进行批量推理，计算结构预测指标并输出到文件
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# 导入Diffold相关模块
from diffold.diffold import Diffold
from diffold.dataloader import create_data_loaders
from diffold.metrics import RNAEvaluationMetrics
from rhofold.utils import get_device, timing

# 导入PDB转换功能
from diffold.output import diffold_coords_to_pdb, validate_diffold_output

# 采样结果筛选函数类型定义
SampleSelectionFunc = Callable[[List[Dict[str, Any]]], int]

# ========== 采样结果筛选策略函数 ==========

def select_best_by_rmsd(samples: List[Dict[str, Any]]) -> int:
    """
    根据RMSD选择最佳采样（默认策略）
    返回RMSD最小的采样索引
    """
    if not samples:
        raise ValueError("样本列表为空")
    
    best_idx = 0
    best_rmsd = samples[0]['rmsd']
    
    for i, sample in enumerate(samples):
        if sample['rmsd'] < best_rmsd:
            best_rmsd = sample['rmsd']
            best_idx = i
    
    return best_idx

def select_best_by_tm_score(samples: List[Dict[str, Any]]) -> int:
    """
    根据TM分数选择最佳采样
    返回TM分数最高的采样索引
    """
    if not samples:
        raise ValueError("样本列表为空")
    
    best_idx = 0
    best_tm = samples[0]['metrics'].get('avg_tm_score', 0.0)
    
    for i, sample in enumerate(samples):
        tm_score = sample['metrics'].get('avg_tm_score', 0.0)
        if tm_score > best_tm:
            best_tm = tm_score
            best_idx = i
    
    return best_idx

def select_best_by_lddt(samples: List[Dict[str, Any]]) -> int:
    """
    根据lDDT分数选择最佳采样
    返回lDDT分数最高的采样索引
    """
    if not samples:
        raise ValueError("样本列表为空")
    
    best_idx = 0
    best_lddt = samples[0]['metrics'].get('avg_lddt', 0.0)
    
    for i, sample in enumerate(samples):
        lddt_score = sample['metrics'].get('avg_lddt', 0.0)
        if lddt_score > best_lddt:
            best_lddt = lddt_score
            best_idx = i
    
    return best_idx

def select_best_by_clash_score(samples: List[Dict[str, Any]]) -> int:
    """
    根据冲突分数选择最佳采样
    返回冲突分数最低的采样索引
    """
    if not samples:
        raise ValueError("样本列表为空")
    
    best_idx = 0
    best_clash = samples[0]['metrics'].get('avg_clash_score', float('inf'))
    
    for i, sample in enumerate(samples):
        clash_score = sample['metrics'].get('avg_clash_score', float('inf'))
        if clash_score < best_clash:
            best_clash = clash_score
            best_idx = i
    
    return best_idx

def select_best_by_composite_score(samples: List[Dict[str, Any]]) -> int:
    """
    根据综合分数选择最佳采样
    综合分数 = (normalized_tm_score + normalized_lddt - normalized_rmsd - normalized_clash) / 4
    """
    if not samples:
        raise ValueError("样本列表为空")
    
    # 提取所有指标
    rmsd_values = [s['rmsd'] for s in samples]
    tm_values = [s['metrics'].get('avg_tm_score', 0.0) for s in samples]
    lddt_values = [s['metrics'].get('avg_lddt', 0.0) for s in samples]
    clash_values = [s['metrics'].get('avg_clash_score', 0.0) for s in samples]
    
    # 归一化（避免除零）
    def safe_normalize(values, reverse=False):
        if len(set(values)) <= 1:  # 所有值相同
            return [0.0] * len(values)
        min_val, max_val = min(values), max(values)
        if reverse:
            return [(max_val - v) / (max_val - min_val) for v in values]
        else:
            return [(v - min_val) / (max_val - min_val) for v in values]
    
    norm_rmsd = safe_normalize(rmsd_values, reverse=True)  # RMSD越小越好
    norm_tm = safe_normalize(tm_values, reverse=False)     # TM分数越大越好
    norm_lddt = safe_normalize(lddt_values, reverse=False) # lDDT越大越好
    norm_clash = safe_normalize(clash_values, reverse=True) # 冲突分数越小越好
    
    # 计算综合分数
    best_idx = 0
    best_score = norm_rmsd[0] + norm_tm[0] + norm_lddt[0] + norm_clash[0]
    
    for i in range(1, len(samples)):
        composite_score = norm_rmsd[i] + norm_tm[i] + norm_lddt[i] + norm_clash[i]
        if composite_score > best_score:
            best_score = composite_score
            best_idx = i
    
    return best_idx

# 预定义策略映射
SELECTION_STRATEGIES = {
    'rmsd': select_best_by_rmsd,
    'tm_score': select_best_by_tm_score,
    'lddt': select_best_by_lddt,
    'clash_score': select_best_by_clash_score,
    'composite': select_best_by_composite_score,
}

def setup_logging(output_dir: str, log_level: str = "INFO"):
    """设置日志"""
    log_file = Path(output_dir) / "batch_inference.log"
    
    # 创建日志目录
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(getattr(logging, log_level.upper()))
    stream_handler.setFormatter(formatter)
    
    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def load_model(config: argparse.Namespace, logger: logging.Logger):
    """加载Diffold模型"""
    logger.info("构建Diffold模型")
    model = Diffold(config, rhofold_checkpoint_path=config.rhofold_checkpoint)
    model = model.to(config.device)
    model.eval()
    
    # 加载检查点
    if config.checkpoint_path:
        logger.info(f"加载检查点: {config.checkpoint_path}")
        try:
            # 首先尝试使用weights_only=True加载
            checkpoint = torch.load(config.checkpoint_path, map_location=config.device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("检查点加载完成")
        except Exception as e:
            logger.warning(f"weights_only=True 加载失败，尝试使用 weights_only=False: {e}")
            try:
                # 回退到weights_only=False
                checkpoint = torch.load(config.checkpoint_path, map_location=config.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info("检查点加载完成")
            except Exception as e2:
                logger.error(f"检查点加载失败: {e2}")
                raise
    
    # 🎯 加载 LoRA 适配器（如果指定）
    if hasattr(config, 'lora_path') and config.lora_path:
        logger.info("=" * 60)
        logger.info(f"🎯 加载 LoRA 适配器: {config.lora_path}")
        logger.info("=" * 60)
        try:
            from diffold.lora_utils import LoRAManager
            model = LoRAManager.load_lora_weights(model, config.lora_path)
            logger.info("✅ LoRA 适配器加载成功!")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"❌ LoRA 适配器加载失败: {e}")
            raise
    
    return model

def load_validation_data(config: argparse.Namespace, logger: logging.Logger):
    """加载验证数据"""
    logger.info("加载验证数据")
    
    # 🎯 支持两种模式：指定样本列表文件 或 使用 fold
    if hasattr(config, 'sample_list_file') and config.sample_list_file:
        # 模式1: 从指定的样本列表文件读取
        sample_list_file = Path(config.sample_list_file)
        if not sample_list_file.exists():
            raise FileNotFoundError(f"样本列表文件不存在: {sample_list_file}")
        
        logger.info(f"从指定文件加载样本列表: {sample_list_file}")
        with open(sample_list_file, 'r') as f:
            sample_names = [line.strip() for line in f if line.strip()]
        
        logger.info(f"样本数量: {len(sample_names)}")
        
        # 创建临时 fold 文件（复用现有的 dataloader）
        temp_list_dir = Path(config.data_dir) / "list"
        temp_list_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建验证集文件
        temp_valid_file = temp_list_dir / "valid_fold-999"
        with open(temp_valid_file, 'w') as f:
            for name in sample_names:
                f.write(f"{name}\n")
        
        # 创建训练集文件（放入相同样本，避免空列表报错，反正只用验证集）
        temp_train_file = temp_list_dir / "fold-999_train_ids"
        with open(temp_train_file, 'w') as f:
            for name in sample_names:
                f.write(f"{name}\n")
        
        logger.info(f"创建临时 fold 文件: {temp_valid_file}")
        
        # 使用临时 fold 加载数据
        train_loader, valid_loader = create_data_loaders(
            data_dir=config.data_dir,
            batch_size=1,
            max_length=config.max_sequence_length,
            num_workers=config.num_workers,
            fold=999,  # 使用临时 fold 编号
            use_msa=config.use_msa,
            use_all_folds=False,
            world_size=1,
            local_rank=0
        )
        
    else:
        # 模式2: 使用原来的 fold 方式
        valid_list_file = Path(config.data_dir) / "list" / f"valid_fold-{config.fold}"
        if not valid_list_file.exists():
            raise FileNotFoundError(f"验证集列表文件不存在: {valid_list_file}")
        
        logger.info(f"使用交叉验证 fold-{config.fold}")
        with open(valid_list_file, 'r') as f:
            sample_names = [line.strip() for line in f if line.strip()]
        
        logger.info(f"验证集样本数量: {len(sample_names)}")
        
        # 创建数据加载器
        train_loader, valid_loader = create_data_loaders(
            data_dir=config.data_dir,
            batch_size=1,  # 批量推理使用batch_size=1
            max_length=config.max_sequence_length,
            num_workers=config.num_workers,
            fold=config.fold,
            use_msa=config.use_msa,
            use_all_folds=False,
            world_size=1,
            local_rank=0
        )
    
    return valid_loader, sample_names

def compute_metrics_for_sample(model: Diffold, 
                             batch: Dict[str, torch.Tensor], 
                             sample_name: str,
                             metrics_calculator: RNAEvaluationMetrics,
                             output_dir: str,
                             num_sampling: int,
                             save_all_samples: bool,
                             selection_func: SampleSelectionFunc,
                             logger: logging.Logger) -> Dict[str, Any]:
    """计算单个样本的指标并保存PDB文件（支持多次采样）"""
    try:
        # 准备输入数据
        device = next(model.parameters()).device  # 获取模型所在的设备
        tokens = batch['tokens'].to(device)
        sequences = batch['sequences']
        coordinates = batch.get('coordinates', None)
        missing_atom_masks = batch.get('missing_atom_masks', None)
        rna_fm_tokens = batch.get('rna_fm_tokens', None)
        
        if coordinates is not None:
            coordinates = coordinates.to(device)
        if missing_atom_masks is not None:
            missing_atom_masks = missing_atom_masks.to(device)
        if rna_fm_tokens is not None:
            rna_fm_tokens = rna_fm_tokens.to(device)
        
        # 获取目标坐标
        target_coords = coordinates
        if target_coords is None:
            logger.warning(f"样本 {sample_name}: 未获取到目标坐标")
            return {
                'sample_name': sample_name,
                'status': 'failed',
                'error': 'no_target_coords'
            }
        
        # 多次采样
        all_samples = []
        
        logger.debug(f"样本 {sample_name}: 开始 {num_sampling} 次采样")
        
        for sample_idx in range(num_sampling):
            # 模型推理（每次采样可能产生不同结果）
            with torch.no_grad():
                result = model(
                    tokens=tokens,
                    rna_fm_tokens=rna_fm_tokens,
                    seq=sequences,
                    target_coords=coordinates,
                    missing_atom_mask=missing_atom_masks
                )
            
            if result is None:
                logger.warning(f"样本 {sample_name} 采样 {sample_idx+1}: 模型推理返回None")
                continue
            
            # 提取预测坐标
            predicted_coords = result.get('predicted_coords')
            atom_mask = result.get('atom_mask', None)
            
            if predicted_coords is None:
                logger.warning(f"样本 {sample_name} 采样 {sample_idx+1}: 未获取到预测坐标")
                continue
            
            # 计算当前采样的指标
            temp_metrics_calculator = RNAEvaluationMetrics()
            temp_metrics_calculator.update(
                loss=0.0,
                batch_size=1,
                predicted_coords=predicted_coords,
                target_coords=target_coords
            )
            
            sample_metrics = temp_metrics_calculator.compute_metrics()
            current_rmsd = sample_metrics.get('avg_rmsd', float('inf'))
            
            # 保存采样结果
            sample_result = {
                'sample_idx': sample_idx,
                'predicted_coords': predicted_coords,
                'atom_mask': atom_mask,
                'metrics': sample_metrics,
                'rmsd': current_rmsd
            }
            
            # 获取详细指标
            detailed_metrics = {}
            if hasattr(temp_metrics_calculator, 'rmsd_values') and temp_metrics_calculator.rmsd_values:
                detailed_metrics['rmsd_values'] = temp_metrics_calculator.rmsd_values
            if hasattr(temp_metrics_calculator, 'tm_scores') and temp_metrics_calculator.tm_scores:
                detailed_metrics['tm_scores'] = temp_metrics_calculator.tm_scores
            if hasattr(temp_metrics_calculator, 'lddt_scores') and temp_metrics_calculator.lddt_scores:
                detailed_metrics['lddt_scores'] = temp_metrics_calculator.lddt_scores
            if hasattr(temp_metrics_calculator, 'clash_scores') and temp_metrics_calculator.clash_scores:
                detailed_metrics['clash_scores'] = temp_metrics_calculator.clash_scores
            
            sample_result['detailed_metrics'] = detailed_metrics
            all_samples.append(sample_result)
        
        if not all_samples:
            logger.warning(f"样本 {sample_name}: 所有采样都失败")
            return {
                'sample_name': sample_name,
                'status': 'failed',
                'error': 'all_sampling_failed'
            }
        
        # 使用筛选函数选择最佳采样
        try:
            best_sample_idx = selection_func(all_samples)
            best_sample = all_samples[best_sample_idx]
            best_rmsd = best_sample['rmsd']
            
            logger.debug(f"样本 {sample_name}: 筛选函数选择采样 {best_sample_idx+1}, RMSD: {best_rmsd:.4f}")
            
        except Exception as e:
            logger.warning(f"样本 {sample_name}: 筛选函数执行失败: {e}, 使用默认RMSD策略")
            # 回退到默认的RMSD策略
            best_sample_idx = select_best_by_rmsd(all_samples)
            best_sample = all_samples[best_sample_idx]
            best_rmsd = best_sample['rmsd']
        
        # 保存PDB文件
        pdb_file_paths = []
        
        # 创建PDB输出目录
        pdb_output_dir = Path(output_dir) / "pdb_files"
        pdb_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取序列 - 优先从batch中获取，确保与当前样本匹配
        # batch['names'] 包含实际的样本名称列表
        batch_sample_name = batch.get('names', [sample_name])[0] if batch.get('names') else sample_name
        
        # 验证样本名称是否匹配
        if batch_sample_name != sample_name:
            logger.warning(f"样本名称不匹配: batch中为 {batch_sample_name}, 期望为 {sample_name}")
            # 使用batch中的实际样本名称
            sample_name = batch_sample_name
        
        sequence = sequences[0] if sequences else ""
        
        # 如果序列为空，尝试从序列文件中读取
        if not sequence:
            logger.warning(f"样本 {sample_name}: batch中序列为空，尝试从文件读取")
            # 这里可以添加从文件读取序列的逻辑，但通常batch中应该有序列
        
        # 保存最佳采样的PDB文件
        best_pdb_path = pdb_output_dir / f"{sample_name}_best.pdb"
        try:
            diffold_coords_to_pdb(
                predicted_coords=best_sample['predicted_coords'],
                sequence=sequence,
                output_path=str(best_pdb_path),
                atom_mask=best_sample['atom_mask'],
                logger_instance=logger
            )
            pdb_file_paths.append(str(best_pdb_path))
            logger.debug(f"样本 {sample_name}: 最佳PDB文件已保存到 {best_pdb_path}")
        except Exception as e:
            logger.warning(f"样本 {sample_name}: 最佳PDB文件保存失败: {e}")
        
        # 如果需要保存所有采样结果
        if save_all_samples:
            for i, sample_result in enumerate(all_samples):
                sample_pdb_path = pdb_output_dir / f"{sample_name}_sample_{i+1}.pdb"
                try:
                    diffold_coords_to_pdb(
                        predicted_coords=sample_result['predicted_coords'],
                        sequence=sequence,
                        output_path=str(sample_pdb_path),
                        atom_mask=sample_result['atom_mask'],
                        logger_instance=logger
                    )
                    pdb_file_paths.append(str(sample_pdb_path))
                except Exception as e:
                    logger.warning(f"样本 {sample_name} 采样 {i+1}: PDB文件保存失败: {e}")
        
        # 准备返回结果
        result_dict = {
            'sample_name': sample_name,
            'status': 'success',
            'sequence_length': len(sequence) if sequence else 0,
            'sequence': sequence,
            'num_sampling': num_sampling,
            'successful_samples': len(all_samples),
            'best_sample_idx': best_sample['sample_idx'],
            'best_rmsd': best_rmsd,
            'predicted_coords_shape': list(best_sample['predicted_coords'].shape),
            'target_coords_shape': list(target_coords.shape),
            'pdb_file_paths': pdb_file_paths,
            'best_pdb_path': str(best_pdb_path) if pdb_file_paths else None,
            # 使用最佳采样的指标作为主要指标
            **best_sample['metrics'],
            'detailed_metrics': best_sample['detailed_metrics']
        }
        
        # 如果保存所有采样，添加所有采样的信息
        if save_all_samples:
            all_samples_info = []
            for sample_result in all_samples:
                sample_info = {
                    'sample_idx': sample_result['sample_idx'],
                    'rmsd': sample_result['rmsd'],
                    'metrics': sample_result['metrics'],
                    'detailed_metrics': sample_result['detailed_metrics']
                }
                all_samples_info.append(sample_info)
            result_dict['all_samples'] = all_samples_info
            
            # 添加采样统计信息
            rmsd_values = [s['rmsd'] for s in all_samples]
            result_dict['sampling_stats'] = {
                'rmsd_mean': np.mean(rmsd_values),
                'rmsd_std': np.std(rmsd_values),
                'rmsd_min': np.min(rmsd_values),
                'rmsd_max': np.max(rmsd_values)
            }
        
        logger.debug(f"样本 {sample_name}: 完成 {len(all_samples)}/{num_sampling} 次成功采样，最佳RMSD: {best_rmsd:.4f}")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"样本 {sample_name} 处理失败: {e}")
        return {
            'sample_name': sample_name,
            'status': 'failed',
            'error': str(e)
        }

def save_results(results: List[Dict[str, Any]], output_dir: str, logger: logging.Logger):
    """保存结果到文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON格式
    json_file = output_path / "batch_inference_results.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"结果已保存到: {json_file}")
    
    # 保存为CSV格式
    csv_file = output_path / "batch_inference_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    logger.info(f"结果已保存到: {csv_file}")
    
    # 保存详细指标数据
    detailed_metrics_file = output_path / "detailed_metrics.json"
    detailed_metrics_data = {}
    for result in results:
        if result['status'] == 'success' and 'detailed_metrics' in result:
            detailed_metrics_data[result['sample_name']] = result['detailed_metrics']
    
    with open(detailed_metrics_file, 'w') as f:
        json.dump(detailed_metrics_data, f, indent=2, default=str)
    logger.info(f"详细指标数据已保存到: {detailed_metrics_file}")
    
    # 生成统计报告
    report_file = output_path / "batch_inference_report.txt"
    generate_report(results, report_file, logger)
    
    return json_file, csv_file, detailed_metrics_file, report_file

def generate_report(results: List[Dict[str, Any]], report_file: Path, logger: logging.Logger):
    """生成统计报告"""
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']
    
    with open(report_file, 'w') as f:
        f.write("批量推理指标计算报告（多次采样版本）\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总样本数: {len(results)}\n")
        f.write(f"成功样本数: {len(successful_results)}\n")
        f.write(f"失败样本数: {len(failed_results)}\n")
        f.write(f"成功率: {len(successful_results)/len(results)*100:.2f}%\n\n")
        
        if successful_results:
            # 采样统计信息
            f.write("采样统计:\n")
            f.write("-" * 30 + "\n")
            
            # 获取采样信息
            num_sampling_values = [r.get('num_sampling', 1) for r in successful_results]
            successful_samples_values = [r.get('successful_samples', 1) for r in successful_results]
            
            if num_sampling_values:
                f.write(f"每样本采样次数: {num_sampling_values[0]}\n")
                f.write(f"平均成功采样数: {np.mean(successful_samples_values):.2f}\n")
                f.write(f"采样成功率: {np.mean(successful_samples_values)/num_sampling_values[0]*100:.2f}%\n\n")
            
            # 最佳RMSD统计
            best_rmsd_values = [r.get('best_rmsd', r.get('avg_rmsd', float('inf'))) for r in successful_results]
            if best_rmsd_values:
                f.write("最佳RMSD统计:\n")
                f.write(f"  平均值: {np.mean(best_rmsd_values):.4f}\n")
                f.write(f"  中位数: {np.median(best_rmsd_values):.4f}\n")
                f.write(f"  标准差: {np.std(best_rmsd_values):.4f}\n")
                f.write(f"  最小值: {np.min(best_rmsd_values):.4f}\n")
                f.write(f"  最大值: {np.max(best_rmsd_values):.4f}\n\n")
            
            # 检查是否有采样统计信息
            samples_with_stats = [r for r in successful_results if 'sampling_stats' in r]
            if samples_with_stats:
                f.write("采样RMSD变异性统计:\n")
                f.write("-" * 30 + "\n")
                rmsd_stds = [r['sampling_stats']['rmsd_std'] for r in samples_with_stats]
                rmsd_ranges = [r['sampling_stats']['rmsd_max'] - r['sampling_stats']['rmsd_min'] for r in samples_with_stats]
                
                f.write(f"RMSD标准差的平均值: {np.mean(rmsd_stds):.4f}\n")
                f.write(f"RMSD标准差的中位数: {np.median(rmsd_stds):.4f}\n")
                f.write(f"RMSD范围的平均值: {np.mean(rmsd_ranges):.4f}\n")
                f.write(f"RMSD范围的中位数: {np.median(rmsd_ranges):.4f}\n\n")
            
            # 计算其他指标统计
            metrics_keys = ['avg_tm_score', 'avg_lddt', 'avg_clash_score']
            available_metrics = [key for key in metrics_keys if any(key in r for r in successful_results)]
            
            f.write("其他指标统计（基于最佳采样）:\n")
            f.write("-" * 30 + "\n")
            
            for metric in available_metrics:
                values = [r[metric] for r in successful_results if metric in r]
                if values:
                    f.write(f"{metric}:\n")
                    f.write(f"  平均值: {np.mean(values):.4f}\n")
                    f.write(f"  中位数: {np.median(values):.4f}\n")
                    f.write(f"  标准差: {np.std(values):.4f}\n")
                    f.write(f"  最小值: {np.min(values):.4f}\n")
                    f.write(f"  最大值: {np.max(values):.4f}\n\n")
        
        if failed_results:
            f.write("失败样本:\n")
            f.write("-" * 30 + "\n")
            for result in failed_results:
                f.write(f"{result['sample_name']}: {result.get('error', 'unknown_error')}\n")
    
    logger.info(f"报告已保存到: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="批量推理和指标计算")
    
    # 基本参数
    parser.add_argument("--data_dir", default="./processed_data", 
                       help="数据目录路径")
    parser.add_argument("--output_dir", default="./batch_inference_output", 
                       help="输出目录路径")
    
    # 🎯 数据选择：支持两种模式
    data_mode = parser.add_mutually_exclusive_group()
    data_mode.add_argument("--fold", type=int, default=None, 
                          help="使用交叉验证折数（例如: 0, 1, 2, 3, 4）")
    data_mode.add_argument("--sample_list_file", type=str, default=None,
                          help="指定样本列表文件路径（每行一个样本名称）")
    
    # 模型参数
    parser.add_argument("--checkpoint_path", required=True,
                       help="模型检查点路径")
    parser.add_argument("--rhofold_checkpoint", default="./pretrained/model_20221010_params.pt",
                       help="RhoFold检查点路径")
    parser.add_argument("--lora_path", default=None,
                       help="LoRA适配器路径（可选），如: ./checkpoints_finetune/best_lora")
    
    # 数据参数
    parser.add_argument("--max_sequence_length", type=int, default=256,
                       help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="数据加载器工作进程数")
    parser.add_argument("--use_msa", action="store_true", default=True,
                       help="是否使用MSA")
    
    # 设备参数
    parser.add_argument("--device", default="auto",
                       help="计算设备 (auto, cpu, cuda)")
    parser.add_argument("--log_level", default="INFO",
                       help="日志级别")
    
    # 可选参数
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大处理样本数（用于测试）")
    parser.add_argument("--num_sampling", type=int, default=1,
                       help="每个样本的采样次数（扩散模型）")
    parser.add_argument("--save_all_samples", action="store_true", default=False,
                       help="是否保存所有采样结果")
    parser.add_argument("--selection_strategy", choices=list(SELECTION_STRATEGIES.keys()), 
                       default="rmsd",
                       help="采样结果筛选策略: rmsd(默认), tm_score, lddt, clash_score, composite")
    
    args = parser.parse_args()
    
    # 验证参数：必须指定 fold 或 sample_list_file 之一
    if args.fold is None and args.sample_list_file is None:
        parser.error("必须指定 --fold 或 --sample_list_file 之一")
    
    # 设置设备
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    else:
        args.device = get_device(args.device)
    
    # 设置日志
    logger = setup_logging(args.output_dir, args.log_level)
    logger.info("=" * 60)
    logger.info("开始批量推理和指标计算（多次采样版本）")
    logger.info("=" * 60)
    logger.info(f"设备: {args.device}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 显示数据选择模式
    if args.sample_list_file:
        logger.info(f"📄 数据模式: 指定样本列表")
        logger.info(f"   样本列表文件: {args.sample_list_file}")
    else:
        logger.info(f"📊 数据模式: 交叉验证")
        logger.info(f"   Fold: {args.fold}")
    
    logger.info(f"模型检查点: {args.checkpoint_path}")
    if args.lora_path:
        logger.info(f"🎯 LoRA适配器: {args.lora_path}")
    logger.info(f"每样本采样次数: {args.num_sampling}")
    logger.info(f"保存所有采样结果: {args.save_all_samples}")
    logger.info(f"筛选策略: {args.selection_strategy}")
    logger.info("=" * 60)
    
    # 获取筛选函数
    selection_func = SELECTION_STRATEGIES[args.selection_strategy]
    logger.info(f"使用筛选函数: {selection_func.__name__}")
    
    try:
        # 加载模型
        model = load_model(args, logger)
        
        # 加载验证数据
        valid_loader, sample_names = load_validation_data(args, logger)
        
        # 创建指标计算器
        metrics_calculator = RNAEvaluationMetrics()
        
        # 批量处理
        results = []
        start_time = time.time()
        
        # 限制样本数量（用于测试）
        if args.max_samples:
            sample_names = sample_names[:args.max_samples]
            logger.info(f"限制处理样本数为: {len(sample_names)}")
        
        logger.info("开始批量推理...")
        # 使用batch中的实际样本名称，而不是依赖外部sample_names的顺序
        # 这样可以避免顺序不匹配的问题
        for i, batch in enumerate(tqdm(valid_loader, 
                                     total=len(valid_loader),
                                     desc="处理样本")):
            
            # 从batch中获取实际的样本名称
            batch_names = batch.get('names', [])
            if not batch_names:
                logger.warning(f"Batch {i}: 未找到样本名称，跳过")
                continue
            
            # batch_size=1时，取第一个样本名称
            sample_name = batch_names[0] if batch_names else f"unknown_{i}"
            
            logger.debug(f"处理样本 {i+1}/{len(valid_loader)}: {sample_name}")
            
            # 计算指标
            result = compute_metrics_for_sample(
                model=model,
                batch=batch,
                sample_name=sample_name,
                metrics_calculator=metrics_calculator,
                output_dir=args.output_dir,
                num_sampling=args.num_sampling,
                save_all_samples=args.save_all_samples,
                selection_func=selection_func,
                logger=logger
            )
            
            results.append(result)
            
            # 每100个样本输出一次进度
            if (i + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_sample = elapsed_time / (i + 1)
                remaining_samples = len(sample_names) - (i + 1)
                estimated_remaining_time = remaining_samples * avg_time_per_sample
                
                logger.info(f"已处理 {i+1}/{len(sample_names)} 样本, "
                          f"平均时间: {avg_time_per_sample:.2f}秒/样本, "
                          f"预计剩余时间: {estimated_remaining_time/60:.1f}分钟")
        
        # 保存结果
        logger.info("保存结果...")
        json_file, csv_file, detailed_metrics_file, report_file = save_results(results, args.output_dir, logger)
        
        # 输出总结
        total_time = time.time() - start_time
        successful_count = len([r for r in results if r['status'] == 'success'])
        failed_count = len([r for r in results if r['status'] == 'failed'])
        
        logger.info("批量推理完成!")
        logger.info(f"总处理时间: {total_time/60:.1f}分钟")
        logger.info(f"成功样本: {successful_count}")
        logger.info(f"失败样本: {failed_count}")
        logger.info(f"成功率: {successful_count/len(results)*100:.2f}%")
        
        # 计算采样统计
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            total_samples_attempted = sum([r.get('num_sampling', 1) for r in successful_results])
            total_samples_successful = sum([r.get('successful_samples', 1) for r in successful_results])
            avg_sampling_success_rate = total_samples_successful / total_samples_attempted * 100
            
            logger.info(f"采样统计:")
            logger.info(f"  每样本采样次数: {args.num_sampling}")
            logger.info(f"  总采样次数: {total_samples_attempted}")
            logger.info(f"  成功采样次数: {total_samples_successful}")
            logger.info(f"  采样成功率: {avg_sampling_success_rate:.2f}%")
        
        logger.info(f"结果文件:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  详细指标: {detailed_metrics_file}")
        logger.info(f"  报告: {report_file}")
        logger.info(f"  PDB文件目录: {Path(args.output_dir) / 'pdb_files'}")
        
        if args.save_all_samples:
            logger.info(f"  注意: 所有采样的PDB文件也已保存")
        
    except Exception as e:
        logger.error(f"批量推理失败: {e}")
        raise

if __name__ == "__main__":
    main() 