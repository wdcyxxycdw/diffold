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
from typing import Dict, List, Any, Optional
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
    
    return model

def load_validation_data(config: argparse.Namespace, logger: logging.Logger):
    """加载验证数据"""
    logger.info("加载验证数据")
    
    # 读取验证集样本列表
    valid_list_file = Path(config.data_dir) / "list" / f"valid_fold-{config.fold}"
    if not valid_list_file.exists():
        raise FileNotFoundError(f"验证集列表文件不存在: {valid_list_file}")
    
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
                             logger: logging.Logger) -> Dict[str, Any]:
    """计算单个样本的指标并保存PDB文件"""
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
        
        # 模型推理
        with torch.no_grad():
            result = model(
                tokens=tokens,
                rna_fm_tokens=rna_fm_tokens,
                seq=sequences,
                target_coords=coordinates,
                missing_atom_mask=missing_atom_masks
            )
        
        if result is None:
            logger.warning(f"样本 {sample_name}: 模型推理返回None")
            return {
                'sample_name': sample_name,
                'status': 'failed',
                'error': 'model_inference_failed'
            }
        
        # 提取预测坐标和目标坐标
        predicted_coords = result.get('predicted_coords')
        target_coords = coordinates
        atom_mask = result.get('atom_mask', None)
        
        if predicted_coords is None:
            logger.warning(f"样本 {sample_name}: 未获取到预测坐标")
            return {
                'sample_name': sample_name,
                'status': 'failed',
                'error': 'no_predicted_coords'
            }
        
        if target_coords is None:
            logger.warning(f"样本 {sample_name}: 未获取到目标坐标")
            return {
                'sample_name': sample_name,
                'status': 'failed',
                'error': 'no_target_coords'
            }
        
        # 计算指标
        metrics_calculator.reset()  # 重置指标计算器
        metrics_calculator.update(
            loss=0.0,  # 推理时没有损失
            batch_size=1,
            predicted_coords=predicted_coords,
            target_coords=target_coords
        )
        
        # 获取计算结果
        metrics = metrics_calculator.compute_metrics()
        
        # 获取详细的指标数据
        detailed_metrics = {}
        if hasattr(metrics_calculator, 'rmsd_values') and metrics_calculator.rmsd_values:
            detailed_metrics['rmsd_values'] = metrics_calculator.rmsd_values
        if hasattr(metrics_calculator, 'tm_scores') and metrics_calculator.tm_scores:
            detailed_metrics['tm_scores'] = metrics_calculator.tm_scores
        if hasattr(metrics_calculator, 'lddt_scores') and metrics_calculator.lddt_scores:
            detailed_metrics['lddt_scores'] = metrics_calculator.lddt_scores
        if hasattr(metrics_calculator, 'clash_scores') and metrics_calculator.clash_scores:
            detailed_metrics['clash_scores'] = metrics_calculator.clash_scores
        
        # 保存PDB文件
        pdb_file_path = None
        try:
            # 创建PDB输出目录
            pdb_output_dir = Path(output_dir) / "pdb_files"
            pdb_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成PDB文件路径
            pdb_file_path = pdb_output_dir / f"{sample_name}_predicted.pdb"
            
            # 获取序列
            sequence = sequences[0] if sequences else ""
            
            # 保存预测的PDB文件
            diffold_coords_to_pdb(
                predicted_coords=predicted_coords,
                sequence=sequence,
                output_path=str(pdb_file_path),
                atom_mask=atom_mask,
                logger_instance=logger
            )
            
            logger.debug(f"样本 {sample_name}: PDB文件已保存到 {pdb_file_path}")
            
        except Exception as e:
            logger.warning(f"样本 {sample_name}: PDB文件保存失败: {e}")
            pdb_file_path = None
        
        # 添加样本信息
        result_dict = {
            'sample_name': sample_name,
            'status': 'success',
            'sequence_length': len(sequences[0]) if sequences else 0,
            'sequence': sequences[0] if sequences else "",
            'predicted_coords_shape': list(predicted_coords.shape) if predicted_coords is not None else None,
            'target_coords_shape': list(target_coords.shape) if target_coords is not None else None,
            'pdb_file_path': str(pdb_file_path) if pdb_file_path else None,
            **metrics,
            'detailed_metrics': detailed_metrics
        }
        
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
        f.write("批量推理指标计算报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总样本数: {len(results)}\n")
        f.write(f"成功样本数: {len(successful_results)}\n")
        f.write(f"失败样本数: {len(failed_results)}\n")
        f.write(f"成功率: {len(successful_results)/len(results)*100:.2f}%\n\n")
        
        if successful_results:
            # 计算指标统计
            metrics_keys = ['avg_rmsd', 'avg_tm_score', 'avg_lddt', 'avg_clash_score']
            available_metrics = [key for key in metrics_keys if any(key in r for r in successful_results)]
            
            f.write("指标统计:\n")
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
    parser.add_argument("--fold", type=int, default=3, 
                       help="验证集折数")
    
    # 模型参数
    parser.add_argument("--checkpoint_path", required=True,
                       help="模型检查点路径")
    parser.add_argument("--rhofold_checkpoint", default="./pretrained/model_20221010_params.pt",
                       help="RhoFold检查点路径")
    
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
    
    args = parser.parse_args()
    
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
    logger.info("开始批量推理和指标计算")
    logger.info(f"设备: {args.device}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    
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
        for i, (batch, sample_name) in enumerate(tqdm(zip(valid_loader, sample_names), 
                                                   total=len(sample_names),
                                                   desc="处理样本")):
            
            logger.debug(f"处理样本 {i+1}/{len(sample_names)}: {sample_name}")
            
            # 计算指标
            result = compute_metrics_for_sample(
                model=model,
                batch=batch,
                sample_name=sample_name,
                metrics_calculator=metrics_calculator,
                output_dir=args.output_dir,
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
        logger.info(f"结果文件:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  详细指标: {detailed_metrics_file}")
        logger.info(f"  报告: {report_file}")
        logger.info(f"  PDB文件目录: {Path(args.output_dir) / 'pdb_files'}")
        
    except Exception as e:
        logger.error(f"批量推理失败: {e}")
        raise

if __name__ == "__main__":
    main() 