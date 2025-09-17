#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于已生成的PDB文件重新计算准确的结构指标
将RhoFold的预测PDB文件与ground truth PDB文件进行对比
"""

import os
import sys
import json
import csv
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# 导入现有的指标计算模块
sys.path.append(str(Path(__file__).parent))
from diffold.metrics import RNAEvaluationMetrics


def setup_logging() -> logging.Logger:
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pdb_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_pdb_coordinates(pdb_path: str, logger: logging.Logger) -> Optional[torch.Tensor]:
    """从PDB文件加载坐标
    
    Args:
        pdb_path: PDB文件路径
        logger: 日志对象
        
    Returns:
        坐标张量 [n_atoms, 3] 或 None（如果加载失败）
    """
    try:
        coords_list = []
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords_list.append([x, y, z])
                    except ValueError:
                        continue
        
        if not coords_list:
            logger.warning(f"PDB文件中未找到有效坐标: {pdb_path}")
            return None
        
        coords = torch.tensor(coords_list, dtype=torch.float32)
        return coords
            
    except Exception as e:
        logger.error(f"加载PDB文件失败 {pdb_path}: {e}")
        return None


def extract_sample_info_from_filename(filename: str) -> Tuple[str, str]:
    """从文件名提取样本信息
    
    Args:
        filename: PDB文件名（如 1c9s_W.pdb）
        
    Returns:
        (sample_name, chain_id)
    """
    base_name = Path(filename).stem  # 去掉.pdb扩展名
    
    if '_' in base_name:
        parts = base_name.split('_')
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            # 处理复杂的命名（如 4v5k_AV.pdb）
            pdb_id = parts[0]
            chain_part = '_'.join(parts[1:])
            return pdb_id, chain_part
    
    return base_name, ""


def find_matching_ground_truth(predicted_file: str, gt_dir: Path, logger: logging.Logger) -> Optional[str]:
    """为预测文件找到对应的ground truth文件
    
    Args:
        predicted_file: 预测PDB文件名
        gt_dir: ground truth目录
        logger: 日志对象
        
    Returns:
        匹配的ground truth文件路径，如果未找到则返回None
    """
    # 直接查找同名文件
    gt_path = gt_dir / predicted_file
    if gt_path.exists():
        return str(gt_path)
    
    # 如果直接匹配失败，尝试其他匹配策略
    sample_name, chain_id = extract_sample_info_from_filename(predicted_file)
    
    # 尝试不同的命名变体
    possible_names = [
        f"{sample_name}_{chain_id}.pdb",
        f"{sample_name.lower()}_{chain_id.lower()}.pdb",
        f"{sample_name.upper()}_{chain_id.upper()}.pdb",
        f"{sample_name}_{chain_id.lower()}.pdb",
        f"{sample_name.lower()}_{chain_id.upper()}.pdb",
    ]
    
    for name in possible_names:
        gt_path = gt_dir / name
        if gt_path.exists():
            return str(gt_path)
    
    logger.warning(f"未找到匹配的ground truth文件: {predicted_file}")
    return None


def calculate_metrics_for_pair(pred_path: str, gt_path: str, logger: logging.Logger) -> Optional[Dict]:
    """计算一对PDB文件的指标
    
    Args:
        pred_path: 预测PDB文件路径
        gt_path: ground truth PDB文件路径
        logger: 日志对象
        
    Returns:
        指标字典或None（如果计算失败）
    """
    try:
        # 加载坐标
        pred_coords = load_pdb_coordinates(pred_path, logger)
        gt_coords = load_pdb_coordinates(gt_path, logger)
        
        if pred_coords is None or gt_coords is None:
            logger.warning(f"无法加载坐标: {pred_path} 或 {gt_path}")
            return None
        
        # 确保坐标维度匹配
        if len(pred_coords.shape) == 3:
            pred_coords = pred_coords.view(-1, 3)
        if len(gt_coords.shape) == 3:
            gt_coords = gt_coords.view(-1, 3)
        
        # 取最小原子数
        min_atoms = min(pred_coords.shape[0], gt_coords.shape[0])
        pred_coords_eval = pred_coords[:min_atoms]
        gt_coords_eval = gt_coords[:min_atoms]
        
        # 计算指标
        metrics_calculator = RNAEvaluationMetrics()
        metrics_calculator.update(
            loss=0.0,
            batch_size=1,
            predicted_coords=pred_coords_eval.unsqueeze(0),  # 添加batch维度
            target_coords=gt_coords_eval.unsqueeze(0),
            confidence_scores=None  # 从PDB文件无法获取置信度
        )
        
        metrics = metrics_calculator.compute_metrics()
        
        # 提取关键指标
        result = {
            'rmsd': metrics.get('avg_rmsd', float('inf')),
            'tm_score': metrics.get('avg_tm_score', 0.0),
            'lddt': metrics.get('avg_lddt', 0.0),
            'clash_score': metrics.get('avg_clash_score', float('inf')),
            'predicted_atoms': pred_coords.shape[0],
            'target_atoms': gt_coords.shape[0],
            'evaluated_atoms': min_atoms
        }
        
        return result
        
    except Exception as e:
        logger.error(f"计算指标失败 {pred_path} vs {gt_path}: {e}")
        return None


def evaluate_all_predictions(pred_dir: Path, gt_dir: Path, output_dir: Path, logger: logging.Logger):
    """评估所有预测结果
    
    Args:
        pred_dir: 预测PDB文件目录
        gt_dir: ground truth PDB文件目录
        output_dir: 输出目录
        logger: 日志对象
    """
    logger.info("开始基于PDB文件的准确指标计算...")
    
    # 获取所有预测文件
    pred_files = list(pred_dir.glob("*.pdb"))
    logger.info(f"找到 {len(pred_files)} 个预测PDB文件")
    
    results = []
    successful_count = 0
    failed_count = 0
    
    for pred_file in tqdm(pred_files, desc="评估预测结果"):
        sample_name = pred_file.stem
        logger.debug(f"处理样本: {sample_name}")
        
        # 找到对应的ground truth文件
        gt_path = find_matching_ground_truth(pred_file.name, gt_dir, logger)
        
        if gt_path is None:
            failed_count += 1
            results.append({
                'sample_name': sample_name,
                'status': 'no_ground_truth',
                'pred_file': str(pred_file),
                'gt_file': None,
                'rmsd': float('inf'),
                'tm_score': 0.0,
                'lddt': 0.0,
                'clash_score': float('inf'),
                'predicted_atoms': 0,
                'target_atoms': 0,
                'evaluated_atoms': 0
            })
            continue
        
        # 计算指标
        metrics = calculate_metrics_for_pair(str(pred_file), gt_path, logger)
        
        if metrics is None:
            failed_count += 1
            results.append({
                'sample_name': sample_name,
                'status': 'calculation_failed',
                'pred_file': str(pred_file),
                'gt_file': gt_path,
                'rmsd': float('inf'),
                'tm_score': 0.0,
                'lddt': 0.0,
                'clash_score': float('inf'),
                'predicted_atoms': 0,
                'target_atoms': 0,
                'evaluated_atoms': 0
            })
            continue
        
        # 记录成功的结果
        successful_count += 1
        results.append({
            'sample_name': sample_name,
            'status': 'success',
            'pred_file': str(pred_file),
            'gt_file': gt_path,
            'rmsd': metrics['rmsd'],
            'tm_score': metrics['tm_score'],
            'lddt': metrics['lddt'],
            'clash_score': metrics['clash_score'],
            'predicted_atoms': metrics['predicted_atoms'],
            'target_atoms': metrics['target_atoms'],
            'evaluated_atoms': metrics['evaluated_atoms']
        })
        
        if successful_count % 100 == 0:
            logger.info(f"已处理 {successful_count} 个成功样本...")
    
    logger.info(f"评估完成: 成功 {successful_count} 个, 失败 {failed_count} 个")
    
    # 保存结果
    save_results(results, output_dir, logger)
    
    # 生成统计报告
    generate_statistics_report(results, output_dir, logger)


def save_results(results: List[Dict], output_dir: Path, logger: logging.Logger):
    """保存评估结果
    
    Args:
        results: 评估结果列表
        output_dir: 输出目录
        logger: 日志对象
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存CSV文件
    csv_file = output_dir / "pdb_evaluation_results.csv"
    
    fieldnames = [
        'sample_name', 'status', 'pred_file', 'gt_file',
        'rmsd', 'tm_score', 'lddt', 'clash_score',
        'predicted_atoms', 'target_atoms', 'evaluated_atoms'
    ]
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"CSV结果已保存到: {csv_file}")
    
    # 保存JSON文件
    json_file = output_dir / "pdb_evaluation_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"JSON结果已保存到: {json_file}")


def generate_statistics_report(results: List[Dict], output_dir: Path, logger: logging.Logger):
    """生成统计报告
    
    Args:
        results: 评估结果列表
        output_dir: 输出目录
        logger: 日志对象
    """
    # 过滤成功的结果
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        logger.warning("没有成功的评估结果，无法生成统计报告")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(successful_results)
    
    # 计算统计信息
    stats = {}
    metrics = ['rmsd', 'tm_score', 'lddt', 'clash_score']
    
    for metric in metrics:
        values = df[metric].dropna()
        if len(values) > 0:
            stats[metric] = {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'count': int(len(values))
            }
    
    # 生成报告
    report_file = output_dir / "pdb_evaluation_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("基于PDB文件的准确指标评估报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总样本数: {len(results)}\n")
        f.write(f"成功样本数: {len(successful_results)}\n")
        f.write(f"失败样本数: {len(results) - len(successful_results)}\n")
        f.write(f"成功率: {len(successful_results) / len(results) * 100:.2f}%\n\n")
        
        for metric, stat in stats.items():
            f.write(f"{metric.upper()}统计:\n")
            f.write("-" * 30 + "\n")
            
            if metric == 'rmsd':
                unit = "Å"
            elif metric in ['tm_score', 'lddt']:
                unit = ""
            elif metric == 'clash_score':
                unit = ""
            else:
                unit = ""
                
            f.write(f"  平均值: {stat['mean']:.4f}{unit}\n")
            f.write(f"  中位数: {stat['median']:.4f}{unit}\n")
            f.write(f"  标准差: {stat['std']:.4f}{unit}\n")
            f.write(f"  最小值: {stat['min']:.4f}{unit}\n")
            f.write(f"  最大值: {stat['max']:.4f}{unit}\n")
            f.write(f"  样本数: {stat['count']}\n\n")
        
        # 质量分析
        f.write("质量分析:\n")
        f.write("-" * 30 + "\n")
        
        if 'rmsd' in stats:
            rmsd_excellent = (df['rmsd'] < 2.0).sum()
            rmsd_good = ((df['rmsd'] >= 2.0) & (df['rmsd'] < 4.0)).sum()
            rmsd_fair = (df['rmsd'] >= 4.0).sum()
            
            f.write(f"RMSD质量分布:\n")
            f.write(f"  优秀 (<2.0Å): {rmsd_excellent} ({rmsd_excellent/len(df)*100:.1f}%)\n")
            f.write(f"  良好 (2.0-4.0Å): {rmsd_good} ({rmsd_good/len(df)*100:.1f}%)\n")
            f.write(f"  一般 (>4.0Å): {rmsd_fair} ({rmsd_fair/len(df)*100:.1f}%)\n\n")
        
        if 'tm_score' in stats:
            tm_excellent = (df['tm_score'] > 0.8).sum()
            tm_good = ((df['tm_score'] > 0.6) & (df['tm_score'] <= 0.8)).sum()
            tm_fair = (df['tm_score'] <= 0.6).sum()
            
            f.write(f"TM-Score质量分布:\n")
            f.write(f"  优秀 (>0.8): {tm_excellent} ({tm_excellent/len(df)*100:.1f}%)\n")
            f.write(f"  良好 (0.6-0.8): {tm_good} ({tm_good/len(df)*100:.1f}%)\n")
            f.write(f"  一般 (≤0.6): {tm_fair} ({tm_fair/len(df)*100:.1f}%)\n\n")
    
    logger.info(f"统计报告已保存到: {report_file}")
    
    # 打印简要统计到控制台
    logger.info("=== 评估结果摘要 ===")
    logger.info(f"成功评估: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
    
    if 'rmsd' in stats:
        logger.info(f"平均RMSD: {stats['rmsd']['mean']:.3f}Å (中位数: {stats['rmsd']['median']:.3f}Å)")
    
    if 'tm_score' in stats:
        logger.info(f"平均TM-Score: {stats['tm_score']['mean']:.3f} (中位数: {stats['tm_score']['median']:.3f})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于PDB文件重新计算准确的结构指标")
    parser.add_argument("--pred_dir", type=str, default="rhofold_test_output/pdb_files",
                        help="预测PDB文件目录")
    parser.add_argument("--gt_dir", type=str, default="processed_data/pdb", 
                        help="Ground truth PDB文件目录")
    parser.add_argument("--output_dir", type=str, default="pdb_evaluation_output",
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 设置路径
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    
    # 检查输入目录
    if not pred_dir.exists():
        print(f"错误: 预测PDB目录不存在: {pred_dir}")
        sys.exit(1)
    
    if not gt_dir.exists():
        print(f"错误: Ground truth PDB目录不存在: {gt_dir}")
        sys.exit(1)
    
    # 设置日志
    logger = setup_logging()
    
    logger.info(f"预测PDB目录: {pred_dir}")
    logger.info(f"Ground truth目录: {gt_dir}")
    logger.info(f"输出目录: {output_dir}")
    
    # 开始评估
    evaluate_all_predictions(pred_dir, gt_dir, output_dir, logger)
    
    logger.info("评估完成!")


if __name__ == "__main__":
    main()
