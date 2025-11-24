#!/usr/bin/env python3
"""
RhoFold模型批量测试脚本
基于batch_inference_metrics.py和inference_rf.py设计
专门用于测试RhoFold模型在验证集上的性能
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

# 导入RhoFold相关模块
from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils import get_device, timing
from rhofold.utils.alphabet import get_features

# 导入Diffold数据加载器
from diffold.dataloader import create_data_loaders

# 导入PDB转换功能
from diffold.rhofold_output import rhofold_coords_to_pdb, validate_rhofold_output, extract_rhofold_features

# 导入Amber relaxation
from rhofold.relax.relax import AmberRelaxation

# 导入US-align wrapper用于权威指标计算
import subprocess
import re

# ========== US-align Wrapper for Metrics Calculation ==========

class USalignWrapper:
    """US-align包装器 - 使用权威工具计算 RMSD, TM-score, GDT-TS"""
    
    def __init__(self, usalign_path: str = "./USalign/USalign"):
        self.usalign_path = usalign_path
        
        # 检查 US-align 是否存在
        if not Path(usalign_path).exists():
            raise FileNotFoundError(
                f"US-align 未找到: {usalign_path}\n"
                f"请先编译 US-align:\n"
                f"  cd USalign\n"
                f"  g++ -static -O3 -ffast-math -o USalign USalign.cpp"
            )
    
    def calculate_metrics(self, pred_pdb: str, native_pdb: str) -> Dict:
        """
        使用 US-align 计算 RMSD, TM-score, GDT-TS
        
        返回格式：
        {
            'rmsd': float,
            'tm_score': float,
            'gdt_ts': float,
            'aligned_length': int,
            'seq_identity': float,
            'raw_output': str
        }
        """
        try:
            # 运行 US-align
            cmd = [
                self.usalign_path,
                pred_pdb,
                native_pdb,
                "-mol", "RNA",  # RNA 模式
                "-ter", "0"     # 不按 TER 记录分割链
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {
                    'error': f"US-align failed: {result.stderr}",
                    'returncode': result.returncode
                }
            
            output = result.stdout
            
            # 解析输出
            metrics = {}
            
            # RMSD
            rmsd_match = re.search(r'RMSD=\s*([\d.]+)', output)
            if rmsd_match:
                metrics['rmsd'] = float(rmsd_match.group(1))
            
            # TM-score (使用第一个，通常是按第一个结构归一化的)
            tm_matches = re.findall(r'TM-score=\s*([\d.]+)', output)
            if tm_matches:
                metrics['tm_score'] = float(tm_matches[0])
            
            # GDT-TS
            gdt_match = re.search(r'GDT-TS-score=\s*([\d.]+)', output)
            if gdt_match:
                metrics['gdt_ts'] = float(gdt_match.group(1))
            
            # Aligned length
            len_match = re.search(r'Aligned length=\s*(\d+)', output)
            if len_match:
                metrics['aligned_length'] = int(len_match.group(1))
            
            # Sequence identity
            seqid_match = re.search(r'Seq_ID.*?=\s*([\d.]+)', output)
            if seqid_match:
                metrics['seq_identity'] = float(seqid_match.group(1))
            
            metrics['raw_output'] = output
            
            return metrics
            
        except subprocess.TimeoutExpired:
            return {'error': 'US-align timeout (>30s)'}
        except Exception as e:
            return {'error': f'Exception: {str(e)}'}

# RhoFold是确定性模型，不需要采样相关的函数

def setup_logging(output_dir: str, log_level: str = "INFO"):
    """设置日志"""
    log_file = Path(output_dir) / "rhofold_test.log"
    
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

def load_rhofold_model(config: argparse.Namespace, logger: logging.Logger):
    """加载RhoFold模型"""
    logger.info("构建RhoFold模型")
    model = RhoFold(rhofold_config)
    model = model.to(config.device)
    model.eval()
    
    # 加载检查点
    if config.rhofold_checkpoint:
        logger.info(f"加载RhoFold检查点: {config.rhofold_checkpoint}")
        try:
            # 首先尝试使用weights_only=True加载
            checkpoint = torch.load(config.rhofold_checkpoint, map_location=config.device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("RhoFold检查点加载完成")
        except Exception as e:
            logger.warning(f"weights_only=True 加载失败，尝试使用 weights_only=False: {e}")
            try:
                # 回退到weights_only=False
                checkpoint = torch.load(config.rhofold_checkpoint, map_location=config.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info("RhoFold检查点加载完成")
            except Exception as e2:
                logger.error(f"RhoFold检查点加载失败: {e2}")
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
    
    # 创建数据加载器（复用Diffold的数据加载器）
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

def find_existing_msa(sample_name: str, config: argparse.Namespace, logger: logging.Logger) -> Optional[str]:
    """查找现有的MSA文件"""
    # 定义可能的MSA文件路径和扩展名
    possible_paths = [
        # 优先查找rMSA目录
        Path(config.data_dir) / "rMSA" / f"{sample_name}.a3m",
        Path(config.data_dir) / "rMSA" / f"{sample_name}.fasta",
        # 如果指定了自定义MSA目录
        Path(config.msa_dir) / f"{sample_name}.a3m" if hasattr(config, 'msa_dir') and config.msa_dir else None,
        Path(config.msa_dir) / f"{sample_name}.fasta" if hasattr(config, 'msa_dir') and config.msa_dir else None,
        # 备选路径
        Path(config.data_dir) / "msa" / f"{sample_name}.a3m",
        Path(config.data_dir) / "msa" / f"{sample_name}.fasta",
        Path(config.data_dir) / "alignments" / f"{sample_name}.a3m",
        Path(config.data_dir) / "alignments" / f"{sample_name}.fasta",
    ]
    
    # 过滤掉None值
    possible_paths = [p for p in possible_paths if p is not None]
    
    for msa_path in possible_paths:
        if msa_path.exists():
            logger.debug(f"找到现有MSA文件: {msa_path}")
            return str(msa_path)
    
    logger.debug(f"未找到样本 {sample_name} 的MSA文件，尝试路径: {[str(p) for p in possible_paths]}")
    return None

def get_msa_or_fasta(sequence: str, sample_name: str, config: argparse.Namespace, 
                     logger: logging.Logger) -> str:
    """获取MSA文件或创建临时FASTA文件"""
    if config.single_seq_pred:
        # 单序列预测：创建临时FASTA文件
        temp_fasta_path = Path(config.output_dir) / "temp" / f"{sample_name}.fasta"
        temp_fasta_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_fasta_path, 'w') as f:
            f.write(f">{sample_name}\n{sequence}\n")
        
        logger.debug(f"单序列模式，创建临时FASTA: {temp_fasta_path}")
        return str(temp_fasta_path)
    
    else:
        # 查找现有MSA文件
        msa_path = find_existing_msa(sample_name, config, logger)
        
        if msa_path:
            return msa_path
        
        # 如果没有找到MSA文件，回退到单序列模式
        logger.warning(f"样本 {sample_name}: 未找到MSA文件，回退到单序列模式")
        temp_fasta_path = Path(config.output_dir) / "temp" / f"{sample_name}.fasta"
        temp_fasta_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_fasta_path, 'w') as f:
            f.write(f">{sample_name}\n{sequence}\n")
        
        return str(temp_fasta_path)

def compute_metrics_for_sample(model: RhoFold, 
                             batch: Dict[str, torch.Tensor], 
                             sample_name: str,
                             usalign_wrapper: USalignWrapper,
                             output_dir: str,
                             config: argparse.Namespace,
                             logger: logging.Logger) -> Dict[str, Any]:
    """计算单个样本的指标并保存PDB文件"""
    try:
        # 准备输入数据
        device = next(model.parameters()).device
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
        
        # 从原始序列文件读取完整序列，而不是使用可能被截断的数据加载器序列
        sequence_file = Path(config.data_dir) / "sequences" / f"{sample_name}.fasta"
        if sequence_file.exists():
            with open(sequence_file, 'r') as f:
                lines = f.readlines()
                sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
            logger.debug(f"样本 {sample_name}: 从文件读取完整序列，长度: {len(sequence)}")
        else:
            # 回退到数据加载器提供的序列
            sequence = sequences[0] if sequences else ""
            logger.warning(f"样本 {sample_name}: 未找到序列文件，使用数据加载器序列，长度: {len(sequence)}")
        
        # 准备RhoFold输入
        # 查找现有MSA文件
        msa_path = find_existing_msa(sample_name, config, logger)
        
        logger.debug(f"样本 {sample_name}: 开始RhoFold推理")
        
        try:
            # 创建临时FASTA文件（使用完整序列）
            temp_fasta_path = Path(config.output_dir) / "temp" / f"{sample_name}.fasta"
            temp_fasta_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_fasta_path, 'w') as f:
                f.write(f">{sample_name}\n{sequence}\n")
            
            # 获取RhoFold特征
            # 如果没有MSA文件，使用单序列预测
            if msa_path is None:
                logger.info(f"样本 {sample_name}: 未找到MSA文件，使用单序列模式")
                data_dict = get_features(str(temp_fasta_path), str(temp_fasta_path))
            else:
                data_dict = get_features(str(temp_fasta_path), msa_path)
            
            # RhoFold模型推理
            with torch.no_grad():
                outputs = model(
                    tokens=data_dict['tokens'].to(device),
                    rna_fm_tokens=data_dict['rna_fm_tokens'].to(device),
                    seq=data_dict['seq']
                )
            
            # 提取预测结果
            output = outputs[0][-1]  # 获取最后一层的输出
            
            # 使用专用函数提取特征
            features = extract_rhofold_features(output)
            
            # 提取坐标预测
            predicted_coords = features.get('predicted_coords')
            if predicted_coords is not None:
                # 安全地处理坐标维度
                if isinstance(predicted_coords, torch.Tensor):
                    if predicted_coords.dim() > 3:
                        predicted_coords = predicted_coords.squeeze(0)  # 移除batch维度
                elif isinstance(predicted_coords, (list, tuple)):
                    # 如果是列表或元组，取第一个元素
                    if len(predicted_coords) > 0:
                        predicted_coords = predicted_coords[0]
                        if isinstance(predicted_coords, torch.Tensor) and predicted_coords.dim() > 3:
                            predicted_coords = predicted_coords.squeeze(0)
            confidence = features.get('confidence')
            
            # 安全地处理置信度数据
            if confidence is not None:
                if isinstance(confidence, (list, tuple)):
                    if len(confidence) > 0:
                        confidence = confidence[0]
                elif isinstance(confidence, torch.Tensor):
                    if confidence.dim() > 2:
                        confidence = confidence.squeeze(0)
            
            if predicted_coords is None:
                logger.warning(f"样本 {sample_name}: 未获取到预测坐标")
                return {
                    'sample_name': sample_name,
                    'status': 'failed',
                    'error': 'no_predicted_coords'
                }
            
            # 转换坐标格式以匹配目标坐标的维度
            if len(predicted_coords.shape) == 3:
                # 如果是[seq_len, atom_types, 3]，需要展平为[seq_len * atom_types, 3]
                predicted_coords_flat = predicted_coords.view(-1, 3)
            else:
                predicted_coords_flat = predicted_coords
            
            # 指标将在保存PDB后使用US-align计算
            current_rmsd = float('inf')  # 临时占位
            sample_metrics = {}
            detailed_metrics = {}
            
        except Exception as e:
            logger.error(f"样本 {sample_name} RhoFold推理失败: {e}")
            return {
                'sample_name': sample_name,
                'status': 'failed',
                'error': f'inference_failed: {str(e)}'
            }
        
        # 保存PDB文件
        pdb_file_paths = []
        
        # 创建PDB输出目录
        pdb_output_dir = Path(output_dir) / "pdb_files"
        pdb_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存预测结果的PDB文件（unrelaxed）
        unrelaxed_pdb_path = pdb_output_dir / f"{sample_name}.pdb"
        success = False
        try:
            success = rhofold_coords_to_pdb(
                predicted_coords=predicted_coords,
                sequence=sequence,
                output_path=str(unrelaxed_pdb_path),
                confidence=confidence,
                model_instance=model,  # 传递RhoFold模型实例
                logger_instance=logger
            )
            if success:
                pdb_file_paths.append(str(unrelaxed_pdb_path))
                logger.debug(f"样本 {sample_name}: Unrelaxed PDB文件已保存到 {unrelaxed_pdb_path}")
        except Exception as e:
            logger.warning(f"样本 {sample_name}: PDB文件保存失败: {e}")
            success = False
        
        # Amber relaxation
        relaxed_pdb_path = None
        if config.relax_steps is not None:
            relax_steps = int(config.relax_steps)
            if relax_steps > 0 and success:  # 只有在成功保存了unrelaxed PDB后才进行relax
                try:
                    logger.debug(f"样本 {sample_name}: 开始Amber优化，步数: {relax_steps}")
                    with timing(f'Amber Relaxation: {relax_steps} iterations', logger=logger):
                        amber_relax = AmberRelaxation(max_iterations=relax_steps, logger=logger)
                        relaxed_pdb_path = pdb_output_dir / f"{sample_name}_relaxed_{relax_steps}.pdb"
                        amber_relax.process(str(unrelaxed_pdb_path), str(relaxed_pdb_path))
                        pdb_file_paths.append(str(relaxed_pdb_path))
                        logger.debug(f"样本 {sample_name}: Relaxed PDB文件已保存到 {relaxed_pdb_path}")
                except Exception as e:
                    logger.warning(f"样本 {sample_name}: Amber优化失败: {e}")
                    logger.info(f"样本 {sample_name}: 继续使用unrelaxed模型")
        
        # 使用US-align计算指标
        native_pdb_path = Path(config.data_dir) / "pdb" / f"{sample_name}.pdb"
        if success and native_pdb_path.exists():
            logger.debug(f"样本 {sample_name}: 使用US-align计算指标")
            # 使用最终的PDB（relaxed如果可用，否则unrelaxed）
            final_pdb = relaxed_pdb_path if relaxed_pdb_path else unrelaxed_pdb_path
            
            usalign_metrics = usalign_wrapper.calculate_metrics(
                str(final_pdb),
                str(native_pdb_path)
            )
            
            if 'error' not in usalign_metrics:
                current_rmsd = usalign_metrics.get('rmsd', float('inf'))
                sample_metrics = {
                    'avg_rmsd': usalign_metrics.get('rmsd'),
                    'avg_tm_score': usalign_metrics.get('tm_score'),
                    'gdt_ts': usalign_metrics.get('gdt_ts'),
                    'aligned_length': usalign_metrics.get('aligned_length'),
                }
                logger.debug(f"样本 {sample_name}: US-align指标: RMSD={current_rmsd:.4f}, TM={sample_metrics.get('avg_tm_score', 0):.4f}")
            else:
                logger.warning(f"样本 {sample_name}: US-align失败: {usalign_metrics['error']}")
                sample_metrics = {}
        elif not native_pdb_path.exists():
            logger.warning(f"样本 {sample_name}: 真实PDB不存在: {native_pdb_path}")
            sample_metrics = {}
        
        # 准备返回结果
        result_dict = {
            'sample_name': sample_name,
            'status': 'success',
            'sequence_length': len(sequence),
            'sequence': sequence,
            'rmsd': current_rmsd,
            'predicted_coords_shape': list(predicted_coords.shape),
            'target_coords_shape': list(target_coords.shape),
            'pdb_file_paths': pdb_file_paths,
            'pdb_path': str(unrelaxed_pdb_path) if pdb_file_paths else None,
            'relaxed_pdb_path': str(relaxed_pdb_path) if relaxed_pdb_path else None,
            # 使用计算的指标
            **sample_metrics,
            'detailed_metrics': detailed_metrics
        }
        
        logger.debug(f"样本 {sample_name}: RhoFold推理完成，RMSD: {current_rmsd:.4f}")
        
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
    json_file = output_path / "rhofold_test_results.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"结果已保存到: {json_file}")
    
    # 保存为CSV格式
    csv_file = output_path / "rhofold_test_results.csv"
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
    report_file = output_path / "rhofold_test_report.txt"
    generate_report(results, report_file, logger)
    
    return json_file, csv_file, detailed_metrics_file, report_file

def generate_report(results: List[Dict[str, Any]], report_file: Path, logger: logging.Logger):
    """生成统计报告"""
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']
    
    with open(report_file, 'w') as f:
        f.write("RhoFold模型批量测试报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总样本数: {len(results)}\n")
        f.write(f"成功样本数: {len(successful_results)}\n")
        f.write(f"失败样本数: {len(failed_results)}\n")
        f.write(f"成功率: {len(successful_results)/len(results)*100:.2f}%\n\n")
        
        if successful_results:
            # RMSD统计
            rmsd_values = [r.get('rmsd', r.get('avg_rmsd', float('inf'))) for r in successful_results]
            rmsd_values = [v for v in rmsd_values if v != float('inf')]
            
            if rmsd_values:
                f.write("RMSD统计:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  平均值: {np.mean(rmsd_values):.4f}Å\n")
                f.write(f"  中位数: {np.median(rmsd_values):.4f}Å\n")
                f.write(f"  标准差: {np.std(rmsd_values):.4f}Å\n")
                f.write(f"  最小值: {np.min(rmsd_values):.4f}Å\n")
                f.write(f"  最大值: {np.max(rmsd_values):.4f}Å\n\n")
            
            # 计算其他指标统计
            metrics_keys = ['avg_tm_score', 'avg_lddt', 'avg_confidence']
            available_metrics = [key for key in metrics_keys if any(key in r for r in successful_results)]
            
            if available_metrics:
                f.write("其他指标统计:\n")
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
            
            # 序列长度统计
            seq_lengths = [r.get('sequence_length', 0) for r in successful_results if 'sequence_length' in r]
            if seq_lengths:
                f.write("序列长度统计:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  平均长度: {np.mean(seq_lengths):.1f}\n")
                f.write(f"  中位数长度: {np.median(seq_lengths):.1f}\n")
                f.write(f"  最短序列: {np.min(seq_lengths)}\n")
                f.write(f"  最长序列: {np.max(seq_lengths)}\n\n")
        
        if failed_results:
            f.write("失败样本:\n")
            f.write("-" * 30 + "\n")
            for result in failed_results:
                f.write(f"{result['sample_name']}: {result.get('error', 'unknown_error')}\n")
    
    logger.info(f"报告已保存到: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="RhoFold模型批量测试")
    
    # 基本参数
    parser.add_argument("--data_dir", default="./processed_data", 
                       help="数据目录路径")
    parser.add_argument("--output_dir", default="./rhofold_test_output", 
                       help="输出目录路径")
    parser.add_argument("--fold", type=int, default=3, 
                       help="验证集折数")
    
    # 模型参数
    parser.add_argument("--rhofold_checkpoint", required=True,
                       help="RhoFold模型检查点路径")
    
    # 数据参数
    parser.add_argument("--max_sequence_length", type=int, default=256,
                       help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="数据加载器工作进程数")
    parser.add_argument("--use_msa", action="store_true", default=True,
                       help="是否使用MSA")
    
    # MSA相关参数
    parser.add_argument("--single_seq_pred", action="store_true", default=False,
                       help="使用单序列预测（不使用MSA）")
    parser.add_argument("--msa_dir", default=None,
                       help="MSA文件目录路径（如果未指定，将优先在data_dir/rMSA目录中查找，然后尝试msa和alignments子目录）")
    
    # 设备参数
    parser.add_argument("--device", default="auto",
                       help="计算设备 (auto, cpu, cuda)")
    parser.add_argument("--log_level", default="INFO",
                       help="日志级别")
    
    # 可选参数
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大处理样本数（用于测试）")
    
    # Amber relaxation参数
    parser.add_argument("--relax_steps", type=int, default=None,
                       help="Number of steps for Amber relaxation (default: None, no relaxation). "
                            "If set to a positive value, will perform structure refinement using Amber.")
    
    # US-align参数
    parser.add_argument("--usalign_path", default="./USalign/USalign",
                       help="US-align可执行文件路径 (用于计算权威指标)")
    
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
    logger.info("开始RhoFold模型批量测试")
    logger.info(f"设备: {args.device}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"单序列预测模式: {args.single_seq_pred}")
    if args.relax_steps is not None and args.relax_steps > 0:
        logger.info(f"Amber relaxation已启用，步数: {args.relax_steps}")
    else:
        logger.info("Amber relaxation未启用")
    
    try:
        # 加载RhoFold模型
        model = load_rhofold_model(args, logger)
        
        # 加载验证数据
        valid_loader, sample_names = load_validation_data(args, logger)
        
        # 创建US-align包装器用于指标计算
        usalign_wrapper = USalignWrapper(usalign_path=args.usalign_path)
        logger.info(f"✅ US-align已就绪: {args.usalign_path}")
        
        # 批量处理
        results = []
        start_time = time.time()
        
        # 限制样本数量（用于测试）
        if args.max_samples:
            sample_names = sample_names[:args.max_samples]
            logger.info(f"限制处理样本数为: {len(sample_names)}")
        
        logger.info("开始批量测试...")
        for i, (batch, sample_name) in enumerate(tqdm(zip(valid_loader, sample_names), 
                                                   total=len(sample_names),
                                                   desc="测试样本")):
            
            logger.debug(f"测试样本 {i+1}/{len(sample_names)}: {sample_name}")
            
            # 计算指标
            result = compute_metrics_for_sample(
                model=model,
                batch=batch,
                sample_name=sample_name,
                usalign_wrapper=usalign_wrapper,
                output_dir=args.output_dir,
                config=args,
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
        
        logger.info("RhoFold批量测试完成!")
        logger.info(f"总处理时间: {total_time/60:.1f}分钟")
        logger.info(f"成功样本: {successful_count}")
        logger.info(f"失败样本: {failed_count}")
        logger.info(f"成功率: {successful_count/len(results)*100:.2f}%")
        
        # 计算基本统计
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            rmsd_values = [r.get('rmsd', float('inf')) for r in successful_results if 'rmsd' in r]
            if rmsd_values:
                logger.info(f"RMSD统计:")
                logger.info(f"  平均值: {np.mean(rmsd_values):.4f}Å")
                logger.info(f"  中位数: {np.median(rmsd_values):.4f}Å")
                logger.info(f"  最小值: {np.min(rmsd_values):.4f}Å")
                logger.info(f"  最大值: {np.max(rmsd_values):.4f}Å")
        
        logger.info(f"结果文件:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  详细指标: {detailed_metrics_file}")
        logger.info(f"  报告: {report_file}")
        logger.info(f"  PDB文件目录: {Path(args.output_dir) / 'pdb_files'}")
        
    except Exception as e:
        logger.error(f"RhoFold批量测试失败: {e}")
        raise

if __name__ == "__main__":
    main()
