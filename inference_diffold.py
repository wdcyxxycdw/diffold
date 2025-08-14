#!/usr/bin/env python3
"""
Diffold推理脚本
将Diffold模型预测的坐标转换为PDB文件并进行结构优化
"""

import logging
from pathlib import Path
import os
import sys
import argparse
from typing import Dict, List, Any, Optional, Callable

import numpy as np
import torch
from tqdm import tqdm

# 导入Diffold相关模块
from diffold.diffold import Diffold
from diffold.dataloader import create_data_loaders
from rhofold.utils import get_device, timing
from rhofold.relax.relax import AmberRelaxation
from rhofold.utils.alphabet import get_features

# 导入PDB转换功能
from diffold.output import diffold_coords_to_pdb, validate_diffold_output

# 导入MSA构建功能
from rhofold.data.balstn import BLASTN

# 尝试导入指标计算模块（可选）
try:
    from diffold.metrics import RNAEvaluationMetrics
except ImportError:
    RNAEvaluationMetrics = None
    print("警告: 无法导入RNAEvaluationMetrics，某些筛选策略可能不可用")

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

def select_random_sample(samples: List[Dict[str, Any]]) -> int:
    """
    随机选择一个采样（用于测试）
    """
    if not samples:
        raise ValueError("样本列表为空")
    
    return np.random.randint(0, len(samples))

# 预定义策略映射
SELECTION_STRATEGIES = {
    'rmsd': select_best_by_rmsd,
    'tm_score': select_best_by_tm_score,
    'lddt': select_best_by_lddt,
    'clash_score': select_best_by_clash_score,
    'composite': select_best_by_composite_score,
    'random': select_random_sample,
}

def compute_simple_rmsd(pred_coords: torch.Tensor, target_coords: torch.Tensor) -> float:
    """
    计算简单的RMSD（当RNAEvaluationMetrics不可用时）
    """
    try:
        # 假设坐标形状为 [batch_size, seq_len, num_atoms, 3]
        # 只计算第一个批次的RMSD
        pred = pred_coords[0].detach().cpu().numpy()  # [seq_len, num_atoms, 3]
        target = target_coords[0].detach().cpu().numpy()  # [seq_len, num_atoms, 3]
        
        # 计算所有原子的平均RMSD
        diff = pred - target
        squared_diff = np.sum(diff ** 2, axis=-1)  # [seq_len, num_atoms]
        msd = np.mean(squared_diff)
        rmsd = np.sqrt(msd)
        
        return float(rmsd)
    except Exception as e:
        print(f"RMSD计算失败: {e}")
        return float('inf')

def perform_multiple_inference(model: Diffold, 
                             tokens: torch.Tensor, 
                             rna_fm_tokens: torch.Tensor, 
                             seq: List[str],
                             num_sampling: int,
                             target_coords: Optional[torch.Tensor] = None,
                             selection_strategy: str = 'rmsd',
                             logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    对单个样本进行多次推理并选择最佳结果
    
    Args:
        model: Diffold模型
        tokens: 输入tokens
        rna_fm_tokens: RNA-FM tokens
        seq: 序列列表
        num_sampling: 采样次数
        target_coords: 目标坐标（用于计算指标，可选）
        selection_strategy: 筛选策略
        logger: 日志器
    
    Returns:
        包含最佳结果和统计信息的字典
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 获取筛选函数
    selection_func = SELECTION_STRATEGIES.get(selection_strategy, select_best_by_rmsd)
    
    all_samples = []
    successful_samples = 0
    
    logger.info(f"开始 {num_sampling} 次推理采样，使用筛选策略: {selection_strategy}")
    
    for sample_idx in range(num_sampling):
        try:
            logger.debug(f"执行第 {sample_idx + 1}/{num_sampling} 次推理")
            
            # 模型推理
            with torch.no_grad():
                model_output = model(
                    tokens=tokens,
                    rna_fm_tokens=rna_fm_tokens,
                    seq=seq
                )
            
            if model_output is None:
                logger.warning(f"采样 {sample_idx + 1}: 模型推理返回None")
                continue
            
            # 提取预测坐标
            predicted_coords = model_output.get('predicted_coords')
            atom_mask = model_output.get('atom_mask', None)
            
            if predicted_coords is None:
                logger.warning(f"采样 {sample_idx + 1}: 未获取到预测坐标")
                continue
            
            # 计算指标
            sample_metrics = {}
            current_rmsd = float('inf')
            
            if target_coords is not None:
                # 如果有目标坐标，计算指标
                if RNAEvaluationMetrics is not None:
                    try:
                        # 使用完整的指标计算器
                        temp_metrics_calculator = RNAEvaluationMetrics()
                        temp_metrics_calculator.update(
                            loss=0.0,
                            batch_size=1,
                            predicted_coords=predicted_coords,
                            target_coords=target_coords
                        )
                        sample_metrics = temp_metrics_calculator.compute_metrics()
                        current_rmsd = sample_metrics.get('avg_rmsd', float('inf'))
                    except Exception as e:
                        logger.warning(f"采样 {sample_idx + 1}: 完整指标计算失败: {e}，使用简单RMSD")
                        current_rmsd = compute_simple_rmsd(predicted_coords, target_coords)
                        sample_metrics = {'avg_rmsd': current_rmsd}
                else:
                    # 使用简单的RMSD计算
                    current_rmsd = compute_simple_rmsd(predicted_coords, target_coords)
                    sample_metrics = {'avg_rmsd': current_rmsd}
            else:
                # 没有目标坐标时，无法计算RMSD，使用占位符
                current_rmsd = 0.0  # 或者使用其他默认值
                sample_metrics = {'avg_rmsd': current_rmsd}
            
            # 验证输出
            validation = validate_diffold_output(predicted_coords, seq[0], atom_mask)
            
            # 保存采样结果
            sample_result = {
                'sample_idx': sample_idx,
                'predicted_coords': predicted_coords,
                'atom_mask': atom_mask,
                'metrics': sample_metrics,
                'rmsd': current_rmsd,
                'validation': validation,
                'model_output': model_output  # 保存完整的模型输出
            }
            
            all_samples.append(sample_result)
            successful_samples += 1
            
            logger.debug(f"采样 {sample_idx + 1}: 成功，RMSD: {current_rmsd:.4f}")
            
        except Exception as e:
            logger.warning(f"采样 {sample_idx + 1} 失败: {e}")
            continue
    
    if not all_samples:
        raise RuntimeError("所有采样都失败了")
    
    # 使用筛选函数选择最佳采样
    try:
        best_sample_idx = selection_func(all_samples)
        best_sample = all_samples[best_sample_idx]
        logger.info(f"筛选函数选择采样 {best_sample_idx + 1}，RMSD: {best_sample['rmsd']:.4f}")
    except Exception as e:
        logger.warning(f"筛选函数执行失败: {e}，使用默认RMSD策略")
        # 回退到默认的RMSD策略
        best_sample_idx = select_best_by_rmsd(all_samples)
        best_sample = all_samples[best_sample_idx]
    
    # 计算采样统计信息
    rmsd_values = [s['rmsd'] for s in all_samples]
    sampling_stats = {
        'rmsd_mean': np.mean(rmsd_values),
        'rmsd_std': np.std(rmsd_values),
        'rmsd_min': np.min(rmsd_values),
        'rmsd_max': np.max(rmsd_values),
        'successful_samples': successful_samples,
        'total_samples': num_sampling,
        'success_rate': successful_samples / num_sampling
    }
    
    # 准备返回结果
    result = {
        'best_sample_idx': best_sample_idx,
        'best_sample': best_sample,
        'all_samples': all_samples,
        'sampling_stats': sampling_stats,
        'selection_strategy': selection_strategy,
        # 为了兼容原始代码，提取最佳样本的关键信息
        'predicted_coords': best_sample['predicted_coords'],
        'atom_mask': best_sample['atom_mask'],
        'validation': best_sample['validation'],
        'model_output': best_sample['model_output']
    }
    
    logger.info(f"多次推理完成: {successful_samples}/{num_sampling} 次成功，" + 
               f"最佳RMSD: {best_sample['rmsd']:.4f}, 平均RMSD: {sampling_stats['rmsd_mean']:.4f} ± {sampling_stats['rmsd_std']:.4f}")
    
    return result

@torch.no_grad()
def main(config):
    '''
    Diffold推理流程
    '''
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 设置日志
    logger = logging.getLogger('Diffold Inference')
    logger.setLevel(level=logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    file_handler = logging.FileHandler(f'{config.output_dir}/diffold_inference.log', mode='w')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    # 设置设备
    config.device = get_device(config.device)
    logger.info(f'使用设备: {config.device}')
    
    # 构建Diffold模型
    logger.info('构建Diffold模型')
    model = Diffold(config, rhofold_checkpoint_path=config.rf_ckpt)
    model = model.to(config.device)
    model.eval()
    
    # 加载检查点
    if config.ckpt:
        logger.info(f'加载检查点: {config.ckpt}')
        try:
            # 尝试只加载模型权重，避免加载优化器状态
            checkpoint = torch.load(config.ckpt, map_location=config.device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            logger.warning(f'weights_only=True 加载失败，回退到完整加载: {e}')
            # 回退到加载完整检查点，但只使用模型权重
            checkpoint = torch.load(config.ckpt, map_location=config.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('检查点加载完成')
    
    # 读取输入序列
    logger.info(f'读取输入序列: {config.input_fas}')
    with open(config.input_fas, 'r') as f:
        lines = f.readlines()
    
    # 解析FASTA文件
    sequences = []
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            continue
        if line:
            sequences.append(line)
    
    if not sequences:
        raise ValueError("未找到有效的序列")
    
    sequence = sequences[0]  # 使用第一个序列
    logger.info(f'序列长度: {len(sequence)}')
    logger.info(f'序列: {sequence}')
    
    # MSA构建逻辑
    logger.info(f"输入FASTA文件: {config.input_fas}")
    
    if config.single_seq_pred:
        config.input_a3m = config.input_fas
        logger.info(f"使用单序列预测模式，MSA文件设置为输入FASTA文件")
    
    elif config.input_a3m is None:
        # 自动构建MSA
        config.input_a3m = f'{config.output_dir}/seq.a3m'
        logger.info(f"未提供MSA文件，开始自动构建MSA: {config.input_a3m}")
        
        # 检查数据库路径
        if not hasattr(config, 'database_dpath') or not config.database_dpath:
            logger.warning("未设置数据库路径，使用默认路径: ./database")
            config.database_dpath = './database'
        
        if not hasattr(config, 'binary_dpath') or not config.binary_dpath:
            logger.warning("未设置BLAST二进制文件路径，使用默认路径: ./rhofold/data/bin")
            config.binary_dpath = './rhofold/data/bin'
        
        # 检查数据库文件是否存在
        databases = [f'{config.database_dpath}/rnacentral.fasta', f'{config.database_dpath}/nt']
        missing_dbs = [db for db in databases if not os.path.exists(db)]
        
        if missing_dbs:
            logger.error(f"缺少数据库文件: {missing_dbs}")
            logger.error("请确保已下载并构建RNA数据库")
            logger.error("可以使用以下命令构建数据库:")
            logger.error("./database/bin/builddb.sh")
            raise FileNotFoundError(f"缺少数据库文件: {missing_dbs}")
        
        # 检查BLAST二进制文件
        blast_binary = f'{config.binary_dpath}/blastn'
        if not os.path.exists(blast_binary):
            logger.error(f"BLAST二进制文件不存在: {blast_binary}")
            raise FileNotFoundError(f"BLAST二进制文件不存在: {blast_binary}")
        
        # 执行BLAST搜索
        try:
            blast = BLASTN(
                binary_dpath=config.binary_dpath, 
                databases=databases,
                n_cpu=getattr(config, 'n_cpu', 4)
            )
            blast.query(config.input_fas, config.input_a3m, logger)
            logger.info(f"MSA构建完成: {config.input_a3m}")
        except Exception as e:
            logger.error(f"MSA构建失败: {e}")
            logger.info("回退到单序列预测模式")
            config.input_a3m = config.input_fas
            config.single_seq_pred = True
    
    else:
        logger.info(f"使用提供的MSA文件: {config.input_a3m}")
    
    # 准备输入数据
    logger.info('准备输入数据')
    data_dict = get_features(config.input_fas, config.input_a3m)
    
    # 转换为Diffold输入格式
    tokens = data_dict['tokens'].to(config.device)  # 添加批次维度
    rna_fm_tokens = data_dict['rna_fm_tokens'].to(config.device)
    seq = [sequence]  # Diffold期望序列列表
    
    logger.info(f'输入张量形状: tokens={tokens.shape}, rna_fm_tokens={rna_fm_tokens.shape}')
    
    # Diffold推理（支持多次采样）
    with timing('Diffold推理', logger=logger):
        logger.info('开始Diffold推理...')
        
        # 检查是否启用多次推理
        num_sampling = getattr(config, 'num_sampling', 1)
        selection_strategy = getattr(config, 'selection_strategy', 'rmsd')
        
        if num_sampling > 1:
            logger.info(f'启用多次推理模式: {num_sampling} 次采样')
            # 使用多次推理函数
            inference_result = perform_multiple_inference(
                model=model,
                tokens=tokens,
                rna_fm_tokens=rna_fm_tokens,
                seq=seq,
                num_sampling=num_sampling,
                target_coords=None,  # 推理时通常没有目标坐标
                selection_strategy=selection_strategy,
                logger=logger
            )
            
            # 提取最佳结果
            predicted_coords = inference_result['predicted_coords']
            atom_mask = inference_result['atom_mask']
            validation = inference_result['validation']
            model_output = inference_result['model_output']
            
            # 保存采样统计信息
            sampling_stats = inference_result['sampling_stats']
            logger.info(f'采样统计: 成功{sampling_stats["successful_samples"]}/{sampling_stats["total_samples"]}次')
            if sampling_stats['successful_samples'] > 1:
                logger.info(f'RMSD变化范围: {sampling_stats["rmsd_min"]:.4f} - {sampling_stats["rmsd_max"]:.4f}, '
                           f'标准差: {sampling_stats["rmsd_std"]:.4f}')
            
            logger.info(f'最终选择第{inference_result["best_sample_idx"] + 1}次采样结果')
        else:
            # 单次推理（原始模式）
            logger.info('使用单次推理模式')
            model_output = model(
                tokens=tokens,
                rna_fm_tokens=rna_fm_tokens,
                seq=seq
            )
            
            # 提取预测坐标
            predicted_coords = model_output['predicted_coords']
            atom_mask = model_output.get('atom_mask', None)
            
            # 验证输出
            validation = validate_diffold_output(predicted_coords, sequence, atom_mask)
            
            # 为了保持一致性，创建inference_result
            inference_result = {
                'predicted_coords': predicted_coords,
                'atom_mask': atom_mask,
                'validation': validation,
                'model_output': model_output,
                'sampling_stats': {
                    'successful_samples': 1,
                    'total_samples': 1,
                    'success_rate': 1.0
                },
                'selection_strategy': 'single'
            }
        
        logger.info(f'推理完成，输出模式: {model_output["mode"]}')
        logger.info(f'预测坐标形状: {predicted_coords.shape}')
        logger.info(f'输出验证结果: {validation}')
        
        if not validation['is_valid']:
            logger.warning('输出验证失败，但继续处理')
    
    # 保存未优化的PDB文件
    unrelaxed_model = f'{config.output_dir}/diffold_unrelaxed_model.pdb'
    logger.info(f'保存未优化的PDB文件: {unrelaxed_model}')
    
    try:
        result_path = diffold_coords_to_pdb(
            predicted_coords=predicted_coords,
            sequence=sequence,
            output_path=unrelaxed_model,
            atom_mask=atom_mask,
            chain_id="A",
            model_name="DIFFOLD_PREDICTION",
            logger_instance=logger
        )
        logger.info(f'未优化PDB文件已保存: {result_path}')
    except Exception as e:
        logger.error(f'保存PDB文件失败: {e}')
        raise
    
    # 保存所有采样的PDB文件（如果启用且是多次采样）
    save_all_samples = getattr(config, 'save_all_samples', False)
    if save_all_samples and num_sampling > 1 and 'all_samples' in inference_result:
        logger.info('保存所有采样的PDB文件...')
        
        # 创建采样PDB目录
        samples_dir = f'{config.output_dir}/all_samples_pdb'
        os.makedirs(samples_dir, exist_ok=True)
        
        for sample in inference_result['all_samples']:
            sample_idx = sample['sample_idx']
            sample_coords = sample['predicted_coords']
            sample_mask = sample['atom_mask']
            sample_rmsd = sample['rmsd']
            
            sample_pdb_path = f'{samples_dir}/diffold_sample_{sample_idx+1:02d}_rmsd_{sample_rmsd:.4f}.pdb'
            
            try:
                diffold_coords_to_pdb(
                    predicted_coords=sample_coords,
                    sequence=sequence,
                    output_path=sample_pdb_path,
                    atom_mask=sample_mask,
                    chain_id="A",
                    model_name=f"DIFFOLD_SAMPLE_{sample_idx+1}",
                    logger_instance=logger
                )
                logger.debug(f'采样{sample_idx+1}的PDB文件已保存: {sample_pdb_path}')
            except Exception as e:
                logger.warning(f'采样{sample_idx+1}的PDB文件保存失败: {e}')
        
        logger.info(f'所有采样PDB文件已保存到: {samples_dir}')
    
    # 保存其他结果
    logger.info('保存其他结果文件')
    
    # 保存坐标数据（包含采样统计信息）
    coords_file = f'{config.output_dir}/diffold_coordinates.npz'
    save_data = {
        'predicted_coords': predicted_coords.detach().cpu().numpy(),
        'atom_mask': atom_mask.detach().cpu().numpy() if atom_mask is not None else None,
        'sequence': sequence,
        'validation': validation,
        'sampling_stats': inference_result['sampling_stats'],
        'selection_strategy': inference_result['selection_strategy']
    }
    
    # 如果是多次采样，保存所有采样结果的简要信息
    if num_sampling > 1 and 'all_samples' in inference_result:
        all_samples_summary = []
        for i, sample in enumerate(inference_result['all_samples']):
            sample_summary = {
                'sample_idx': sample['sample_idx'],
                'rmsd': sample['rmsd'],
                'metrics': sample['metrics'],
                'validation': sample['validation']
            }
            all_samples_summary.append(sample_summary)
        save_data['all_samples_summary'] = all_samples_summary
        save_data['best_sample_idx'] = inference_result['best_sample_idx']
    
    np.savez_compressed(coords_file, **save_data)
    logger.info(f'坐标数据已保存: {coords_file}')
    
    # 如果有置信度信息，也保存
    if 'confidence_logits' in model_output:
        confidence = torch.sigmoid(model_output['confidence_logits']).detach().cpu().numpy()
        confidence_file = f'{config.output_dir}/diffold_confidence.npy'
        np.save(confidence_file, confidence)
        logger.info(f'置信度数据已保存: {confidence_file}')
    
    # 保存采样统计报告（如果是多次采样）
    if num_sampling > 1:
        stats_file = f'{config.output_dir}/sampling_statistics.txt'
        with open(stats_file, 'w') as f:
            f.write("Diffold多次推理采样统计报告\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"序列: {sequence}\n")
            f.write(f"序列长度: {len(sequence)}\n")
            f.write(f"采样次数: {sampling_stats['total_samples']}\n")
            f.write(f"成功采样: {sampling_stats['successful_samples']}\n")
            f.write(f"成功率: {sampling_stats['success_rate']:.2%}\n")
            f.write(f"筛选策略: {selection_strategy}\n")
            f.write(f"最佳采样索引: {inference_result.get('best_sample_idx', 'N/A')}\n\n")
            
            if sampling_stats['successful_samples'] > 1:
                f.write("RMSD统计信息:\n")
                f.write(f"  平均值: {sampling_stats['rmsd_mean']:.4f}\n")
                f.write(f"  标准差: {sampling_stats['rmsd_std']:.4f}\n")
                f.write(f"  最小值: {sampling_stats['rmsd_min']:.4f}\n")
                f.write(f"  最大值: {sampling_stats['rmsd_max']:.4f}\n")
                f.write(f"  变异系数: {sampling_stats['rmsd_std']/sampling_stats['rmsd_mean']:.4f}\n\n")
                
                # 详细的采样结果
                if 'all_samples' in inference_result:
                    f.write("详细采样结果:\n")
                    f.write("采样序号\tRMSD\t验证状态\n")
                    for sample in inference_result['all_samples']:
                        f.write(f"{sample['sample_idx']+1:>6}\t{sample['rmsd']:.4f}\t{sample['validation']['is_valid']}\n")
            
        logger.info(f'采样统计报告已保存: {stats_file}')
    
    # Amber relaxation
    if config.relax_steps is not None:
        relax_steps = int(config.relax_steps)
        if relax_steps > 0:
            logger.info(f'开始Amber优化，步数: {relax_steps}')
            with timing(f'Amber优化: {relax_steps} 步', logger=logger):
                try:
                    amber_relax = AmberRelaxation(
                        max_iterations=relax_steps, 
                        use_gpu=False,  # 强制使用CPU避免CUDA问题
                        logger=logger
                    )
                    relaxed_model = f'{config.output_dir}/diffold_relaxed_{relax_steps}_model.pdb'
                    amber_relax.process(unrelaxed_model, relaxed_model)
                    logger.info(f'优化完成，文件已保存: {relaxed_model}')
                except Exception as e:
                    logger.error(f'Amber优化失败: {e}')
                    logger.info('继续执行，但跳过优化步骤')
        else:
            logger.info('跳过优化步骤 (relax_steps <= 0)')
    else:
        logger.info('跳过优化步骤 (relax_steps 未设置)')
    
    # 输出最终总结
    logger.info('Diffold推理完成！')
    
    if num_sampling > 1:
        logger.info(f'多次推理总结:')
        logger.info(f'  采样次数: {num_sampling}')
        logger.info(f'  成功采样: {sampling_stats["successful_samples"]}')
        logger.info(f'  筛选策略: {selection_strategy}')
        logger.info(f'  最佳采样: 第{inference_result.get("best_sample_idx", "?")+1}次')
        if save_all_samples:
            logger.info(f'  所有采样PDB: {config.output_dir}/all_samples_pdb/')
        logger.info(f'  采样统计: {config.output_dir}/sampling_statistics.txt')
    else:
        logger.info('使用单次推理模式')
    
    logger.info(f'输出文件:')
    logger.info(f'  最佳PDB: {unrelaxed_model}')
    logger.info(f'  坐标数据: {coords_file}')
    if 'confidence_logits' in model_output:
        logger.info(f'  置信度: {confidence_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffold推理脚本')
    
    # 基本参数
    parser.add_argument("--device", 
                       help="设备类型，默认自动检测。可设置为 cuda:<GPU_index> 或 cpu", 
                       default=None)
    parser.add_argument("--ckpt", 
                       help="Diffold模型检查点路径", 
                       default=None,
                       required=True)
    parser.add_argument("--rf_ckpt", 
                       help="RhoFold预训练模型路径", 
                       default='./pretrained/model_20221010_params.pt',
                       required=True)
    
    # 输入输出
    parser.add_argument("--input_fas", 
                       help="输入FASTA文件路径", 
                       default=None,
                       required=True)
    parser.add_argument("--input_a3m", 
                       help="输入MSA文件路径 (可选，如果不提供将自动构建)", 
                       default=None)
    parser.add_argument("--output_dir", 
                       help="输出目录路径", 
                       default=None,
                       required=True)
    
    # MSA构建参数
    parser.add_argument("--single_seq_pred", 
                       help="使用单序列预测模式，不使用MSA", 
                       action='store_true')
    parser.add_argument("--database_dpath", 
                       help="RNA数据库路径，默认 ./database", 
                       default='./database')
    parser.add_argument("--binary_dpath", 
                       help="BLAST二进制文件路径，默认 ./rhofold/data/bin", 
                       default='./rhofold/data/bin')
    parser.add_argument("--n_cpu", 
                       help="BLAST搜索使用的CPU核心数，默认4", 
                       type=int, 
                       default=4)
    
    # 优化参数
    parser.add_argument("--relax_steps", 
                       help="Amber优化步数，默认1000", 
                       type=int, 
                       default=1000)
    
    # 多次推理参数
    parser.add_argument("--num_sampling", 
                       help="每个样本的推理采样次数", 
                       type=int, 
                       default=5)
    parser.add_argument("--selection_strategy", 
                       help="多次采样结果的筛选策略", 
                       choices=['rmsd', 'tm_score', 'lddt', 'clash_score', 'composite', 'random'],
                       default='rmsd')
    parser.add_argument("--save_all_samples", 
                       help="是否保存所有采样的PDB文件（仅在多次采样时有效）", 
                       action='store_true')
    
    # 模型配置
    parser.add_argument("--config_file", 
                       help="Diffold配置文件路径", 
                       default=None)
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，加载配置
    if args.config_file:
        import yaml
        with open(args.config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        # 将命令行参数覆盖配置文件
        for key, value in vars(args).items():
            if value is not None:
                config_dict[key] = value
        args = argparse.Namespace(**config_dict)
    
    main(args) 