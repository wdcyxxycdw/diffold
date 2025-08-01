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
from diffold.diffold_to_pdb import diffold_coords_to_pdb, validate_diffold_output

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
    model = Diffold(config, rhofold_checkpoint_path=config.rhofold_checkpoint)
    model = model.to(config.device)
    model.eval()
    
    # 加载检查点
    if config.checkpoint_path:
        logger.info(f'加载检查点: {config.checkpoint_path}')
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
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
    
    # 准备输入数据
    logger.info('准备输入数据')
    data_dict = get_features(config.input_fas, config.input_a3m)
    
    # 转换为Diffold输入格式
    tokens = data_dict['tokens'].unsqueeze(0).to(config.device)  # 添加批次维度
    rna_fm_tokens = data_dict['rna_fm_tokens'].unsqueeze(0).to(config.device)
    seq = [sequence]  # Diffold期望序列列表
    
    logger.info(f'输入张量形状: tokens={tokens.shape}, rna_fm_tokens={rna_fm_tokens.shape}')
    
    # Diffold推理
    with timing('Diffold推理', logger=logger):
        logger.info('开始Diffold推理...')
        
        # 前向传播
        model_output = model(
            tokens=tokens,
            rna_fm_tokens=rna_fm_tokens,
            seq=seq
        )
        
        logger.info(f'推理完成，输出模式: {model_output["mode"]}')
        
        # 提取预测坐标
        predicted_coords = model_output['predicted_coords']
        atom_mask = model_output.get('atom_mask', None)
        
        logger.info(f'预测坐标形状: {predicted_coords.shape}')
        
        # 验证输出
        validation = validate_diffold_output(predicted_coords, sequence, atom_mask)
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
    
    # 保存其他结果
    logger.info('保存其他结果文件')
    
    # 保存坐标数据
    coords_file = f'{config.output_dir}/diffold_coordinates.npz'
    np.savez_compressed(
        coords_file,
        predicted_coords=predicted_coords.detach().cpu().numpy(),
        atom_mask=atom_mask.detach().cpu().numpy() if atom_mask is not None else None,
        sequence=sequence,
        validation=validation
    )
    logger.info(f'坐标数据已保存: {coords_file}')
    
    # 如果有置信度信息，也保存
    if 'confidence_logits' in model_output:
        confidence = torch.sigmoid(model_output['confidence_logits']).detach().cpu().numpy()
        confidence_file = f'{config.output_dir}/diffold_confidence.npy'
        np.save(confidence_file, confidence)
        logger.info(f'置信度数据已保存: {confidence_file}')
    
    # Amber relaxation
    if config.relax_steps is not None:
        relax_steps = int(config.relax_steps)
        if relax_steps > 0:
            logger.info(f'开始Amber优化，步数: {relax_steps}')
            with timing(f'Amber优化: {relax_steps} 步', logger=logger):
                try:
                    amber_relax = AmberRelaxation(
                        max_iterations=relax_steps, 
                        use_gpu=config.device.startswith('cuda'),
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
    
    logger.info('Diffold推理完成！')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffold推理脚本')
    
    # 基本参数
    parser.add_argument("--device", 
                       help="设备类型，默认自动检测。可设置为 cuda:<GPU_index> 或 cpu", 
                       default=None)
    parser.add_argument("--checkpoint_path", 
                       help="Diffold模型检查点路径", 
                       default=None)
    parser.add_argument("--rhofold_checkpoint", 
                       help="RhoFold预训练模型路径", 
                       default='./pretrained/model_20221010_params.pt')
    
    # 输入输出
    parser.add_argument("--input_fas", 
                       help="输入FASTA文件路径", 
                       default='./example/input/3owzA/3owzA.fasta')
    parser.add_argument("--input_a3m", 
                       help="输入MSA文件路径 (可选)", 
                       default=None)
    parser.add_argument("--output_dir", 
                       help="输出目录路径", 
                       default='./example/output/diffold_3owzA')
    
    # 优化参数
    parser.add_argument("--relax_steps", 
                       help="Amber优化步数，默认1000", 
                       type=int, 
                       default=1000)
    
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