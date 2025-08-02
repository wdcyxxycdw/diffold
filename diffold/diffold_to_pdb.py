#!/usr/bin/env python3
"""
Diffold坐标到PDB文件转换工具
将Diffold模型输出的全原子坐标张量转换为标准PDB格式文件
"""

import os
import numpy as np
import torch
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import logging

# 导入项目中的常量
from rhofold.utils.constants import RNA_CONSTANTS

logger = logging.getLogger(__name__)


def diffold_coords_to_pdb(
    predicted_coords: Union[torch.Tensor, np.ndarray],
    sequence: str,
    output_path: str,
    atom_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    confidence: Optional[Union[torch.Tensor, np.ndarray]] = None,
    chain_id: str = "A",
    model_name: str = "DIFFOLD_PREDICTION",
    b_factor: float = 50.0,
    occupancy: float = 1.0,
    logger_instance: Optional[logging.Logger] = None
) -> str:
    """
    将Diffold模型预测的坐标转换为PDB文件
    
    Args:
        predicted_coords: Diffold模型输出的坐标张量
                        形状: [batch_size, num_atoms, 3] 或 [num_atoms, 3]
        sequence: RNA序列字符串 (如 "AUGC")
        output_path: 输出PDB文件的路径
        atom_mask: 原子掩码，指示哪些原子是有效的
                  形状: [batch_size, num_atoms] 或 [num_atoms]
        confidence: 置信度分数，用于设置B因子
                  形状: [batch_size, seq_len] 或 [seq_len]
        chain_id: PDB文件中的链ID
        model_name: 模型名称，用于PDB文件头部
        b_factor: 默认B因子值
        occupancy: 默认占有率
        logger_instance: 日志记录器实例
    
    Returns:
        str: 输出PDB文件的完整路径
    
    Raises:
        ValueError: 当输入参数格式不正确时
        RuntimeError: 当无法创建输出文件时
    """
    
    # 使用传入的logger或默认logger
    log = logger_instance if logger_instance is not None else logger
    
    # 确保输入是numpy数组
    if isinstance(predicted_coords, torch.Tensor):
        predicted_coords = predicted_coords.detach().cpu().numpy()
    
    if isinstance(atom_mask, torch.Tensor):
        atom_mask = atom_mask.detach().cpu().numpy()
    
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.detach().cpu().numpy()
    
    # 处理批次维度
    if predicted_coords.ndim == 3:
        # 如果有批次维度，取第一个样本
        if predicted_coords.shape[0] > 1:
            log.warning(f"检测到批次维度，使用第一个样本 (shape: {predicted_coords.shape})")
        predicted_coords = predicted_coords[0]
    
    if atom_mask is not None and atom_mask.ndim == 2:
        atom_mask = atom_mask[0]
    
    if confidence is not None and confidence.ndim == 2:
        confidence = confidence[0]
    
    # 验证输入
    if predicted_coords.ndim != 2 or predicted_coords.shape[1] != 3:
        raise ValueError(f"坐标张量形状错误: {predicted_coords.shape}, 期望: [num_atoms, 3]")
    
    if len(sequence) == 0:
        raise ValueError("序列不能为空")
    
    # 计算预期的原子数量
    expected_atoms = sum(len(RNA_CONSTANTS.ATOM_NAMES_PER_RESD[res]) for res in sequence)
    actual_atoms = predicted_coords.shape[0]
    
    if actual_atoms != expected_atoms:
        log.warning(f"原子数量不匹配: 预期 {expected_atoms}, 实际 {actual_atoms}")
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入PDB文件
    try:
        with open(output_path, 'w') as f:
            # 写入PDB头部信息
            f.write(f"REMARK 250 MODEL: {model_name}\n")
            f.write(f"REMARK 250 SEQUENCE: {sequence}\n")
            f.write(f"REMARK 250 CHAIN: {chain_id}\n")
            f.write(f"REMARK 250 ATOMS: {actual_atoms}\n")
            f.write("REMARK 250\n")
            
            atom_idx = 0
            res_idx = 0
            
            for seq_idx, residue in enumerate(sequence):
                if residue not in RNA_CONSTANTS.ATOM_NAMES_PER_RESD:
                    log.warning(f"未知残基类型: {residue}, 跳过")
                    continue
                
                residue_atoms = RNA_CONSTANTS.ATOM_NAMES_PER_RESD[residue]
                
                for atom_idx_in_res, atom_name in enumerate(residue_atoms):
                    # 检查是否还有足够的坐标
                    if atom_idx >= predicted_coords.shape[0]:
                        log.warning(f"坐标数量不足，在残基 {seq_idx} 处停止")
                        break
                    
                    # 检查原子掩码
                    if atom_mask is not None:
                        if atom_idx >= atom_mask.shape[0] or atom_mask[atom_idx] == 0:
                            atom_idx += 1
                            continue
                    
                    # 获取坐标
                    coords = predicted_coords[atom_idx]
                    
                    # 检查坐标有效性
                    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                        log.warning(f"残基 {seq_idx} 的原子 {atom_name} 坐标无效，跳过")
                        atom_idx += 1
                        continue
                    
                    # 计算B因子
                    if confidence is not None and seq_idx < len(confidence):
                        b_factor_actual = float(confidence[seq_idx] * 100)
                    else:
                        b_factor_actual = b_factor
                    
                    # 格式化原子行
                    atom_line = format_atom_line(
                        atom_idx + 1,
                        atom_name,
                        residue,
                        chain_id,
                        res_idx + 1,
                        coords[0], coords[1], coords[2],
                        occupancy,
                        b_factor_actual
                    )
                    
                    f.write(atom_line + '\n')
                    atom_idx += 1
                
                res_idx += 1
            
            # 写入PDB文件结束标记
            f.write("END\n")
        
        log.info(f"成功导出PDB文件: {output_path}")
        return str(output_path)
        
    except Exception as e:
        raise RuntimeError(f"写入PDB文件失败: {e}")


def format_atom_line(
    atom_num: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_num: int,
    x: float,
    y: float,
    z: float,
    occupancy: float,
    b_factor: float
) -> str:
    """
    格式化PDB ATOM行，严格遵循PDB格式标准
    
    Args:
        atom_num: 原子编号
        atom_name: 原子名称
        residue_name: 残基名称
        chain_id: 链ID
        residue_num: 残基编号
        x, y, z: 坐标
        occupancy: 占有率
        b_factor: B因子
    
    Returns:
        str: 格式化的ATOM行
    """
    
    # 确保原子名称格式正确(4个字符，左对齐)
    if len(atom_name) == 1:
        atom_name_formatted = f" {atom_name}  "
    elif len(atom_name) == 2:
        atom_name_formatted = f" {atom_name} "
    elif len(atom_name) == 3:
        atom_name_formatted = f" {atom_name}"
    elif len(atom_name) == 4:
        atom_name_formatted = atom_name
    else:
        atom_name_formatted = atom_name[:4]
    
    # 提取元素符号（原子名的第一个字母，通常是元素）
    element = atom_name.strip()[0]
    
    # 严格按照PDB格式构建ATOM行 - 逐列精确构建
    # PDB格式要求坐标从第31列开始（0-based index 30）
    
    # 构建各个部分
    record_name = "ATOM  "                          # 列1-6
    atom_serial = f"{atom_num:5d}"                  # 列7-11  
    space1 = " "                                    # 列12
    atom_name_part = f"{atom_name_formatted}"       # 列13-16
    space2 = " "                                    # 列17
    residue_name_part = f"{residue_name:>3}"        # 列18-20
    space3 = " "                                    # 列21
    chain_part = f"{chain_id}"                      # 列22
    residue_num_part = f"{residue_num:4d}"          # 列23-26
    insertion_code = " "                            # 列27
    spaces_before_coords = "   "                    # 列28-30
    
    # 坐标字段（右对齐，宽度8，精度3）
    x_coord = f"{x:8.3f}"                          # 列31-38
    y_coord = f"{y:8.3f}"                          # 列39-46
    z_coord = f"{z:8.3f}"                          # 列47-54
    
    # 其他字段
    occupancy_part = f"{occupancy:6.2f}"           # 列55-60
    b_factor_part = f"{b_factor:6.2f}"             # 列61-66
    spaces_before_element = "          "            # 列67-76
    element_part = f"{element:>2}"                  # 列77-78
    
    # 组装完整行
    line = (
        record_name + atom_serial + space1 + atom_name_part + space2 +
        residue_name_part + space3 + chain_part + residue_num_part + 
        insertion_code + spaces_before_coords + x_coord + y_coord + z_coord +
        occupancy_part + b_factor_part + spaces_before_element + element_part
    )
    
    # 确保行长度为78字符（不包括换行符）
    if len(line) > 78:
        line = line[:78]
    elif len(line) < 78:
        line = line.ljust(78)
    
    return line


def validate_diffold_output(
    predicted_coords: Union[torch.Tensor, np.ndarray],
    sequence: str,
    atom_mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    验证Diffold输出数据的有效性
    
    Args:
        predicted_coords: 预测坐标
        sequence: RNA序列
        atom_mask: 原子掩码
    
    Returns:
        Dict[str, Any]: 验证结果
    """
    
    if isinstance(predicted_coords, torch.Tensor):
        predicted_coords = predicted_coords.detach().cpu().numpy()
    
    if isinstance(atom_mask, torch.Tensor):
        atom_mask = atom_mask.detach().cpu().numpy()
    
    # 处理批次维度
    if predicted_coords.ndim == 3:
        predicted_coords = predicted_coords[0]
    
    if atom_mask is not None and atom_mask.ndim == 2:
        atom_mask = atom_mask[0]
    
    # 计算预期的原子数量
    expected_atoms = sum(len(RNA_CONSTANTS.ATOM_NAMES_PER_RESD.get(res, [])) for res in sequence)
    actual_atoms = predicted_coords.shape[0]
    
    # 检查坐标有效性
    valid_coords = ~(np.isnan(predicted_coords).any(axis=1) | np.isinf(predicted_coords).any(axis=1))
    num_valid_coords = valid_coords.sum()
    
    # 检查掩码有效性
    num_masked_atoms = 0
    if atom_mask is not None:
        num_masked_atoms = (atom_mask == 1).sum()
    
    return {
        'expected_atoms': expected_atoms,
        'actual_atoms': actual_atoms,
        'valid_coords': num_valid_coords,
        'total_coords': actual_atoms,
        'masked_atoms': num_masked_atoms,
        'sequence_length': len(sequence),
        'is_valid': (actual_atoms >= expected_atoms and num_valid_coords > 0)
    } 