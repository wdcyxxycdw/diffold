#!/usr/bin/env python3
"""
使用真实PDB数据测试RMSD计算功能
"""

import os
import sys
import numpy as np
import torch
from typing import Tuple, Optional, List
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffold.metrics import RNAEvaluationMetrics
from rhofold.utils.constants import RNA_CONSTANTS

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_pdb_coordinates(pdb_file: str, chain_id: str = 'A') -> Tuple[np.ndarray, str]:
    """
    从PDB文件中读取原子坐标和序列
    
    Args:
        pdb_file: PDB文件路径
        chain_id: 链ID，默认为'A'
        
    Returns:
        coordinates: [num_atoms, 3] 原子坐标数组
        sequence: RNA序列字符串
    """
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB文件不存在: {pdb_file}")
    
    coordinates = []
    sequence = ""
    current_residue = None
    residue_atoms = {}
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # 只处理ATOM行
                if line.startswith('ATOM'):
                    # 解析PDB ATOM行
                    try:
                        # 提取基本信息
                        atom_name = line[12:16].strip()
                        residue_name = line[17:20].strip()
                        chain = line[21]
                        residue_num = int(line[22:26])
                        
                        # 处理不同的链ID格式（包括数字链ID）
                        if chain != chain_id and chain != '0':  # 允许链ID为'0'
                            continue
                        
                        # 只处理RNA残基
                        if residue_name not in ['A', 'U', 'G', 'C']:
                            continue
                        
                        # 提取坐标
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        
                        # 检查坐标是否有效（不是缺失值）
                        if abs(x) < 999.0 and abs(y) < 999.0 and abs(z) < 999.0:
                            # 如果是新的残基
                            if current_residue != residue_num:
                                # 保存前一个残基的原子
                                if current_residue is not None and residue_atoms:
                                    # 按标准顺序排列原子
                                    ordered_atoms = []
                                    if residue_name in RNA_CONSTANTS.ATOM_NAMES_PER_RESD:
                                        expected_atoms = RNA_CONSTANTS.ATOM_NAMES_PER_RESD[residue_name]
                                        for expected_atom in expected_atoms:
                                            if expected_atom in residue_atoms:
                                                ordered_atoms.append(residue_atoms[expected_atom])
                                    
                                    # 添加到坐标列表
                                    coordinates.extend(ordered_atoms)
                                    
                                    # 添加到序列
                                    if residue_name == 'T':  # DNA->RNA
                                        residue_name = 'U'
                                    sequence += residue_name
                                
                                # 开始新残基
                                current_residue = residue_num
                                residue_atoms = {}
                            
                            # 添加原子坐标
                            residue_atoms[atom_name] = [x, y, z]
                    
                    except (ValueError, IndexError) as e:
                        logger.warning(f"解析PDB行失败: {line[:50]}... 错误: {e}")
                        continue
        
        # 处理最后一个残基
        if current_residue is not None and residue_atoms:
            ordered_atoms = []
            if residue_name in RNA_CONSTANTS.ATOM_NAMES_PER_RESD:
                expected_atoms = RNA_CONSTANTS.ATOM_NAMES_PER_RESD[residue_name]
                for expected_atom in expected_atoms:
                    if expected_atom in residue_atoms:
                        ordered_atoms.append(residue_atoms[expected_atom])
            
            coordinates.extend(ordered_atoms)
            
            if residue_name == 'T':
                residue_name = 'U'
            sequence += residue_name
        
        if not coordinates:
            raise ValueError(f"未能从PDB文件提取到有效坐标: {pdb_file}")
        
        coordinates = np.array(coordinates, dtype=np.float32)
        logger.info(f"成功读取PDB文件: {pdb_file}")
        logger.info(f"  序列长度: {len(sequence)}")
        logger.info(f"  原子数量: {len(coordinates)}")
        logger.info(f"  序列: {sequence}")
        
        return coordinates, sequence
    
    except Exception as e:
        logger.error(f"读取PDB文件失败: {pdb_file}, 错误: {e}")
        raise


def test_rmsd_with_real_pdb(pdb_file1: str, pdb_file2: str, chain_id: str = 'A') -> float:
    """
    使用真实PDB数据测试RMSD计算
    
    Args:
        pdb_file1: 第一个PDB文件路径
        pdb_file2: 第二个PDB文件路径
        chain_id: 链ID
        
    Returns:
        rmsd: 计算得到的RMSD值
    """
    logger.info("=" * 60)
    logger.info("🧬 使用真实PDB数据测试RMSD计算")
    logger.info("=" * 60)
    
    # 读取两个PDB文件
    logger.info(f"📖 读取第一个PDB文件: {pdb_file1}")
    coords1, seq1 = read_pdb_coordinates(pdb_file1, chain_id)
    
    logger.info(f"📖 读取第二个PDB文件: {pdb_file2}")
    coords2, seq2 = read_pdb_coordinates(pdb_file2, chain_id)
    
    # 检查序列是否匹配
    if seq1 != seq2:
        logger.warning(f"序列不匹配!")
        logger.warning(f"  文件1序列: {seq1}")
        logger.warning(f"  文件2序列: {seq2}")
        logger.warning("将使用较短序列的长度进行对齐")
        
        # 使用较短序列的长度
        min_len = min(len(seq1), len(seq2))
        expected_atoms = 0
        for i in range(min_len):
            residue = seq1[i]  # 使用第一个文件的序列
            if residue in RNA_CONSTANTS.ATOM_NAMES_PER_RESD:
                expected_atoms += len(RNA_CONSTANTS.ATOM_NAMES_PER_RESD[residue])
        
        # 截取坐标
        coords1 = coords1[:expected_atoms]
        coords2 = coords2[:expected_atoms]
        logger.info(f"  截取到 {expected_atoms} 个原子")
    
    # 转换为tensor
    coords1_tensor = torch.tensor(coords1, dtype=torch.float32).unsqueeze(0)  # [1, N, 3]
    coords2_tensor = torch.tensor(coords2, dtype=torch.float32).unsqueeze(0)  # [1, N, 3]
    
    logger.info(f"📊 坐标张量形状:")
    logger.info(f"  文件1: {coords1_tensor.shape}")
    logger.info(f"  文件2: {coords2_tensor.shape}")
    
    # 计算RMSD
    metric = RNAEvaluationMetrics()
    rmsd = metric._compute_rmsd(coords1_tensor, coords2_tensor)
    
    logger.info(f"🎯 RMSD计算结果: {rmsd.item():.4f} Å")
    
    # 计算TM-score作为对比
    tm_score = metric._compute_rna_tm_score(coords1_tensor, coords2_tensor)
    logger.info(f"🎯 TM-score计算结果: {tm_score.item():.4f}")
    
    return rmsd.item()


def test_self_rmsd(pdb_file: str, chain_id: str = 'A') -> float:
    """
    测试同一文件与自身的RMSD（应该为0）
    
    Args:
        pdb_file: PDB文件路径
        chain_id: 链ID
        
    Returns:
        rmsd: 计算得到的RMSD值（应该接近0）
    """
    logger.info("=" * 60)
    logger.info("🧪 测试自对齐RMSD（应该为0）")
    logger.info("=" * 60)
    
    coords, seq = read_pdb_coordinates(pdb_file, chain_id)
    coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    
    metric = RNAEvaluationMetrics()
    rmsd = metric._compute_rmsd(coords_tensor, coords_tensor)
    
    logger.info(f"🎯 自对齐RMSD: {rmsd.item():.6f} Å")
    
    if rmsd.item() < 1e-6:
        logger.info("✅ 自对齐RMSD正确（接近0）")
    else:
        logger.warning(f"⚠️ 自对齐RMSD不为0: {rmsd.item():.6f}")
    
    return rmsd.item()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="使用真实PDB数据测试RMSD计算")
    parser.add_argument("--pdb1", type=str, required=True, help="第一个PDB文件路径")
    parser.add_argument("--pdb2", type=str, required=True, help="第二个PDB文件路径")
    parser.add_argument("--chain", type=str, default="A", help="链ID (默认: A)")
    parser.add_argument("--self-test", action="store_true", help="测试自对齐RMSD")
    
    args = parser.parse_args()
    
    try:
        if args.self_test:
            # 测试自对齐
            test_self_rmsd(args.pdb1, args.chain)
        else:
            # 测试两个文件之间的RMSD
            rmsd = test_rmsd_with_real_pdb(args.pdb1, args.pdb2, args.chain)
            logger.info(f"✅ RMSD测试完成: {rmsd:.4f} Å")
    
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


