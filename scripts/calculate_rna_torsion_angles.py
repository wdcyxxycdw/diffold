#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算RNA扭转角的MAE和RMSE

从预测文件夹和ground truth文件夹中读取PDB文件，计算每个扭转角的MAE和RMSE。
支持的扭转角：alpha, beta, gamma, delta, epsilon, zeta, chi

用法:
    python calculate_rna_torsion_angles.py --pred_folder <预测文件夹> --gt_folder <ground_truth文件夹> [--output <输出文件>]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_pdb_atoms(pdb_file: Path) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """
    解析PDB文件，提取每个残基的原子坐标
    
    参数:
        pdb_file: PDB文件路径
        
    返回:
        Dict[chain_id, Dict[residue_num, Dict[atom_name, coords]]]
    """
    atoms_dict = {}
    in_model = False
    model_count = 0
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                # 处理多模型PDB文件
                if line.startswith('MODEL'):
                    model_count += 1
                    if model_count == 1:
                        in_model = True
                    continue
                
                if line.startswith('ENDMDL'):
                    if model_count == 1:
                        in_model = False
                        break
                    continue
                
                if model_count > 0 and not in_model:
                    continue
                
                # 只处理ATOM记录
                if not line.startswith('ATOM'):
                    continue
                
                if len(line) < 54:
                    continue
                
                try:
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21:22].strip() or 'A'
                    residue_num = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except (ValueError, IndexError):
                    continue
                
                # 只处理RNA残基 (A, G, U, C)
                if residue_name not in ['A', 'G', 'U', 'C', 'ADE', 'GUA', 'URA', 'CYT']:
                    continue
                
                # 标准化残基名称
                if residue_name == 'ADE':
                    residue_name = 'A'
                elif residue_name == 'GUA':
                    residue_name = 'G'
                elif residue_name == 'URA':
                    residue_name = 'U'
                elif residue_name == 'CYT':
                    residue_name = 'C'
                
                if chain_id not in atoms_dict:
                    atoms_dict[chain_id] = {}
                
                if residue_num not in atoms_dict[chain_id]:
                    atoms_dict[chain_id][residue_num] = {
                        'residue_name': residue_name,
                        'atoms': {}
                    }
                
                atoms_dict[chain_id][residue_num]['atoms'][atom_name] = np.array([x, y, z])
    
    except Exception as e:
        logger.error(f"解析PDB文件 {pdb_file} 时出错: {e}")
        return {}
    
    return atoms_dict


def calc_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    计算四个点之间的二面角（扭转角）
    
    参数:
        p1, p2, p3, p4: 四个点的坐标
        
    返回:
        角度（度）
    """
    try:
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        
        if n1_norm < 1e-6 or n2_norm < 1e-6:
            return np.nan
        
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        
        cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # 判断符号
        if np.dot(np.cross(n1, n2), v2 / np.linalg.norm(v2)) < 0:
            angle = -angle
        
        return np.degrees(angle)
    
    except Exception:
        return np.nan


def get_atom_coord(atoms: Dict[str, np.ndarray], atom_name: str) -> Optional[np.ndarray]:
    """获取原子坐标，支持多种可能的原子名称"""
    # 尝试精确匹配
    if atom_name in atoms:
        return atoms[atom_name]
    
    # 尝试去除空格和引号
    atom_name_clean = atom_name.replace("'", "").replace(" ", "")
    for key in atoms.keys():
        key_clean = key.replace("'", "").replace(" ", "")
        if key_clean == atom_name_clean:
            return atoms[key]
    
    return None


def calculate_torsion_angles(atoms_dict: Dict[str, Dict[int, Dict]]) -> Dict[str, List[float]]:
    """
    计算RNA扭转角
    
    参数:
        atoms_dict: 解析后的原子字典
        
    返回:
        Dict[torsion_name, List[angles]]: 每个扭转角的列表
    """
    torsion_angles = {
        'alpha': [],
        'beta': [],
        'gamma': [],
        'delta': [],
        'epsilon': [],
        'zeta': [],
        'chi': []
    }
    
    for chain_id, chain_residues in atoms_dict.items():
        # 按残基编号排序
        residue_nums = sorted(chain_residues.keys())
        
        for i, res_num in enumerate(residue_nums):
            residue = chain_residues[res_num]
            atoms = residue['atoms']
            residue_name = residue['residue_name']
            
            # 获取当前残基的原子
            P = get_atom_coord(atoms, "P")
            O5_prime = get_atom_coord(atoms, "O5'")
            C5_prime = get_atom_coord(atoms, "C5'")
            C4_prime = get_atom_coord(atoms, "C4'")
            C3_prime = get_atom_coord(atoms, "C3'")
            O3_prime = get_atom_coord(atoms, "O3'")
            C1_prime = get_atom_coord(atoms, "C1'")
            O4_prime = get_atom_coord(atoms, "O4'")
            
            # 获取碱基原子（用于chi角）
            if residue_name in ['A', 'G']:
                N9 = get_atom_coord(atoms, "N9")
                C4 = get_atom_coord(atoms, "C4")
            else:  # U, C
                N1 = get_atom_coord(atoms, "N1")
                C2 = get_atom_coord(atoms, "C2")
            
            # 获取前一个残基的O3'
            prev_O3_prime = None
            if i > 0:
                prev_res_num = residue_nums[i - 1]
                prev_atoms = chain_residues[prev_res_num]['atoms']
                prev_O3_prime = get_atom_coord(prev_atoms, "O3'")
            
            # 获取下一个残基的P和O5'
            next_P = None
            next_O5_prime = None
            if i < len(residue_nums) - 1:
                next_res_num = residue_nums[i + 1]
                next_atoms = chain_residues[next_res_num]['atoms']
                next_P = get_atom_coord(next_atoms, "P")
                next_O5_prime = get_atom_coord(next_atoms, "O5'")
            
            # 计算alpha: O3'(i-1)-P-O5'-C5'
            if prev_O3_prime is not None and P is not None and O5_prime is not None and C5_prime is not None:
                angle = calc_dihedral(prev_O3_prime, P, O5_prime, C5_prime)
                torsion_angles['alpha'].append(angle)
            
            # 计算beta: P-O5'-C5'-C4'
            if P is not None and O5_prime is not None and C5_prime is not None and C4_prime is not None:
                angle = calc_dihedral(P, O5_prime, C5_prime, C4_prime)
                torsion_angles['beta'].append(angle)
            
            # 计算gamma: O5'-C5'-C4'-C3'
            if O5_prime is not None and C5_prime is not None and C4_prime is not None and C3_prime is not None:
                angle = calc_dihedral(O5_prime, C5_prime, C4_prime, C3_prime)
                torsion_angles['gamma'].append(angle)
            
            # 计算delta: C5'-C4'-C3'-O3'
            if C5_prime is not None and C4_prime is not None and C3_prime is not None and O3_prime is not None:
                angle = calc_dihedral(C5_prime, C4_prime, C3_prime, O3_prime)
                torsion_angles['delta'].append(angle)
            
            # 计算epsilon: C4'-C3'-O3'-P(i+1)
            if C4_prime is not None and C3_prime is not None and O3_prime is not None and next_P is not None:
                angle = calc_dihedral(C4_prime, C3_prime, O3_prime, next_P)
                torsion_angles['epsilon'].append(angle)
            
            # 计算zeta: C3'-O3'-P(i+1)-O5'(i+1)
            if C3_prime is not None and O3_prime is not None and next_P is not None and next_O5_prime is not None:
                angle = calc_dihedral(C3_prime, O3_prime, next_P, next_O5_prime)
                torsion_angles['zeta'].append(angle)
            
            # 计算chi: 糖苷键扭转角
            # 嘌呤: O4'-C1'-N9-C4
            # 嘧啶: O4'-C1'-N1-C2
            if residue_name in ['A', 'G']:
                if O4_prime is not None and C1_prime is not None and N9 is not None and C4 is not None:
                    angle = calc_dihedral(O4_prime, C1_prime, N9, C4)
                    torsion_angles['chi'].append(angle)
            else:  # U, C
                if O4_prime is not None and C1_prime is not None and N1 is not None and C2 is not None:
                    angle = calc_dihedral(O4_prime, C1_prime, N1, C2)
                    torsion_angles['chi'].append(angle)
    
    return torsion_angles


def normalize_angle(angle: float) -> float:
    """将角度归一化到[-180, 180]范围"""
    if np.isnan(angle):
        return angle
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def calculate_angle_difference(angle1: float, angle2: float) -> float:
    """计算两个角度之间的最小差值（考虑周期性）"""
    if np.isnan(angle1) or np.isnan(angle2):
        return np.nan
    
    # 归一化角度到[-180, 180]
    angle1 = normalize_angle(angle1)
    angle2 = normalize_angle(angle2)
    
    # 计算差值
    diff = angle1 - angle2
    
    # 考虑周期性，取[-180, 180]范围内的最小差值
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    
    return abs(diff)




def process_folders(pred_folder: Path, gt_folder: Path) -> pd.DataFrame:
    """
    处理预测文件夹和ground truth文件夹中的所有PDB文件
    
    参数:
        pred_folder: 预测文件夹路径
        gt_folder: ground truth文件夹路径
        
    返回:
        DataFrame: 包含每个扭转角的MAE和RMSE统计
    """
    pred_files = sorted(pred_folder.glob("*.pdb"))
    gt_files = sorted(gt_folder.glob("*.pdb"))
    
    logger.info(f"找到 {len(pred_files)} 个预测PDB文件")
    logger.info(f"找到 {len(gt_files)} 个ground truth PDB文件")
    
    # 按文件名匹配
    pred_dict = {f.stem: f for f in pred_files}
    gt_dict = {f.stem: f for f in gt_files}
    
    common_files = set(pred_dict.keys()) & set(gt_dict.keys())
    logger.info(f"找到 {len(common_files)} 个匹配的文件对")
    
    if len(common_files) == 0:
        logger.error("没有找到匹配的文件对！")
        return pd.DataFrame()
    
    # 收集所有扭转角
    all_pred_angles = {angle: [] for angle in ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']}
    all_gt_angles = {angle: [] for angle in ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']}
    
    for filename in sorted(common_files):
        logger.info(f"处理文件: {filename}")
        
        # 解析PDB文件
        pred_atoms = parse_pdb_atoms(pred_dict[filename])
        gt_atoms = parse_pdb_atoms(gt_dict[filename])
        
        # 计算扭转角
        pred_torsions = calculate_torsion_angles(pred_atoms)
        gt_torsions = calculate_torsion_angles(gt_atoms)
        
        # 收集角度
        for angle_name in all_pred_angles.keys():
            all_pred_angles[angle_name].extend(pred_torsions[angle_name])
            all_gt_angles[angle_name].extend(gt_torsions[angle_name])
    
    # 计算每个扭转角的MAE和RMSE
    results = []
    for angle_name in ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']:
        pred_angles = all_pred_angles[angle_name]
        gt_angles = all_gt_angles[angle_name]
        
        if len(pred_angles) == 0 or len(gt_angles) == 0:
            logger.warning(f"扭转角 {angle_name} 没有有效数据 (pred: {len(pred_angles)}, gt: {len(gt_angles)})")
            results.append({
                'Torsion_Angle': angle_name,
                'Count': 0,
                'MAE': np.nan,
                'RMSE': np.nan
            })
            continue
        
        # 计算有效差值
        differences = []
        min_len = min(len(pred_angles), len(gt_angles))
        for i in range(min_len):
            diff = calculate_angle_difference(pred_angles[i], gt_angles[i])
            if not np.isnan(diff):
                differences.append(diff)
        
        if len(differences) == 0:
            logger.warning(f"扭转角 {angle_name} 没有有效的角度差值")
            results.append({
                'Torsion_Angle': angle_name,
                'Count': 0,
                'MAE': np.nan,
                'RMSE': np.nan
            })
            continue
        
        mae = np.mean(differences)
        rmse = np.sqrt(np.mean([d**2 for d in differences]))
        count = len(differences)
        
        logger.info(f"{angle_name}: Count={count}, MAE={mae:.2f}°, RMSE={rmse:.2f}°")
        
        results.append({
            'Torsion_Angle': angle_name,
            'Count': count,
            'MAE': mae,
            'RMSE': rmse
        })
    
    df = pd.DataFrame(results)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='计算RNA扭转角的MAE和RMSE',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--pred_folder',
        type=str,
        required=True,
        help='预测PDB文件文件夹路径'
    )
    parser.add_argument(
        '--gt_folder',
        type=str,
        required=True,
        help='Ground truth PDB文件文件夹路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出CSV文件路径（默认：stdout）'
    )
    
    args = parser.parse_args()
    
    pred_folder = Path(args.pred_folder)
    gt_folder = Path(args.gt_folder)
    
    if not pred_folder.exists():
        logger.error(f"预测文件夹不存在: {pred_folder}")
        sys.exit(1)
    
    if not gt_folder.exists():
        logger.error(f"Ground truth文件夹不存在: {gt_folder}")
        sys.exit(1)
    
    # 处理文件夹
    df = process_folders(pred_folder, gt_folder)
    
    if df.empty:
        logger.error("没有生成结果数据")
        sys.exit(1)
    
    # 输出结果
    if args.output:
        output_path = Path(args.output)
        df.to_csv(output_path, index=False, float_format='%.2f')
        logger.info(f"结果已保存到: {output_path}")
    else:
        print("\n" + "="*60)
        print("RNA扭转角MAE和RMSE统计结果")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)


if __name__ == '__main__':
    main()

