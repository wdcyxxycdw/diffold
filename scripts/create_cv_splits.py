#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNA 3D数据集交叉验证分割脚本

这个脚本会：
1. 扫描processed_data目录中的所有有效样本（同时有PDB和序列文件）
2. 生成10折交叉验证分割
3. 保存分割文件到processed_data/list目录
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CrossValidationSplitter:
    """交叉验证分割器"""
    
    def __init__(self, data_dir: str, n_folds: int = 10, random_seed: int = 42):
        """
        初始化交叉验证分割器
        
        Args:
            data_dir: 数据目录路径 (processed_data)
            n_folds: 交叉验证折数
            random_seed: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.n_folds = n_folds
        self.random_seed = random_seed
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 数据目录
        self.pdb_dir = self.data_dir / "pdb"
        self.seq_dir = self.data_dir / "sequences"
        self.list_dir = self.data_dir / "list"
        
        # 验证目录
        self._validate_directories()
        
        # 创建list目录
        self.list_dir.mkdir(exist_ok=True)
    
    def _validate_directories(self):
        """验证必要目录是否存在"""
        if not self.pdb_dir.exists():
            raise FileNotFoundError(f"PDB目录不存在: {self.pdb_dir}")
        if not self.seq_dir.exists():
            raise FileNotFoundError(f"序列目录不存在: {self.seq_dir}")
        
        logger.info(f"数据目录: {self.data_dir}")
        logger.info(f"PDB目录: {self.pdb_dir}")
        logger.info(f"序列目录: {self.seq_dir}")
    
    def scan_samples(self) -> List[str]:
        """
        扫描数据目录，找到所有有效的样本ID
        
        Returns:
            有效样本ID列表
        """
        logger.info("开始扫描数据目录...")
        
        # 获取所有PDB文件
        pdb_files = list(self.pdb_dir.glob("*.pdb"))
        logger.info(f"找到 {len(pdb_files)} 个PDB文件")
        
        valid_samples = []
        missing_seq = 0
        
        for pdb_file in pdb_files:
            sample_id = pdb_file.stem  # 获取文件名（不含扩展名）
            
            # 检查对应的序列文件是否存在
            possible_seq_files = [
                self.seq_dir / f"{sample_id}.fasta",
                self.seq_dir / f"{sample_id}.fa"
            ]
            
            has_seq_file = any(seq_file.exists() for seq_file in possible_seq_files)
            
            if has_seq_file:
                valid_samples.append(sample_id)
            else:
                missing_seq += 1
                if missing_seq <= 10:  # 只显示前10个缺失的
                    logger.warning(f"缺少序列文件: {sample_id}")
                elif missing_seq == 11:
                    logger.warning("...")
        
        if missing_seq > 0:
            logger.warning(f"总共 {missing_seq} 个样本缺少序列文件")
        
        logger.info(f"扫描完成，找到 {len(valid_samples)} 个有效样本")
        return valid_samples
    
    def create_stratified_splits(self, sample_ids: List[str]) -> Dict[int, Dict[str, List[str]]]:
        """
        创建分层交叉验证分割
        
        Args:
            sample_ids: 样本ID列表
            
        Returns:
            分割字典 {fold: {'train': [...], 'valid': [...]}}
        """
        logger.info(f"开始创建 {self.n_folds} 折交叉验证分割...")
        
        # 随机打乱样本
        shuffled_samples = sample_ids.copy()
        random.shuffle(shuffled_samples)
        
        # 计算每折的大小
        total_samples = len(shuffled_samples)
        fold_sizes = [total_samples // self.n_folds] * self.n_folds
        
        # 将剩余样本分配到前几个fold
        remainder = total_samples % self.n_folds
        for i in range(remainder):
            fold_sizes[i] += 1
        
        # 创建分割
        splits = {}
        start_idx = 0
        
        for fold in range(self.n_folds):
            end_idx = start_idx + fold_sizes[fold]
            
            # 当前fold作为验证集
            valid_samples = shuffled_samples[start_idx:end_idx]
            
            # 其他fold作为训练集
            train_samples = shuffled_samples[:start_idx] + shuffled_samples[end_idx:]
            
            splits[fold] = {
                'train': train_samples,
                'valid': valid_samples
            }
            
            logger.info(f"Fold {fold}: 训练样本 {len(train_samples)}, 验证样本 {len(valid_samples)}")
            
            start_idx = end_idx
        
        return splits
    
    def save_splits(self, splits: Dict[int, Dict[str, List[str]]]):
        """
        保存分割到文件
        
        Args:
            splits: 分割字典
        """
        logger.info(f"保存分割文件到: {self.list_dir}")
        
        for fold in range(self.n_folds):
            # 保存训练集
            train_file = self.list_dir / f"fold-{fold}_train_ids"
            with open(train_file, 'w') as f:
                for sample_id in splits[fold]['train']:
                    f.write(f"{sample_id}\n")
            logger.info(f"保存 {train_file}: {len(splits[fold]['train'])} 个训练样本")
            
            # 保存验证集
            valid_file = self.list_dir / f"valid_fold-{fold}"
            with open(valid_file, 'w') as f:
                for sample_id in splits[fold]['valid']:
                    f.write(f"{sample_id}\n")
            logger.info(f"保存 {valid_file}: {len(splits[fold]['valid'])} 个验证样本")
    
    def create_splits(self) -> Dict[int, Dict[str, List[str]]]:
        """
        完整的分割创建流程
        
        Returns:
            分割字典
        """
        # 1. 扫描样本
        sample_ids = self.scan_samples()
        
        if len(sample_ids) == 0:
            raise ValueError("没有找到有效的样本！")
        
        # 2. 创建分割
        splits = self.create_stratified_splits(sample_ids)
        
        # 3. 保存分割
        self.save_splits(splits)
        
        return splits
    
    def validate_splits(self, splits: Dict[int, Dict[str, List[str]]]) -> bool:
        """
        验证分割的正确性
        
        Args:
            splits: 分割字典
            
        Returns:
            是否验证通过
        """
        logger.info("验证分割正确性...")
        
        all_train_samples = set()
        all_valid_samples = set()
        
        for fold in range(self.n_folds):
            train_samples = set(splits[fold]['train'])
            valid_samples = set(splits[fold]['valid'])
            
            # 检查训练集和验证集是否有重叠
            overlap = train_samples & valid_samples
            if overlap:
                logger.error(f"Fold {fold}: 训练集和验证集有重叠! {len(overlap)} 个样本")
                return False
            
            all_train_samples.update(train_samples)
            all_valid_samples.update(valid_samples)
        
        # 检查所有样本是否都被包含
        total_samples = all_train_samples | all_valid_samples
        original_samples = set(self.scan_samples())
        
        if total_samples != original_samples:
            missing = original_samples - total_samples
            extra = total_samples - original_samples
            if missing:
                logger.error(f"缺少样本: {len(missing)} 个")
            if extra:
                logger.error(f"多余样本: {len(extra)} 个")
            return False
        
        logger.info("✅ 分割验证通过!")
        return True


def print_statistics(splits: Dict[int, Dict[str, List[str]]]):
    """打印分割统计信息"""
    print("\n" + "="*50)
    print("交叉验证分割统计")
    print("="*50)
    
    total_train = 0
    total_valid = 0
    
    for fold in sorted(splits.keys()):
        train_count = len(splits[fold]['train'])
        valid_count = len(splits[fold]['valid'])
        total_train += train_count
        total_valid += valid_count
        
        print(f"Fold {fold:2d}: 训练 {train_count:4d}, 验证 {valid_count:3d}")
    
    total_samples = total_train + total_valid
    avg_train = total_train // len(splits)
    avg_valid = total_valid // len(splits)
    
    print("-" * 50)
    print(f"总样本数: {total_samples}")
    print(f"平均每折: 训练 {avg_train}, 验证 {avg_valid}")
    print(f"训练/验证比例: {total_train/total_valid:.2f}:1")
    print("="*50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="为RNA 3D数据集创建交叉验证分割")
    parser.add_argument("--data_dir", type=str, default="processed_data",
                       help="数据目录路径 (默认: processed_data)")
    parser.add_argument("--n_folds", type=int, default=10,
                       help="交叉验证折数 (默认: 10)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    parser.add_argument("--backup_old", action="store_true",
                       help="是否备份旧的分割文件")
    
    args = parser.parse_args()
    
    try:
        # 备份旧的分割文件
        if args.backup_old:
            backup_dir = Path(args.data_dir) / "list_backup"
            list_dir = Path(args.data_dir) / "list"
            if list_dir.exists():
                import shutil
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.copytree(list_dir, backup_dir)
                logger.info(f"备份旧分割文件到: {backup_dir}")
        
        # 创建分割器
        splitter = CrossValidationSplitter(
            data_dir=args.data_dir,
            n_folds=args.n_folds,
            random_seed=args.random_seed
        )
        
        # 创建分割
        splits = splitter.create_splits()
        
        # 验证分割
        if splitter.validate_splits(splits):
            print_statistics(splits)
            logger.info("✅ 交叉验证分割创建成功!")
        else:
            logger.error("❌ 分割验证失败!")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 创建分割失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 