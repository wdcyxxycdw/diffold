#!/usr/bin/env python3
"""
基于cd-hit聚类的十折交叉验证数据划分脚本

该脚本会：
1. 收集所有RNA序列数据
2. 使用cd-hit进行聚类（阈值80%）
3. 基于聚类结果进行十折划分
4. 生成训练/验证集划分文件
"""

import os
import subprocess
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import random
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CDHitCVSplitter:
    def __init__(self, data_dir: str, output_dir: str, cdhit_threshold: float = 0.8):
        """
        初始化CD-hit交叉验证划分器
        
        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
            cdhit_threshold: cd-hit相似度阈值
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.cdhit_threshold = cdhit_threshold
        self.sequences_dir = self.data_dir / "sequences"
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "clusters").mkdir(exist_ok=True)
        
        # 检查cd-hit是否可用
        self._check_cdhit()
    
    def _check_cdhit(self):
        """检查cd-hit是否可用"""
        try:
            result = subprocess.run(['cd-hit', '-h'], 
                                  capture_output=True, text=True, timeout=10)
            # cd-hit即使显示帮助信息也可能返回非零退出码，所以检查输出内容
            if "CD-HIT version" in result.stdout:
                logger.info("✓ cd-hit可用")
            else:
                raise FileNotFoundError("cd-hit not found")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.error("❌ cd-hit不可用，请安装cd-hit")
            logger.info("安装方法：")
            logger.info("  conda install -c bioconda cd-hit")
            logger.info("  或")
            logger.info("  sudo apt-get install cd-hit")
            raise
    
    def collect_sequences(self) -> str:
        """
        收集所有序列到一个FASTA文件
        
        Returns:
            FASTA文件路径
        """
        logger.info("收集所有序列...")
        
        all_sequences_file = self.output_dir / "all_sequences.fasta"
        
        with open(all_sequences_file, 'w') as f:
            seq_count = 0
            for fasta_file in self.sequences_dir.glob("*.fasta"):
                try:
                    with open(fasta_file, 'r') as seq_file:
                        content = seq_file.read().strip()
                        if content:
                            # 提取序列ID（文件名去掉.fasta）
                            seq_id = fasta_file.stem
                            # 提取序列内容（第二行）
                            lines = content.split('\n')
                            if len(lines) >= 2:
                                sequence = lines[1]
                                f.write(f">{seq_id}\n{sequence}\n")
                                seq_count += 1
                except Exception as e:
                    logger.warning(f"跳过文件 {fasta_file}: {e}")
        
        logger.info(f"✓ 收集了 {seq_count} 个序列到 {all_sequences_file}")
        return str(all_sequences_file)
    
    def run_cdhit(self, input_fasta: str) -> str:
        """
        运行cd-hit聚类
        
        Args:
            input_fasta: 输入FASTA文件路径
            
        Returns:
            聚类结果文件路径
        """
        logger.info(f"运行cd-hit聚类（阈值: {self.cdhit_threshold}）...")
        
        output_clstr = self.output_dir / "clusters" / "sequences.clstr"
        output_fasta = self.output_dir / "clusters" / "sequences.fasta"
        
        # cd-hit命令
        cmd = [
            'cd-hit',
            '-i', input_fasta,
            '-o', str(output_fasta),
            '-c', str(self.cdhit_threshold),
            '-n', '5',  # word length
            '-M', '16000',  # memory limit
            '-d', '0'  # full header in output
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"cd-hit运行失败: {result.stderr}")
                raise RuntimeError("cd-hit聚类失败")
            
            logger.info("✓ cd-hit聚类完成")
            logger.info(f"聚类结果: {output_fasta}")
            logger.info(f"聚类信息: {output_fasta}.clstr")
            
            return str(output_fasta)
            
        except subprocess.TimeoutExpired:
            logger.error("cd-hit运行超时")
            raise
        except Exception as e:
            logger.error(f"cd-hit运行出错: {e}")
            raise
    
    def parse_cluster_file(self, cluster_file: str) -> Dict[str, List[str]]:
        """
        解析cd-hit聚类文件
        
        Args:
            cluster_file: 聚类文件路径
            
        Returns:
            聚类结果字典 {cluster_id: [sequence_ids]}
        """
        logger.info("解析聚类文件...")
        
        clusters = {}
        current_cluster = None
        
        with open(cluster_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>Cluster'):
                    current_cluster = line.split()[1]
                    clusters[current_cluster] = []
                elif line and current_cluster:
                    # 解析序列信息
                    # 格式: 0	123aa, >seq_id... *
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        seq_info = parts[1]
                        if '>' in seq_info:
                            seq_id = seq_info.split('>')[1].split('...')[0]
                            clusters[current_cluster].append(seq_id)
        
        # 统计聚类信息
        cluster_count = len(clusters)
        total_sequences = sum(len(seqs) for seqs in clusters.values())
        avg_cluster_size = total_sequences / cluster_count if cluster_count > 0 else 0
        
        logger.info(f"✓ 解析完成:")
        logger.info(f"  聚类数量: {cluster_count}")
        logger.info(f"  总序列数: {total_sequences}")
        logger.info(f"  平均聚类大小: {avg_cluster_size:.2f}")
        
        return clusters
    
    def create_cv_splits(self, clusters: Dict[str, List[str]], n_folds: int = 10) -> List[Tuple[List[str], List[str]]]:
        """
        创建交叉验证划分
        
        Args:
            clusters: 聚类结果
            n_folds: 折数
            
        Returns:
            划分结果列表 [(train_clusters, val_clusters), ...]
        """
        logger.info(f"创建 {n_folds} 折交叉验证划分...")
        
        cluster_ids = list(clusters.keys())
        random.shuffle(cluster_ids)  # 随机打乱聚类顺序
        
        # 计算每折的聚类数量
        clusters_per_fold = len(cluster_ids) // n_folds
        remainder = len(cluster_ids) % n_folds
        
        splits = []
        start_idx = 0
        
        for fold in range(n_folds):
            # 计算当前折的聚类数量
            fold_size = clusters_per_fold + (1 if fold < remainder else 0)
            end_idx = start_idx + fold_size
            
            # 当前折的验证集聚类
            val_clusters = cluster_ids[start_idx:end_idx]
            # 其余作为训练集
            train_clusters = cluster_ids[:start_idx] + cluster_ids[end_idx:]
            
            splits.append((train_clusters, val_clusters))
            start_idx = end_idx
            
            logger.info(f"  折 {fold + 1}: 训练集 {len(train_clusters)} 聚类, 验证集 {len(val_clusters)} 聚类")
        
        return splits
    
    def save_splits(self, splits: List[Tuple[List[str], List[str]]], 
                   clusters: Dict[str, List[str]]) -> None:
        """
        保存划分结果（参考processed_data/list格式）
        
        Args:
            splits: 划分结果
            clusters: 聚类结果
        """
        logger.info("保存划分结果...")
        
        # 创建list目录（参考processed_data/list格式）
        list_dir = self.output_dir / "list"
        list_dir.mkdir(exist_ok=True)
        
        for fold, (train_clusters, val_clusters) in enumerate(splits):
            # 获取训练集和验证集的序列ID
            train_sequences = []
            val_sequences = []
            
            for cluster_id in train_clusters:
                train_sequences.extend(clusters[cluster_id])
            
            for cluster_id in val_clusters:
                val_sequences.extend(clusters[cluster_id])
            
            # 保存训练集（格式：fold-{fold}_train_ids）
            train_file = list_dir / f"fold-{fold}_train_ids"
            with open(train_file, 'w') as f:
                for seq_id in train_sequences:
                    f.write(f"{seq_id}\n")
            
            # 保存验证集（格式：valid_fold-{fold}）
            val_file = list_dir / f"valid_fold-{fold}"
            with open(val_file, 'w') as f:
                for seq_id in val_sequences:
                    f.write(f"{seq_id}\n")
            
            logger.info(f"  折 {fold}: 训练集 {len(train_sequences)} 序列, 验证集 {len(val_sequences)} 序列")
        
        # 保存总体统计信息
        stats_file = self.output_dir / "cv_stats.txt"
        with open(stats_file, 'w') as f:
            f.write(f"CD-hit聚类交叉验证统计\n")
            f.write(f"相似度阈值: {self.cdhit_threshold}\n")
            f.write(f"总聚类数: {len(clusters)}\n")
            f.write(f"总序列数: {sum(len(seqs) for seqs in clusters.values())}\n")
            f.write(f"折数: {len(splits)}\n\n")
            
            for fold, (train_clusters, val_clusters) in enumerate(splits):
                train_sequences = sum(len(clusters[c]) for c in train_clusters)
                val_sequences = sum(len(clusters[c]) for c in val_clusters)
                f.write(f"折 {fold}:\n")
                f.write(f"  训练集: {len(train_clusters)} 聚类, {train_sequences} 序列\n")
                f.write(f"  验证集: {len(val_clusters)} 聚类, {val_sequences} 序列\n\n")
        
        logger.info(f"✓ 划分结果已保存到 {list_dir}")
        logger.info(f"✓ 统计信息已保存到 {stats_file}")
    
    def run(self, n_folds: int = 10):
        """
        运行完整的交叉验证划分流程
        
        Args:
            n_folds: 折数
        """
        logger.info("开始CD-hit聚类交叉验证划分...")
        
        try:
            # 1. 收集序列
            all_sequences_file = self.collect_sequences()
            
            # 2. 运行cd-hit聚类
            cluster_fasta = self.run_cdhit(all_sequences_file)
            cluster_file = cluster_fasta + ".clstr"
            
            # 3. 解析聚类结果
            clusters = self.parse_cluster_file(cluster_file)
            
            # 4. 创建交叉验证划分
            splits = self.create_cv_splits(clusters, n_folds)
            
            # 5. 保存划分结果
            self.save_splits(splits, clusters)
            
            logger.info("✓ 交叉验证划分完成!")
            
        except Exception as e:
            logger.error(f"❌ 划分失败: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="基于cd-hit聚类的十折交叉验证数据划分")
    parser.add_argument("--data_dir", default="processed_data", 
                       help="数据目录路径 (默认: processed_data)")
    parser.add_argument("--output_dir", default="processed_data/cv_splits_cdhit", 
                       help="输出目录路径 (默认: processed_data/cv_splits_cdhit)")
    parser.add_argument("--threshold", type=float, default=0.8, 
                       help="cd-hit相似度阈值 (默认: 0.8)")
    parser.add_argument("--folds", type=int, default=10, 
                       help="交叉验证折数 (默认: 10)")
    parser.add_argument("--seed", type=int, default=412907, 
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建划分器并运行
    splitter = CDHitCVSplitter(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cdhit_threshold=args.threshold
    )
    
    splitter.run(n_folds=args.folds)

if __name__ == "__main__":
    main() 