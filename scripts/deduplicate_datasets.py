#!/usr/bin/env python3
"""
去重脚本：去除数据集内部重复和与现有数据集的重复
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Set, Dict
import argparse
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def read_fasta_sequence(fasta_file: str) -> str:
    """读取FASTA文件中的序列"""
    try:
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
            # 跳过头部，提取序列
            sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
            return sequence
    except Exception as e:
        logger.error(f"读取文件 {fasta_file} 失败: {e}")
        return ""


def collect_sequences(seq_dir: Path) -> Dict[str, str]:
    """
    收集目录中所有序列
    返回：{basename: sequence}
    """
    sequences = {}
    for fasta_file in seq_dir.glob("*.fasta"):
        basename = fasta_file.stem
        sequence = read_fasta_sequence(str(fasta_file))
        if sequence:
            sequences[basename] = sequence
    return sequences


def create_fasta_file(sequences: Dict[str, str], output_file: str):
    """创建合并的FASTA文件用于CD-HIT"""
    with open(output_file, 'w') as f:
        for basename, sequence in sequences.items():
            f.write(f">{basename}\n")
            f.write(f"{sequence}\n")


def run_cdhit(input_fasta: str, output_fasta: str, similarity: float = 0.95, threads: int = 8):
    """
    运行CD-HIT进行序列聚类
    
    Args:
        input_fasta: 输入FASTA文件
        output_fasta: 输出FASTA文件
        similarity: 相似度阈值（0.95表示95%相似度）
        threads: 线程数
    """
    logger.info(f"🧬 运行CD-HIT进行去重（相似度阈值: {similarity*100}%）...")
    
    # 使用cd-hit-est（核酸序列）
    cmd = [
        'cd-hit-est',
        '-i', input_fasta,
        '-o', output_fasta,
        '-c', str(similarity),  # 相似度阈值
        '-n', '10',  # word size (对于95%相似度，推荐10)
        '-M', '0',  # 内存限制（0表示无限制）
        '-T', str(threads),  # 线程数
        '-d', '0',  # 描述长度（0表示完整）
        '-aS', '0.8',  # 对齐覆盖度阈值
        '-g', '1',  # 精确模式
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logger.info("✅ CD-HIT运行完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ CD-HIT运行失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False


def parse_cdhit_output(output_fasta: str) -> Set[str]:
    """
    解析CD-HIT输出，获取保留的序列ID
    
    Returns:
        保留的序列basename集合
    """
    kept_sequences = set()
    try:
        with open(output_fasta, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    # 提取序列ID
                    seq_id = line.strip()[1:].split()[0]
                    kept_sequences.add(seq_id)
    except Exception as e:
        logger.error(f"解析CD-HIT输出失败: {e}")
    
    return kept_sequences


def find_duplicates_with_reference(
    new_sequences: Dict[str, str],
    ref_sequences: Dict[str, str],
    similarity: float = 0.95,
    threads: int = 8
) -> Set[str]:
    """
    找出新数据集中与参考数据集重复的序列
    
    Args:
        new_sequences: 新数据集的序列字典
        ref_sequences: 参考数据集的序列字典
        similarity: 相似度阈值
        threads: 线程数
        
    Returns:
        新数据集中需要移除的序列basename集合
    """
    logger.info("🔍 与现有数据集比较，寻找重复...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建合并的FASTA文件
        combined_fasta = os.path.join(tmpdir, "combined.fasta")
        cdhit_output = os.path.join(tmpdir, "cdhit_output.fasta")
        
        # 写入参考数据集（先写入，这样它们会被优先保留）
        with open(combined_fasta, 'w') as f:
            for basename, sequence in ref_sequences.items():
                f.write(f">ref_{basename}\n")
                f.write(f"{sequence}\n")
            
            # 写入新数据集
            for basename, sequence in new_sequences.items():
                f.write(f">new_{basename}\n")
                f.write(f"{sequence}\n")
        
        # 运行CD-HIT
        if not run_cdhit(combined_fasta, cdhit_output, similarity, threads):
            logger.error("CD-HIT运行失败，无法完成去重")
            return set()
        
        # 解析结果
        kept_sequences = parse_cdhit_output(cdhit_output)
        
        # 找出新数据集中被移除的序列
        removed_new_sequences = set()
        for basename in new_sequences.keys():
            new_id = f"new_{basename}"
            if new_id not in kept_sequences:
                removed_new_sequences.add(basename)
        
        logger.info(f"  发现 {len(removed_new_sequences)} 个与现有数据集重复的序列")
        
        return removed_new_sequences


def deduplicate_internal(
    sequences: Dict[str, str],
    similarity: float = 0.95,
    threads: int = 8
) -> Set[str]:
    """
    对数据集内部进行去重
    
    Args:
        sequences: 序列字典
        similarity: 相似度阈值
        threads: 线程数
        
    Returns:
        保留的序列basename集合
    """
    logger.info("🔍 对数据集内部进行去重...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_fasta = os.path.join(tmpdir, "input.fasta")
        output_fasta = os.path.join(tmpdir, "output.fasta")
        
        # 创建输入FASTA文件
        create_fasta_file(sequences, input_fasta)
        
        # 运行CD-HIT
        if not run_cdhit(input_fasta, output_fasta, similarity, threads):
            logger.error("CD-HIT运行失败，保留所有序列")
            return set(sequences.keys())
        
        # 解析结果
        kept_sequences = parse_cdhit_output(output_fasta)
        
        removed_count = len(sequences) - len(kept_sequences)
        logger.info(f"  内部去重：移除 {removed_count} 个重复序列")
        
        return kept_sequences


def main():
    parser = argparse.ArgumentParser(description='去除数据集内部重复和与现有数据集的重复')
    parser.add_argument('--new_data', type=str, required=True,
                       help='新数据集目录（包含pdb/和seq/子目录）')
    parser.add_argument('--reference_data', type=str, default=None,
                       help='参考数据集目录（包含pdb/和seq/子目录），如果提供则去除与之重复的序列')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录（去重后的数据）')
    parser.add_argument('--similarity', type=float, default=0.95,
                       help='相似度阈值 (默认: 0.95, 即95%%)')
    parser.add_argument('--threads', type=int, default=8,
                       help='CD-HIT线程数 (默认: 8)')
    
    args = parser.parse_args()
    
    new_data_dir = Path(args.new_data)
    output_dir = Path(args.output)
    
    logger.info("=" * 60)
    logger.info("🧬 RNA数据集去重工具")
    logger.info("=" * 60)
    logger.info(f"📁 新数据集: {new_data_dir}")
    if args.reference_data:
        logger.info(f"📁 参考数据集: {args.reference_data}")
    logger.info(f"📁 输出目录: {output_dir}")
    logger.info(f"🔬 相似度阈值: {args.similarity*100}%")
    logger.info(f"🧵 线程数: {args.threads}")
    logger.info("=" * 60)
    
    # 检查目录
    new_seq_dir = new_data_dir / "seq"
    new_pdb_dir = new_data_dir / "pdb"
    
    if not new_seq_dir.exists():
        logger.error(f"❌ 序列目录不存在: {new_seq_dir}")
        return
    
    if not new_pdb_dir.exists():
        logger.error(f"❌ PDB目录不存在: {new_pdb_dir}")
        return
    
    # 收集新数据集的序列
    logger.info("\n📊 第一步: 收集新数据集的序列...")
    new_sequences = collect_sequences(new_seq_dir)
    logger.info(f"  找到 {len(new_sequences)} 个序列")
    
    # 内部去重
    logger.info("\n📊 第二步: 对新数据集内部进行去重...")
    kept_sequences = deduplicate_internal(new_sequences, args.similarity, args.threads)
    
    # 如果提供了参考数据集，则去除与之重复的序列
    if args.reference_data:
        ref_data_dir = Path(args.reference_data)
        ref_seq_dir = ref_data_dir / "seq"
        
        if not ref_seq_dir.exists():
            logger.warning(f"⚠️  参考数据集序列目录不存在: {ref_seq_dir}")
            logger.warning("⚠️  跳过与参考数据集的去重")
        else:
            logger.info("\n📊 第三步: 与参考数据集比较...")
            ref_sequences = collect_sequences(ref_seq_dir)
            logger.info(f"  参考数据集有 {len(ref_sequences)} 个序列")
            
            # 只保留kept_sequences中的序列进行比较
            new_sequences_to_check = {k: v for k, v in new_sequences.items() if k in kept_sequences}
            
            removed_duplicates = find_duplicates_with_reference(
                new_sequences_to_check,
                ref_sequences,
                args.similarity,
                args.threads
            )
            
            # 从kept_sequences中移除与参考数据集重复的
            kept_sequences = kept_sequences - removed_duplicates
    
    # 复制保留的文件
    logger.info("\n📊 最后一步: 复制保留的文件...")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdb_dir = output_dir / "pdb"
    output_seq_dir = output_dir / "seq"
    output_pdb_dir.mkdir(exist_ok=True)
    output_seq_dir.mkdir(exist_ok=True)
    
    copied_count = 0
    for basename in kept_sequences:
        # 复制PDB文件
        src_pdb = new_pdb_dir / f"{basename}.pdb"
        dst_pdb = output_pdb_dir / f"{basename}.pdb"
        if src_pdb.exists():
            shutil.copy2(src_pdb, dst_pdb)
        
        # 复制序列文件
        src_seq = new_seq_dir / f"{basename}.fasta"
        dst_seq = output_seq_dir / f"{basename}.fasta"
        if src_seq.exists():
            shutil.copy2(src_seq, dst_seq)
        
        copied_count += 1
        if copied_count % 100 == 0:
            logger.info(f"  已复制 {copied_count}/{len(kept_sequences)} 个文件...")
    
    logger.info(f"  完成！共复制 {copied_count} 个样本")
    
    # 生成汇总报告
    logger.info("\n" + "=" * 60)
    logger.info("✅ 去重完成!")
    logger.info(f"📊 原始数据: {len(new_sequences)} 个")
    logger.info(f"📊 去重后: {len(kept_sequences)} 个")
    logger.info(f"📊 移除: {len(new_sequences) - len(kept_sequences)} 个")
    logger.info(f"📁 输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

