#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 benchmark_data 和 fine_tuning_data 之间是否有序列重复（数据泄露）
"""

import os
import glob
from pathlib import Path
from collections import defaultdict


def read_sequences_from_fasta(fasta_file):
    """从FASTA文件读取所有序列"""
    sequences = []
    current_seq = []
    current_header = None
    
    try:
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_seq and current_header:
                        sequences.append({
                            'header': current_header,
                            'sequence': ''.join(current_seq).upper()
                        })
                    current_header = line[1:]  # 去掉 '>'
                    current_seq = []
                else:
                    current_seq.append(line)
            
            # 添加最后一个序列
            if current_seq and current_header:
                sequences.append({
                    'header': current_header,
                    'sequence': ''.join(current_seq).upper()
                })
    except Exception as e:
        print(f"  警告: 读取文件 {fasta_file} 时出错: {e}")
        return []
    
    return sequences


def load_sequences_from_directory(directory):
    """从目录中加载所有FASTA文件的序列"""
    seq_dict = {}  # sequence -> list of (file, header)
    file_count = 0
    
    # 查找所有FASTA文件
    fasta_patterns = ['*.fasta', '*.fa', '*.fna']
    fasta_files = []
    for pattern in fasta_patterns:
        fasta_files.extend(glob.glob(os.path.join(directory, pattern)))
    
    print(f"在 {directory} 中找到 {len(fasta_files)} 个序列文件")
    
    for fasta_file in fasta_files:
        file_count += 1
        filename = os.path.basename(fasta_file)
        sequences = read_sequences_from_fasta(fasta_file)
        
        for seq_info in sequences:
            seq = seq_info['sequence']
            if seq not in seq_dict:
                seq_dict[seq] = []
            seq_dict[seq].append({
                'file': filename,
                'header': seq_info['header'],
                'length': len(seq)
            })
    
    print(f"  共加载 {len(seq_dict)} 个唯一序列")
    return seq_dict


def check_overlap(benchmark_seqs, training_seqs):
    """检查两个序列集合之间的重叠"""
    overlaps = []
    
    for seq, benchmark_sources in benchmark_seqs.items():
        if seq in training_seqs:
            training_sources = training_seqs[seq]
            overlaps.append({
                'sequence': seq,
                'length': len(seq),
                'benchmark_sources': benchmark_sources,
                'training_sources': training_sources
            })
    
    return overlaps


def main():
    print("=" * 80)
    print("数据泄露检查工具")
    print("=" * 80)
    print()
    
    # 定义路径
    benchmark_dir = "benchmark_data/casp15/sequences"
    training_dir = "fine_tuning_data/sequences"
    processed_dir = "processed_data/seq"
    
    # 检查目录是否存在
    if not os.path.exists(benchmark_dir):
        print(f"错误: benchmark目录不存在: {benchmark_dir}")
        return 1
    
    if not os.path.exists(training_dir):
        print(f"错误: training目录不存在: {training_dir}")
        return 1
    
    if not os.path.exists(processed_dir):
        print(f"错误: processed目录不存在: {processed_dir}")
        return 1
    
    print("步骤 1: 加载 benchmark 数据集")
    print("-" * 80)
    benchmark_seqs = load_sequences_from_directory(benchmark_dir)
    print()
    
    print("步骤 2: 加载 fine-tuning 数据集")
    print("-" * 80)
    training_seqs = load_sequences_from_directory(training_dir)
    print()
    
    print("步骤 3: 加载 processed 数据集")
    print("-" * 80)
    processed_seqs = load_sequences_from_directory(processed_dir)
    print()
    
    print("步骤 4: 检查与 fine-tuning 数据集的重叠")
    print("-" * 80)
    overlaps_training = check_overlap(benchmark_seqs, training_seqs)
    
    print("步骤 5: 检查与 processed 数据集的重叠")
    print("-" * 80)
    overlaps_processed = check_overlap(benchmark_seqs, processed_seqs)
    
    overlaps = overlaps_training + overlaps_processed
    
    if not overlaps_training and not overlaps_processed:
        print("✓ 未发现数据泄露！")
        print(f"  benchmark 数据集的 {len(benchmark_seqs)} 个序列均不在 fine-tuning 或 processed 数据集中")
    else:
        print(f"✗ 发现数据泄露！")
        if overlaps_training:
            print(f"  - Fine-tuning 数据集中有 {len(overlaps_training)} 个重复序列")
        if overlaps_processed:
            print(f"  - Processed 数据集中有 {len(overlaps_processed)} 个重复序列")
        print()
        
        if overlaps_training:
            print("与 Fine-tuning 数据集的重复:")
            print("=" * 80)
            for i, overlap in enumerate(overlaps_training, 1):
                print(f"\n重复序列 #{i}:")
                print(f"  序列长度: {overlap['length']} 个核苷酸")
                print(f"  序列: {overlap['sequence'][:60]}{'...' if len(overlap['sequence']) > 60 else ''}")
                print()
                print("  在 benchmark 数据集中:")
                for src in overlap['benchmark_sources']:
                    print(f"    - {src['file']} (header: {src['header']})")
                print()
                print("  在 fine-tuning 数据集中:")
                for src in overlap['training_sources']:
                    print(f"    - {src['file']} (header: {src['header']})")
                print("-" * 80)
        
        if overlaps_processed:
            print("\n与 Processed 数据集的重复:")
            print("=" * 80)
            for i, overlap in enumerate(overlaps_processed, 1):
                print(f"\n重复序列 #{i}:")
                print(f"  序列长度: {overlap['length']} 个核苷酸")
                print(f"  序列: {overlap['sequence'][:60]}{'...' if len(overlap['sequence']) > 60 else ''}")
                print()
                print("  在 benchmark 数据集中:")
                for src in overlap['benchmark_sources']:
                    print(f"    - {src['file']} (header: {src['header']})")
                print()
                print("  在 processed 数据集中:")
                for src in overlap['training_sources']:
                    print(f"    - {src['file']} (header: {src['header']})")
                print("-" * 80)
    
    print()
    print("=" * 80)
    print("检查完成")
    print("=" * 80)
    print(f"Benchmark 数据集: {len(benchmark_seqs)} 个唯一序列")
    print(f"Fine-tuning 数据集: {len(training_seqs)} 个唯一序列")
    print(f"Processed 数据集: {len(processed_seqs)} 个唯一序列")
    print(f"与 Fine-tuning 重复: {len(overlaps_training)} 个")
    print(f"与 Processed 重复: {len(overlaps_processed)} 个")
    print("=" * 80)
    
    return 0 if (len(overlaps_training) == 0 and len(overlaps_processed) == 0) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

