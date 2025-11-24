#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版数据泄露检查工具
功能：
1. 检查完全重复的序列
2. 检查子序列包含关系
3. 检查高相似度序列
"""

import os
import glob
import argparse
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import json


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


def load_sequences_from_directory(directory, recursive=False):
    """从目录中加载所有FASTA文件的序列"""
    seq_list = []  # 保持顺序和完整信息
    file_count = 0
    
    # 查找所有FASTA文件
    fasta_patterns = ['*.fasta', '*.fa', '*.fna']
    fasta_files = []
    
    if recursive:
        for pattern in fasta_patterns:
            fasta_files.extend(glob.glob(os.path.join(directory, '**', pattern), recursive=True))
    else:
        for pattern in fasta_patterns:
            fasta_files.extend(glob.glob(os.path.join(directory, pattern)))
    
    print(f"在 {directory} 中找到 {len(fasta_files)} 个序列文件")
    
    for fasta_file in fasta_files:
        file_count += 1
        filename = os.path.relpath(fasta_file, directory)
        sequences = read_sequences_from_fasta(fasta_file)
        
        for seq_info in sequences:
            seq_list.append({
                'file': filename,
                'header': seq_info['header'],
                'sequence': seq_info['sequence'],
                'length': len(seq_info['sequence'])
            })
    
    print(f"  共加载 {len(seq_list)} 个序列")
    return seq_list


def check_exact_duplicates(target_seqs, training_seqs):
    """检查1: 完全重复的序列"""
    print("\n" + "=" * 80)
    print("检查 1: 完全重复的序列")
    print("=" * 80)
    
    # 建立训练集序列索引
    training_seq_dict = defaultdict(list)
    for item in training_seqs:
        training_seq_dict[item['sequence']].append(item)
    
    duplicates = []
    for target in target_seqs:
        if target['sequence'] in training_seq_dict:
            duplicates.append({
                'target': target,
                'training_matches': training_seq_dict[target['sequence']]
            })
    
    if duplicates:
        print(f"⚠️  发现 {len(duplicates)} 个完全重复的序列！")
        for i, dup in enumerate(duplicates, 1):
            print(f"\n重复 #{i}:")
            print(f"  目标数据: {dup['target']['file']} - {dup['target']['header']}")
            print(f"  序列长度: {dup['target']['length']}")
            print(f"  序列: {dup['target']['sequence'][:80]}{'...' if len(dup['target']['sequence']) > 80 else ''}")
            print(f"  在训练集中的匹配:")
            for match in dup['training_matches']:
                print(f"    - {match['file']} - {match['header']}")
    else:
        print("✓ 未发现完全重复的序列")
    
    return duplicates


def check_substring_containment(target_seqs, training_seqs, min_length=10):
    """检查2: 子序列包含关系"""
    print("\n" + "=" * 80)
    print("检查 2: 子序列包含关系")
    print("=" * 80)
    print(f"(最小子序列长度: {min_length})")
    
    containments = []
    
    # 检查目标序列是否是训练集序列的子序列
    for i, target in enumerate(target_seqs):
        if len(target['sequence']) < min_length:
            continue
        
        for training in training_seqs:
            # 检查 target 是否是 training 的子序列
            if target['sequence'] in training['sequence'] and target['sequence'] != training['sequence']:
                containments.append({
                    'type': 'target_in_training',
                    'target': target,
                    'training': training,
                    'target_length': len(target['sequence']),
                    'training_length': len(training['sequence']),
                    'coverage': len(target['sequence']) / len(training['sequence'])
                })
            # 检查 training 是否是 target 的子序列
            elif training['sequence'] in target['sequence'] and target['sequence'] != training['sequence']:
                if len(training['sequence']) >= min_length:
                    containments.append({
                        'type': 'training_in_target',
                        'target': target,
                        'training': training,
                        'target_length': len(target['sequence']),
                        'training_length': len(training['sequence']),
                        'coverage': len(training['sequence']) / len(target['sequence'])
                    })
        
        # 进度提示
        if (i + 1) % 10 == 0:
            print(f"  已检查 {i + 1}/{len(target_seqs)} 个目标序列...", end='\r')
    
    print(" " * 80, end='\r')  # 清除进度提示
    
    if containments:
        print(f"⚠️  发现 {len(containments)} 个子序列包含关系！")
        
        # 按类型分组
        target_in_training = [c for c in containments if c['type'] == 'target_in_training']
        training_in_target = [c for c in containments if c['type'] == 'training_in_target']
        
        if target_in_training:
            print(f"\n目标序列是训练集序列的子序列: {len(target_in_training)} 个")
            for i, cont in enumerate(target_in_training[:5], 1):  # 只显示前5个
                print(f"\n  #{i}:")
                print(f"    目标: {cont['target']['file']} (长度: {cont['target_length']})")
                print(f"    包含在训练集: {cont['training']['file']} (长度: {cont['training_length']})")
                print(f"    覆盖率: {cont['coverage']*100:.1f}%")
            if len(target_in_training) > 5:
                print(f"\n    ... 还有 {len(target_in_training) - 5} 个")
        
        if training_in_target:
            print(f"\n训练集序列是目标序列的子序列: {len(training_in_target)} 个")
            for i, cont in enumerate(training_in_target[:5], 1):
                print(f"\n  #{i}:")
                print(f"    目标: {cont['target']['file']} (长度: {cont['target_length']})")
                print(f"    包含训练集: {cont['training']['file']} (长度: {cont['training_length']})")
                print(f"    覆盖率: {cont['coverage']*100:.1f}%")
            if len(training_in_target) > 5:
                print(f"\n    ... 还有 {len(training_in_target) - 5} 个")
    else:
        print("✓ 未发现子序列包含关系")
    
    return containments


def calculate_sequence_similarity(seq1, seq2):
    """计算两个序列的相似度（使用 SequenceMatcher）"""
    return SequenceMatcher(None, seq1, seq2).ratio()


def calculate_kmer_similarity(seq1, seq2, k=3):
    """计算基于 k-mer 的序列相似度（Jaccard 相似度）"""
    def get_kmers(seq, k):
        return set(seq[i:i+k] for i in range(len(seq) - k + 1))
    
    kmers1 = get_kmers(seq1, k)
    kmers2 = get_kmers(seq2, k)
    
    if not kmers1 or not kmers2:
        return 0.0
    
    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    
    return intersection / union if union > 0 else 0.0


def calculate_cdhit_similarity(seq1, seq2):
    """
    计算 CD-HIT 风格的序列相似度
    
    CD-HIT 相似度定义：
    相似度 = 匹配的字符数 / min(len(seq1), len(seq2))
    
    使用简单的逐字符比较（不考虑gaps）
    对于更精确的比对，可以使用 Biopython 的 pairwise2
    """
    if not seq1 or not seq2:
        return 0.0
    
    # 计算匹配的字符数
    min_len = min(len(seq1), len(seq2))
    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
    
    # CD-HIT 相似度：匹配数 / 较短序列长度
    return matches / min_len if min_len > 0 else 0.0


def calculate_cdhit_similarity_aligned(seq1, seq2):
    """
    计算 CD-HIT 风格的序列相似度（带全局比对）
    
    使用 SequenceMatcher 找到最佳比对，然后计算相似度
    相似度 = 匹配的字符数 / min(len(seq1), len(seq2))
    """
    if not seq1 or not seq2:
        return 0.0
    
    # 使用 SequenceMatcher 获取匹配块
    matcher = SequenceMatcher(None, seq1, seq2)
    matches = sum(size for _, _, size in matcher.get_matching_blocks())
    
    # CD-HIT 相似度：基于较短序列
    min_len = min(len(seq1), len(seq2))
    return matches / min_len if min_len > 0 else 0.0


def check_high_similarity(target_seqs, training_seqs, similarity_threshold=0.8, method='sequence_matcher', min_seq_length=0):
    """检查3: 高相似度序列"""
    print("\n" + "=" * 80)
    print("检查 3: 高相似度序列")
    print("=" * 80)
    print(f"(相似度阈值: {similarity_threshold}, 方法: {method}, 最小序列长度: {min_seq_length})")
    
    high_similarities = []
    total_comparisons = len(target_seqs) * len(training_seqs)
    comparison_count = 0
    
    for i, target in enumerate(target_seqs):
        for training in training_seqs:
            comparison_count += 1
            
            # 跳过完全相同的序列（已在检查1中处理）
            if target['sequence'] == training['sequence']:
                continue
            
            # 跳过过短的序列（避免 CD-HIT 方法的短序列问题）
            if min_seq_length > 0:
                if len(target['sequence']) < min_seq_length or len(training['sequence']) < min_seq_length:
                    continue
            
            # 计算相似度
            if method == 'sequence_matcher':
                similarity = calculate_sequence_similarity(target['sequence'], training['sequence'])
            elif method == 'kmer':
                similarity = calculate_kmer_similarity(target['sequence'], training['sequence'], k=3)
            elif method == 'cdhit':
                similarity = calculate_cdhit_similarity(target['sequence'], training['sequence'])
            elif method == 'cdhit_aligned':
                similarity = calculate_cdhit_similarity_aligned(target['sequence'], training['sequence'])
            else:
                similarity = calculate_sequence_similarity(target['sequence'], training['sequence'])
            
            if similarity >= similarity_threshold:
                high_similarities.append({
                    'target': target,
                    'training': training,
                    'similarity': similarity,
                    'target_length': len(target['sequence']),
                    'training_length': len(training['sequence'])
                })
            
            # 进度提示
            if comparison_count % 100 == 0:
                progress = comparison_count / total_comparisons * 100
                print(f"  已比较 {comparison_count}/{total_comparisons} ({progress:.1f}%)...", end='\r')
    
    print(" " * 80, end='\r')  # 清除进度提示
    
    if high_similarities:
        # 按相似度排序
        high_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"⚠️  发现 {len(high_similarities)} 对高相似度序列！")
        print(f"\n前 10 个最相似的序列对:")
        
        for i, sim in enumerate(high_similarities[:10], 1):
            print(f"\n  #{i}:")
            print(f"    相似度: {sim['similarity']*100:.2f}%")
            print(f"    目标: {sim['target']['file']} - {sim['target']['header']}")
            print(f"      长度: {sim['target_length']}")
            print(f"      序列: {sim['target']['sequence'][:60]}{'...' if sim['target_length'] > 60 else ''}")
            print(f"    训练集: {sim['training']['file']} - {sim['training']['header']}")
            print(f"      长度: {sim['training_length']}")
            print(f"      序列: {sim['training']['sequence'][:60]}{'...' if sim['training_length'] > 60 else ''}")
        
        if len(high_similarities) > 10:
            print(f"\n    ... 还有 {len(high_similarities) - 10} 对")
    else:
        print("✓ 未发现高相似度序列")
    
    return high_similarities


def save_results(results, output_file):
    """保存检查结果到JSON文件"""
    # 转换为可序列化的格式
    serializable_results = {
        'exact_duplicates': [
            {
                'target': dup['target'],
                'training_matches': dup['training_matches']
            }
            for dup in results['exact_duplicates']
        ],
        'substrings': [
            {
                'type': cont['type'],
                'target': cont['target'],
                'training': cont['training'],
                'coverage': cont['coverage']
            }
            for cont in results['substrings']
        ],
        'high_similarities': [
            {
                'target': sim['target'],
                'training': sim['training'],
                'similarity': sim['similarity']
            }
            for sim in results['high_similarities']
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="数据泄露检查工具 - 增强版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 检查 CASP16 与训练集:
   python check_data_leakage_enhanced.py \\
       --target ./benchmark_data/casp16/sequences \\
       --training ./fine_tuning_data/sequences

2. 使用更严格的阈值:
   python check_data_leakage_enhanced.py \\
       --target ./benchmark_data/casp16/sequences \\
       --training ./fine_tuning_data/sequences \\
       --similarity-threshold 0.7 \\
       --min-substring-length 15

3. 使用 k-mer 相似度方法:
   python check_data_leakage_enhanced.py \\
       --target ./benchmark_data/casp16/sequences \\
       --training ./fine_tuning_data/sequences \\
       --similarity-method kmer

4. 使用 CD-HIT 相似度方法:
   python check_data_leakage_enhanced.py \\
       --target ./benchmark_data/casp16/sequences \\
       --training ./fine_tuning_data/sequences \\
       --similarity-method cdhit_aligned
        """
    )
    
    parser.add_argument("--target", required=True,
                       help="目标数据集目录（benchmark/test set）")
    parser.add_argument("--training", required=True,
                       help="训练数据集目录")
    parser.add_argument("--similarity-threshold", type=float, default=0.8,
                       help="高相似度阈值 (0.0-1.0，默认: 0.8)")
    parser.add_argument("--min-substring-length", type=int, default=10,
                       help="最小子序列长度 (默认: 10)")
    parser.add_argument("--similarity-method", 
                       choices=['sequence_matcher', 'kmer', 'cdhit', 'cdhit_aligned'], 
                       default='sequence_matcher',
                       help="相似度计算方法: sequence_matcher (默认), kmer (Jaccard), cdhit (CD-HIT简单), cdhit_aligned (CD-HIT比对)")
    parser.add_argument("--output", default=None,
                       help="保存详细结果的JSON文件路径")
    parser.add_argument("--recursive", action="store_true",
                       help="递归搜索子目录中的序列文件")
    parser.add_argument("--skip-similarity", action="store_true",
                       help="跳过相似度检查（大数据集时可节省时间）")
    parser.add_argument("--min-seq-length-similarity", type=int, default=0,
                       help="相似度检查时的最小序列长度（推荐 RNA: 15-20, Protein: 30，默认: 0 表示不过滤）")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("数据泄露检查工具 - 增强版")
    print("=" * 80)
    print(f"目标数据集: {args.target}")
    print(f"训练数据集: {args.training}")
    print(f"相似度阈值: {args.similarity_threshold}")
    print(f"最小子序列长度: {args.min_substring_length}")
    print(f"相似度方法: {args.similarity_method}")
    print("=" * 80)
    print()
    
    # 检查目录是否存在
    if not os.path.exists(args.target):
        print(f"错误: 目标目录不存在: {args.target}")
        return 1
    
    if not os.path.exists(args.training):
        print(f"错误: 训练目录不存在: {args.training}")
        return 1
    
    # 加载序列
    print("步骤 1: 加载目标数据集")
    print("-" * 80)
    target_seqs = load_sequences_from_directory(args.target, recursive=args.recursive)
    print()
    
    print("步骤 2: 加载训练数据集")
    print("-" * 80)
    training_seqs = load_sequences_from_directory(args.training, recursive=args.recursive)
    print()
    
    # 执行检查
    results = {}
    
    # 检查1: 完全重复
    results['exact_duplicates'] = check_exact_duplicates(target_seqs, training_seqs)
    
    # 检查2: 子序列包含
    results['substrings'] = check_substring_containment(
        target_seqs, 
        training_seqs, 
        min_length=args.min_substring_length
    )
    
    # 检查3: 高相似度
    if not args.skip_similarity:
        results['high_similarities'] = check_high_similarity(
            target_seqs, 
            training_seqs,
            similarity_threshold=args.similarity_threshold,
            method=args.similarity_method,
            min_seq_length=args.min_seq_length_similarity
        )
    else:
        print("\n⏩ 跳过相似度检查")
        results['high_similarities'] = []
    
    # 总结
    print("\n" + "=" * 80)
    print("检查完成 - 总结")
    print("=" * 80)
    print(f"目标数据集序列数: {len(target_seqs)}")
    print(f"训练数据集序列数: {len(training_seqs)}")
    print()
    print(f"完全重复: {len(results['exact_duplicates'])} 个")
    print(f"子序列包含: {len(results['substrings'])} 个")
    print(f"高相似度 (>={args.similarity_threshold}): {len(results['high_similarities'])} 对")
    print("=" * 80)
    
    # 保存结果
    if args.output:
        save_results(results, args.output)
    
    # 判断是否存在泄露
    has_leakage = (
        len(results['exact_duplicates']) > 0 or 
        len(results['substrings']) > 0 or 
        len(results['high_similarities']) > 0
    )
    
    if has_leakage:
        print("\n⚠️  警告: 检测到潜在的数据泄露！")
        return 1
    else:
        print("\n✓ 未检测到数据泄露")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

