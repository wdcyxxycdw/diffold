#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并序列相似度（BLAST）和结构相似度（TM-score）分析结果
"""

import argparse
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_base_id(test_sequence: str) -> str:
    """从test_sequence中提取base_id（第一个下划线或竖线前的部分）"""
    # 处理格式如: "7Q48_1|Chain A|..." 或 "R1203"
    if '_' in test_sequence:
        return test_sequence.split('_')[0]
    elif '|' in test_sequence:
        return test_sequence.split('|')[0]
    else:
        return test_sequence


def merge_tables(tm_file: str, blast_file: str, output_file: str):
    """合并TM-score和BLAST结果表"""
    logger.info("=" * 80)
    logger.info("合并相似度分析表")
    logger.info("=" * 80)
    
    # 读取TM-score表（包含结构相似度和预测结果）
    logger.info(f"读取TM-score表: {tm_file}")
    tm_df = pd.read_csv(tm_file, sep='\t')
    logger.info(f"  TM-score表包含 {len(tm_df)} 行")
    
    # 读取BLAST表（包含序列相似度）
    logger.info(f"读取BLAST表: {blast_file}")
    blast_df = pd.read_csv(blast_file, sep='\t')
    logger.info(f"  BLAST表包含 {len(blast_df)} 行")
    
    # 从BLAST表的test_sequence中提取base_id
    blast_df['base_id'] = blast_df['test_sequence'].apply(extract_base_id)
    
    # 重命名TM-score表的列（如果需要）
    if '测试样本' in tm_df.columns:
        tm_df = tm_df.rename(columns={'测试样本': 'base_id'})
    
    # 合并两个表
    logger.info("合并表...")
    merged_df = pd.merge(
        tm_df,
        blast_df[['base_id', 'max_identity', 'max_qcov', 'max_tcov', 
                  'effective_identity', 'best_match', 'best_match_identity',
                  'best_match_qcov', 'best_match_tcov', 'num_hits', 'query_length']],
        on='base_id',
        how='inner'
    )
    
    logger.info(f"合并后包含 {len(merged_df)} 行")
    
    # 重新排列列的顺序，使其更易读
    column_order = [
        'base_id',
        'query_length',
        'max_identity',
        'max_qcov',
        'effective_identity',
        'best_match',
        'num_hits',
        '最大TM-score',
        '最相似训练样本',
        '高相似度计数',
        'Diffold_TM-score',
        'Diffold_RMSD',
        'RhoFold+_TM-score',
        'RhoFold+_RMSD',
    ]
    
    # 只保留存在的列
    available_columns = [col for col in column_order if col in merged_df.columns]
    # 添加其他未列出的列
    other_columns = [col for col in merged_df.columns if col not in available_columns]
    final_columns = available_columns + other_columns
    
    merged_df = merged_df[final_columns]
    
    # 按effective_identity降序排序（如果有的话）
    if 'effective_identity' in merged_df.columns:
        merged_df = merged_df.sort_values('effective_identity', ascending=False, na_position='last')
    
    # 保存结果
    logger.info(f"保存合并结果到: {output_file}")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, sep='\t', index=False, float_format='%.4f')
    
    # 打印统计信息
    logger.info("\n" + "=" * 80)
    logger.info("合并统计")
    logger.info("=" * 80)
    logger.info(f"总样本数: {len(merged_df)}")
    
    if 'effective_identity' in merged_df.columns:
        eff_id = merged_df['effective_identity'].dropna()
        if len(eff_id) > 0:
            logger.info(f"有效序列相似度范围: {eff_id.min():.2f}% - {eff_id.max():.2f}%")
            logger.info(f"平均有效序列相似度: {eff_id.mean():.2f}%")
    
    if '最大TM-score' in merged_df.columns:
        max_tm = merged_df['最大TM-score'].dropna()
        if len(max_tm) > 0:
            logger.info(f"最大结构相似度范围: {max_tm.min():.4f} - {max_tm.max():.4f}")
            logger.info(f"平均最大结构相似度: {max_tm.mean():.4f}")
    
    logger.info("=" * 80)
    logger.info("合并完成！")


def main():
    parser = argparse.ArgumentParser(
        description="合并序列相似度（BLAST）和结构相似度（TM-score）分析结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python scripts/merge_similarity_tables.py \\
      --tm_file results/similarity_tests/combined/tm_scores_analysis_with_pred.tsv \\
      --blast_file results/similarity_tests/combined/train_test_similarity_blast.tsv \\
      --output results/similarity_tests/combined/comprehensive_similarity.tsv
        """
    )
    
    parser.add_argument(
        '--tm_file',
        required=True,
        help='TM-score分析结果文件（包含结构相似度和预测结果）'
    )
    parser.add_argument(
        '--blast_file',
        required=True,
        help='BLAST序列相似度分析结果文件'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='输出合并后的TSV文件路径'
    )
    
    args = parser.parse_args()
    
    merge_tables(args.tm_file, args.blast_file, args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())

