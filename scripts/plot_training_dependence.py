#!/usr/bin/env python3
"""
绘制训练集依赖性分析图
该脚本用于分析测试集与训练集相似程度和预测结果之间的相关性
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from pathlib import Path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='绘制训练集依赖性分析图，展示测试集与训练集相似程度和预测结果之间的相关性'
    )
    
    parser.add_argument(
        '--similarity-files',
        nargs='+',
        required=True,
        help='相似性评估文件路径（TSV格式），可以输入多个文件'
    )
    
    parser.add_argument(
        '--prediction-files',
        nargs='+',
        required=True,
        help='模型预测评估报告文件路径（CSV格式），可以输入多个文件，需与相似性文件一一对应'
    )
    
    parser.add_argument(
        '--dataset-names',
        nargs='+',
        default=None,
        help='数据集名称，用于标识不同的测试集。如果不提供，将从文件路径中自动提取'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        default='training_dependence_analysis.png',
        help='输出图片文件路径（默认：training_dependence_analysis.png）'
    )
    
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=float,
        default=[14, 6],
        help='图片大小（宽度 高度），默认为 14 6'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='图片分辨率（默认：300）'
    )
    
    parser.add_argument(
        '--title',
        type=str,
        default='Training Set Dependence Analysis',
        help='图片总标题（已废弃，将被忽略）'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='Model',
        help='模型名称（如RhoFold、DiffOld等），用于图表标题'
    )
    
    return parser.parse_args()


def read_similarity_file(filepath):
    """读取相似性评估文件（TSV格式）"""
    df = pd.read_csv(filepath, sep='\t')
    # 重命名列以便统一处理
    df.columns = ['sample_name', 'max_tm_score', 'most_similar_sample', 'high_similarity_count']
    return df


def read_prediction_file(filepath):
    """读取预测评估文件（CSV格式）"""
    df = pd.read_csv(filepath)
    # 确保包含必要的列
    if 'sample_name' not in df.columns or 'tm_score' not in df.columns:
        raise ValueError(f"预测文件 {filepath} 必须包含 'sample_name' 和 'tm_score' 列")
    return df[['sample_name', 'tm_score']]


def merge_data(similarity_df, prediction_df):
    """合并相似性数据和预测数据"""
    merged_df = pd.merge(
        similarity_df,
        prediction_df,
        on='sample_name',
        how='inner'
    )
    return merged_df


def categorize_similarity(tm_score):
    """根据TM-score将样本分类"""
    if tm_score < 0.25:
        return 'x < 0.25'
    elif tm_score < 0.50:
        return '0.25 ≤ x < 0.50'
    elif tm_score < 0.75:
        return '0.50 ≤ x < 0.75'
    else:
        return 'x ≥ 0.75'


def prepare_combined_data(similarity_files, prediction_files, dataset_names=None):
    """准备合并的数据"""
    all_data = []
    
    if len(similarity_files) != len(prediction_files):
        raise ValueError("相似性文件和预测文件的数量必须相同")
    
    # 如果没有提供数据集名称，从文件路径中提取
    if dataset_names is None:
        dataset_names = []
        for sim_file in similarity_files:
            # 尝试从文件路径中提取有意义的名称
            path = Path(sim_file)
            name = path.parent.name if path.parent.name != 'dataset_comparision' else path.stem
            dataset_names.append(name.replace('_tm_scores_analysis', ''))
    
    if len(dataset_names) != len(similarity_files):
        raise ValueError("数据集名称的数量必须与文件数量相同")
    
    # 读取并合并所有数据集
    for sim_file, pred_file, dataset_name in zip(similarity_files, prediction_files, dataset_names):
        print(f"处理数据集: {dataset_name}")
        
        # 读取数据
        sim_df = read_similarity_file(sim_file)
        pred_df = read_prediction_file(pred_file)
        
        # 合并数据
        merged_df = merge_data(sim_df, pred_df)
        
        # 添加数据集标识
        merged_df['dataset'] = dataset_name
        
        # 添加分类
        merged_df['tm_score_category'] = merged_df['max_tm_score'].apply(categorize_similarity)
        
        all_data.append(merged_df)
        
        print(f"  - 样本数: {len(merged_df)}")
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df, dataset_names


def plot_training_dependence(combined_df, output_path, figsize, dpi, model_name):
    """绘制训练集依赖性分析图"""
    
    # 创建图形，增加底部空间用于显示统计信息
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    # 调整子图位置，为底部留出空间，同时优化整体布局
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.92, wspace=0.25)
    
    # 定义分类顺序
    category_order = ['x < 0.25', '0.25 ≤ x < 0.50', '0.50 ≤ x < 0.75', 'x ≥ 0.75']
    
    # 左图：小提琴图
    # 设置颜色
    color_palette = sns.color_palette("Blues", n_colors=1)
    
    # 绘制小提琴图
    parts = ax1.violinplot(
        [combined_df[combined_df['tm_score_category'] == cat]['tm_score'].values 
         for cat in category_order],
        positions=range(len(category_order)),
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # 设置小提琴图颜色
    for pc in parts['bodies']:
        pc.set_facecolor('#5975A4')
        pc.set_alpha(0.7)
    
    # 设置x轴
    ax1.set_xticks(range(len(category_order)))
    ax1.set_xticklabels(category_order, rotation=0)
    ax1.set_xlabel('TM-score Category', fontsize=12)
    ax1.set_ylabel('TM-score', fontsize=12)
    ax1.set_title(f'Training Set Dependence - {model_name}', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 计算相关系数
    corr, p_value = stats.pearsonr(combined_df['max_tm_score'], combined_df['tm_score'])
    
    # 右图：散点图
    ax2.scatter(combined_df['max_tm_score'], combined_df['tm_score'], 
                alpha=0.6, s=50, color='#5975A4')
    
    # 添加回归线
    z = np.polyfit(combined_df['max_tm_score'], combined_df['tm_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(combined_df['max_tm_score'].min(), 
                         combined_df['max_tm_score'].max(), 100)
    ax2.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2)
    
    # 添加置信区间
    from scipy import stats as sp_stats
    predict_y = p(combined_df['max_tm_score'])
    pred_error = combined_df['tm_score'] - predict_y
    degrees_of_freedom = len(combined_df['max_tm_score']) - 2
    residual_std_error = np.sqrt(np.sum(pred_error**2) / degrees_of_freedom)
    
    # 计算置信区间
    x_mean = np.mean(combined_df['max_tm_score'])
    sxx = np.sum((combined_df['max_tm_score'] - x_mean)**2)
    confidence = 0.95
    t_val = sp_stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
    
    confs = t_val * residual_std_error * np.sqrt(1/len(combined_df['max_tm_score']) + 
                                                   (x_line - x_mean)**2 / sxx)
    
    ax2.fill_between(x_line, p(x_line) - confs, p(x_line) + confs, 
                     alpha=0.2, color='red')
    
    ax2.set_xlabel('Training set TM-Score', fontsize=12)
    ax2.set_ylabel('TM-score', fontsize=12)
    ax2.set_title(f'Training set TM-Score vs True TM-scores - {model_name}', 
                  fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    
    # 在图表底部添加相关系数信息（不遮挡数据）
    fig.text(0.5, 0.02, f'Pearson Correlation: r = {corr:.4f}, p = {p_value:.2e}', 
             ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 保存图片
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")
    
    # 显示图片
    plt.close()


def print_statistics(combined_df):
    """打印统计信息"""
    print("\n=== 数据统计 ===")
    print(f"总样本数: {len(combined_df)}")
    print(f"\n各数据集样本数:")
    print(combined_df['dataset'].value_counts())
    
    print(f"\n各分类样本数:")
    category_order = ['x < 0.25', '0.25 ≤ x < 0.50', '0.50 ≤ x < 0.75', 'x ≥ 0.75']
    for cat in category_order:
        count = len(combined_df[combined_df['tm_score_category'] == cat])
        mean_tm = combined_df[combined_df['tm_score_category'] == cat]['tm_score'].mean()
        print(f"  {cat}: {count} 个样本, 平均TM-score: {mean_tm:.4f}")
    
    # 计算相关系数
    corr, p_value = stats.pearsonr(combined_df['max_tm_score'], combined_df['tm_score'])
    print(f"\n相关性分析:")
    print(f"  Pearson相关系数: {corr:.4f}")
    print(f"  P值: {p_value:.4e}")
    
    # Spearman相关系数
    spearman_corr, spearman_p = stats.spearmanr(combined_df['max_tm_score'], 
                                                  combined_df['tm_score'])
    print(f"  Spearman相关系数: {spearman_corr:.4f}")
    print(f"  P值: {spearman_p:.4e}")


def main():
    """主函数"""
    args = parse_args()
    
    print("=== 训练集依赖性分析 ===\n")
    
    # 准备数据
    print("读取数据文件...")
    combined_df, dataset_names = prepare_combined_data(
        args.similarity_files,
        args.prediction_files,
        args.dataset_names
    )
    
    # 打印统计信息
    print_statistics(combined_df)
    
    # 绘制图表
    print("\n绘制图表...")
    plot_training_dependence(
        combined_df,
        args.output,
        tuple(args.figsize),
        args.dpi,
        args.model_name
    )
    
    print("\n完成！")


if __name__ == '__main__':
    main()

