#!/usr/bin/env python3
"""
合并绘制Diffold和RhoFold的训练集依赖性分析图
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path


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


def prepare_data(similarity_file, prediction_file):
    """准备单个模型的数据"""
    sim_df = read_similarity_file(similarity_file)
    pred_df = read_prediction_file(prediction_file)
    merged_df = merge_data(sim_df, pred_df)
    merged_df['tm_score_category'] = merged_df['max_tm_score'].apply(categorize_similarity)
    return merged_df


def plot_combined_comparison(diffold_df, rhofold_df, output_path, figsize=(16, 10), dpi=300):
    """绘制Diffold和RhoFold的合并对比图（2x2布局）"""
    
    # 设置学术化样式
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # 定义分类顺序
    category_order = ['x < 0.25', '0.25 ≤ x < 0.50', '0.50 ≤ x < 0.75', 'x ≥ 0.75']
    
    # 模型数据和名称
    models = [
        (diffold_df, 'Diffold', '#4472C4'),  # 专业蓝色
        (rhofold_df, 'RhoFold+', '#ED7D31')  # 专业橙色
    ]
    
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    
    for idx, (combined_df, model_name, color) in enumerate(models):
        ax_violin = axes[idx * 2]
        ax_scatter = axes[idx * 2 + 1]
        
        # === 左图：小提琴图 ===
        parts = ax_violin.violinplot(
            [combined_df[combined_df['tm_score_category'] == cat]['tm_score'].values 
             for cat in category_order],
            positions=range(len(category_order)),
            showmeans=False,
            showmedians=False,
            widths=0.7
        )
        
        # 设置小提琴图颜色
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.0)
        
        # 添加箱线图叠加
        bp = ax_violin.boxplot(
            [combined_df[combined_df['tm_score_category'] == cat]['tm_score'].values 
             for cat in category_order],
            positions=range(len(category_order)),
            widths=0.15,
            patch_artist=False,
            showfliers=False,
            showcaps=False,
            whiskerprops=dict(linewidth=1.5, color='black'),
            boxprops=dict(linewidth=1.5, color='black'),
            medianprops=dict(linewidth=2.0, color='black')
        )
        
        ax_violin.set_xticks(range(len(category_order)))
        ax_violin.set_xticklabels(category_order, rotation=0, fontsize=10)
        ax_violin.set_xlabel('Training set TM-score category', fontsize=12)
        ax_violin.set_ylabel('TM-score', fontsize=12)
        # 不显示标题
        ax_violin.grid(axis='y', alpha=0.25, linestyle='--')
        ax_violin.set_axisbelow(True)
        ax_violin.spines['top'].set_visible(False)
        ax_violin.spines['right'].set_visible(False)
        ax_violin.spines['left'].set_linewidth(1.2)
        ax_violin.spines['bottom'].set_linewidth(1.2)
        
        # 添加面板标签
        ax_violin.text(-0.15, 1.05, panel_labels[idx * 2], transform=ax_violin.transAxes,
                      fontsize=14, fontweight='bold', va='top', ha='right')
        
        # === 右图：散点图 ===
        # 计算相关系数
        corr, p_value = stats.pearsonr(combined_df['max_tm_score'], combined_df['tm_score'])
        
        ax_scatter.scatter(combined_df['max_tm_score'], combined_df['tm_score'], 
                          alpha=0.6, s=40, color=color, edgecolor='black', linewidth=0.5)
        
        # 添加回归线
        z = np.polyfit(combined_df['max_tm_score'], combined_df['tm_score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(combined_df['max_tm_score'].min(), 
                            combined_df['max_tm_score'].max(), 100)
        ax_scatter.plot(x_line, p(x_line), color='darkred', alpha=0.8, linewidth=2.5)
        
        # 添加置信区间
        predict_y = p(combined_df['max_tm_score'])
        pred_error = combined_df['tm_score'] - predict_y
        degrees_of_freedom = len(combined_df['max_tm_score']) - 2
        residual_std_error = np.sqrt(np.sum(pred_error**2) / degrees_of_freedom)
        
        x_mean = np.mean(combined_df['max_tm_score'])
        sxx = np.sum((combined_df['max_tm_score'] - x_mean)**2)
        confidence = 0.95
        t_val = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
        
        confs = t_val * residual_std_error * np.sqrt(1/len(combined_df['max_tm_score']) + 
                                                       (x_line - x_mean)**2 / sxx)
        
        ax_scatter.fill_between(x_line, p(x_line) - confs, p(x_line) + confs, 
                               alpha=0.2, color='darkred')
        
        ax_scatter.set_xlabel('Training set TM-score', fontsize=12)
        ax_scatter.set_ylabel('TM-score', fontsize=12)
        # 不显示标题
        ax_scatter.grid(alpha=0.25, linestyle='--')
        ax_scatter.set_axisbelow(True)
        ax_scatter.spines['top'].set_visible(False)
        ax_scatter.spines['right'].set_visible(False)
        ax_scatter.spines['left'].set_linewidth(1.2)
        ax_scatter.spines['bottom'].set_linewidth(1.2)
        
        # 添加面板标签
        ax_scatter.text(-0.15, 1.05, panel_labels[idx * 2 + 1], transform=ax_scatter.transAxes,
                       fontsize=14, fontweight='bold', va='top', ha='right')
        
        # 在图内添加相关系数 - Diffold在右上角，RhoFold在左上角
        if idx == 0:  # Diffold
            text_x, text_ha = 0.95, 'right'
        else:  # RhoFold
            text_x, text_ha = 0.05, 'left'
        
        ax_scatter.text(text_x, 0.95, f'r = {corr:.3f}, p = {p_value:.1e}', 
                       transform=ax_scatter.transAxes, fontsize=10,
                       verticalalignment='top', horizontalalignment=text_ha,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, 
                                edgecolor='gray', linewidth=0.8))
    
    # 增加子图间距
    plt.subplots_adjust(wspace=0.35, hspace=0.25)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\n合并图片已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='合并绘制Diffold和RhoFold的训练集依赖性分析图'
    )
    
    parser.add_argument('--diffold-similarity', required=True, help='Diffold相似性文件')
    parser.add_argument('--diffold-prediction', required=True, help='Diffold预测文件')
    parser.add_argument('--rhofold-similarity', required=True, help='RhoFold相似性文件')
    parser.add_argument('--rhofold-prediction', required=True, help='RhoFold预测文件')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--figsize', nargs=2, type=float, default=[16, 10], help='图片大小')
    parser.add_argument('--dpi', type=int, default=300, help='图片分辨率')
    
    args = parser.parse_args()
    
    print("读取Diffold数据...")
    diffold_df = prepare_data(args.diffold_similarity, args.diffold_prediction)
    print(f"  样本数: {len(diffold_df)}")
    
    print("读取RhoFold数据...")
    rhofold_df = prepare_data(args.rhofold_similarity, args.rhofold_prediction)
    print(f"  样本数: {len(rhofold_df)}")
    
    print("\n绘制合并图...")
    plot_combined_comparison(
        diffold_df, rhofold_df, 
        args.output, 
        tuple(args.figsize), 
        args.dpi
    )
    
    print("\n完成！")


if __name__ == '__main__':
    main()

