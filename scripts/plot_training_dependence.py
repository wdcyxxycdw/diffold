#!/usr/bin/env python3
"""
合并绘制Diffold和RhoFold的训练集依赖性分析图
支持序列相似度和结构相似度的综合分析
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path


def read_comprehensive_file(filepath):
    """读取综合相似度分析文件（包含序列和结构相似度）"""
    df = pd.read_csv(filepath, sep='\t')
    # 确保包含必要的列
    required_cols = ['base_id', 'effective_identity', '最大TM-score', 
                     'Diffold_TM-score', 'RhoFold+_TM-score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"文件缺少必要的列: {missing_cols}")
    return df


def read_similarity_file(filepath):
    """读取相似性评估文件（TSV格式）- 向后兼容"""
    df = pd.read_csv(filepath, sep='\t')
    # 重命名列以便统一处理
    df.columns = ['sample_name', 'max_tm_score', 'most_similar_sample', 'high_similarity_count']
    return df


def read_prediction_file(filepath):
    """读取预测评估文件（CSV格式）- 向后兼容"""
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
    """准备单个模型的数据 - 向后兼容"""
    sim_df = read_similarity_file(similarity_file)
    pred_df = read_prediction_file(prediction_file)
    merged_df = merge_data(sim_df, pred_df)
    merged_df['tm_score_category'] = merged_df['max_tm_score'].apply(categorize_similarity)
    return merged_df


def prepare_comprehensive_data(comprehensive_file):
    """从综合文件准备数据"""
    df = read_comprehensive_file(comprehensive_file)
    
    # 添加分类
    df['tm_score_category'] = df['最大TM-score'].apply(categorize_similarity)
    
    # 为两个模型准备数据，分别过滤各自有结果的样本
    # Diffold数据：只保留有Diffold_TM-score的样本
    diffold_df = df[df['Diffold_TM-score'].notna()][['base_id', 'effective_identity', '最大TM-score', 
                     'Diffold_TM-score', 'tm_score_category']].copy()
    diffold_df.columns = ['sample_name', 'effective_identity', 'max_tm_score', 
                          'tm_score', 'tm_score_category']
    
    # RhoFold数据：只保留有RhoFold+_TM-score的样本
    rhofold_df = df[df['RhoFold+_TM-score'].notna()][['base_id', 'effective_identity', '最大TM-score', 
                     'RhoFold+_TM-score', 'tm_score_category']].copy()
    rhofold_df.columns = ['sample_name', 'effective_identity', 'max_tm_score', 
                          'tm_score', 'tm_score_category']
    
    return diffold_df, rhofold_df


def plot_scatter_with_regression(ax, x_data, y_data, color, x_label, y_label, 
                                 panel_label, text_pos='right'):
    """绘制带回归线和置信区间的散点图"""
    # 过滤NaN值
    mask = ~(pd.isna(x_data) | pd.isna(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    if len(x_clean) < 3:
        ax.text(0.5, 0.5, '数据不足', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        return
    
    # 计算相关系数
    corr, p_value = stats.pearsonr(x_clean, y_clean)
    
    # 散点图
    ax.scatter(x_clean, y_clean, alpha=0.6, s=40, color=color, 
               edgecolor='black', linewidth=0.5)
    
    # 回归线
    z = np.polyfit(x_clean, y_clean, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    ax.plot(x_line, p(x_line), color='darkred', alpha=0.8, linewidth=2.5)
    
    # 置信区间
    predict_y = p(x_clean)
    pred_error = y_clean - predict_y
    degrees_of_freedom = len(x_clean) - 2
    residual_std_error = np.sqrt(np.sum(pred_error**2) / degrees_of_freedom)
    
    x_mean = np.mean(x_clean)
    sxx = np.sum((x_clean - x_mean)**2)
    confidence = 0.95
    t_val = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
    
    confs = t_val * residual_std_error * np.sqrt(1/len(x_clean) + 
                                                   (x_line - x_mean)**2 / sxx)
    
    ax.fill_between(x_line, p(x_line) - confs, p(x_line) + confs, 
                    alpha=0.2, color='darkred')
    
    # 标签和样式
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # 面板标签
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right')
    
    # 相关系数
    text_x = 0.95 if text_pos == 'right' else 0.05
    text_ha = 'right' if text_pos == 'right' else 'left'
    
    ax.text(text_x, 0.95, f'r = {corr:.3f}, p = {p_value:.1e}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment=text_ha,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, 
                     edgecolor='gray', linewidth=0.8))


def plot_combined_comparison(diffold_df, rhofold_df, output_path, figsize=(20, 11), dpi=300):
    """绘制Diffold和RhoFold的合并对比图（2x3布局）"""
    
    # 设置学术化样式
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5
    
    # 创建2x3子图，为顶部图例留出空间
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 定义分类顺序
    category_order = ['x < 0.25', '0.25 ≤ x < 0.50', '0.50 ≤ x < 0.75', 'x ≥ 0.75']
    
    # 模型数据和名称
    models = [
        (diffold_df, 'Diffold', '#4472C4'),  # 专业蓝色
        (rhofold_df, 'RhoFold+', '#ED7D31')  # 专业橙色
    ]
    
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    # 列标题
    col_titles = [
        'Performance Distribution by Structural Similarity',
        'Structural Similarity vs Performance', 
        'Sequence Similarity vs Performance'
    ]
    
    for row_idx, (combined_df, model_name, color) in enumerate(models):
        # === 第1列：小提琴图 ===
        ax_violin = axes[row_idx, 0]
        
        # 准备数据，过滤掉空分类
        violin_data = []
        violin_positions = []
        valid_labels = []
        
        for i, cat in enumerate(category_order):
            cat_data = combined_df[combined_df['tm_score_category'] == cat]['tm_score'].values
            if len(cat_data) > 0:  # 只添加有数据的分类
                violin_data.append(cat_data)
                violin_positions.append(i)
                valid_labels.append(cat)
        
        # 只有在有数据时才绘制小提琴图
        if len(violin_data) > 0:
            parts = ax_violin.violinplot(
                violin_data,
                positions=violin_positions,
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
            ax_violin.boxplot(
                violin_data,
                positions=violin_positions,
                widths=0.15,
                patch_artist=False,
                showfliers=False,
                showcaps=False,
                whiskerprops=dict(linewidth=1.5, color='black'),
                boxprops=dict(linewidth=1.5, color='black'),
                medianprops=dict(linewidth=2.0, color='black')
            )
            
            # 设置x轴刻度和标签（显示所有分类，即使某些没有数据）
            ax_violin.set_xticks(range(len(category_order)))
            ax_violin.set_xticklabels(category_order, rotation=0, fontsize=10)
        else:
            # 如果没有数据，显示提示
            ax_violin.text(0.5, 0.5, 'No data', ha='center', va='center', 
                          transform=ax_violin.transAxes, fontsize=12)
            ax_violin.set_xticks(range(len(category_order)))
            ax_violin.set_xticklabels(category_order, rotation=0, fontsize=10)
        ax_violin.set_xlabel('Training set TM-score category', fontsize=12)
        ax_violin.set_ylabel('TM-score', fontsize=12)
        ax_violin.grid(axis='y', alpha=0.25, linestyle='--')
        ax_violin.set_axisbelow(True)
        ax_violin.spines['top'].set_visible(False)
        ax_violin.spines['right'].set_visible(False)
        ax_violin.spines['left'].set_linewidth(1.2)
        ax_violin.spines['bottom'].set_linewidth(1.2)
        
        # 面板标签
        ax_violin.text(-0.15, 1.05, panel_labels[row_idx * 3], 
                      transform=ax_violin.transAxes,
                      fontsize=14, fontweight='bold', va='top', ha='right')
        
        # 只在第一行添加列标题
        if row_idx == 0:
            ax_violin.set_title(col_titles[0], fontsize=13, fontweight='bold', pad=15)
        
        # === 第2列：结构相似度（max_tm_score）散点图（对调后） ===
        ax_struct = axes[row_idx, 1]
        text_pos = 'right' if row_idx == 0 else 'left'
        
        plot_scatter_with_regression(
            ax_struct,
            combined_df['max_tm_score'],
            combined_df['tm_score'],
            color,
            'Structural similarity (training set TM-score)',
            'TM-score',
            panel_labels[row_idx * 3 + 1],
            text_pos
        )
        
        # 只在第一行添加列标题
        if row_idx == 0:
            ax_struct.set_title(col_titles[1], fontsize=13, fontweight='bold', pad=15)
        
        # === 第3列：序列相似度（effective_identity）散点图（对调后） ===
        ax_seq = axes[row_idx, 2]
        
        plot_scatter_with_regression(
            ax_seq, 
            combined_df['effective_identity'], 
            combined_df['tm_score'],
            color,
            'Sequence similarity (effective identity, %)',
            'TM-score',
            panel_labels[row_idx * 3 + 2],
            text_pos
        )
        
        # 只在第一行添加列标题
        if row_idx == 0:
            ax_seq.set_title(col_titles[2], fontsize=13, fontweight='bold', pad=15)
    
    # 在图的底部添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4472C4', edgecolor='black', label='Diffold'),
        Patch(facecolor='#ED7D31', edgecolor='black', label='RhoFold+')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               fontsize=13, frameon=True, edgecolor='black', 
               bbox_to_anchor=(0.5, -0.05), columnspacing=2)
    
    # 调整子图间距，为底部图例留出更多空间
    plt.subplots_adjust(wspace=0.3, hspace=0.3, bottom=0.08)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\n合并图片已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='合并绘制Diffold和RhoFold的训练集依赖性分析图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：

1. 使用综合分析文件（推荐，包含序列和结构相似度）：
   python plot_training_dependence.py \\
       --comprehensive-file merged_similarity_performance.tsv \\
       --output combined_analysis.png

2. 使用分离的文件（旧方式，向后兼容）：
   python plot_training_dependence.py \\
       --diffold-similarity diffold_sim.tsv \\
       --diffold-prediction diffold_pred.csv \\
       --rhofold-similarity rhofold_sim.tsv \\
       --rhofold-prediction rhofold_pred.csv \\
       --output combined_analysis.png
        """
    )
    
    parser.add_argument('--comprehensive-file', help='综合相似度分析文件（包含两个模型的数据）')
    parser.add_argument('--diffold-similarity', help='Diffold相似性文件（旧格式）')
    parser.add_argument('--diffold-prediction', help='Diffold预测文件（旧格式）')
    parser.add_argument('--rhofold-similarity', help='RhoFold相似性文件（旧格式）')
    parser.add_argument('--rhofold-prediction', help='RhoFold预测文件（旧格式）')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--figsize', nargs=2, type=float, default=[20, 10], help='图片大小')
    parser.add_argument('--dpi', type=int, default=300, help='图片分辨率')
    
    args = parser.parse_args()
    
    # 判断使用哪种模式
    if args.comprehensive_file:
        # 新模式：从综合文件读取
        print(f"读取综合分析文件: {args.comprehensive_file}")
        diffold_df, rhofold_df = prepare_comprehensive_data(args.comprehensive_file)
        print(f"  Diffold样本数: {len(diffold_df)}")
        print(f"  RhoFold样本数: {len(rhofold_df)}")
    else:
        # 旧模式：从分离文件读取
        if not all([args.diffold_similarity, args.diffold_prediction, 
                   args.rhofold_similarity, args.rhofold_prediction]):
            parser.error("使用旧模式时，必须提供所有四个文件参数，或使用 --comprehensive-file")
        
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

