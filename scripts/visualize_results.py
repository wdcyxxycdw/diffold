#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNA结构预测结果可视化脚本
分析batch_inference_results.csv中的测试结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

import os
SAVE_PATH = "/work/gs58/s58009/rhofold/plots/casp15_ft2"
LOAD_PATH = "/work/gs58/s58009/archive/ft2/casp15_eval_results/merged_results.csv"

# 设置字体和样式 - 使用英文标签避免字体问题
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data(file_path):
    """加载和清理数据"""
    print("正在加载数据...")
    df = pd.read_csv(file_path)
    
    # 数据基本信息
    print(f"数据总数: {len(df)} 个样本")
    print(f"成功预测: {len(df[df['status'] == 'success'])} 个")
    print(f"失败预测: {len(df[df['status'] != 'success'])} 个")
    
    # 只分析成功的预测
    df_success = df[df['status'] == 'success'].copy()
    print(f"用于分析的成功样本数: {len(df_success)}")
    
    return df, df_success

def plot_metric_distributions(df, figsize=(20, 15)):
    """绘制主要指标的分布图（小提琴图 + 箱线图）"""
    # 主要分析指标 - 使用Diffold的列名
    metrics = {
        'best_rmsd': 'RMSD (Angstrom)',
        'avg_tm_score': 'TM-Score',
        'avg_lddt': 'LDDT Score',
        'avg_clash_score': 'Clash Score',
        'sequence_length': 'Sequence Length'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(metrics.items()):
        if i < len(axes):
            ax = axes[i]
            
            # 小提琴图
            violin_parts = ax.violinplot([df[metric].dropna()], positions=[0], 
                                       showmeans=True, showmedians=True, widths=0.6)
            
            # 设置小提琴图颜色
            for pc in violin_parts['bodies']:
                pc.set_facecolor('#8dd3c7')
                pc.set_alpha(0.7)
            
            # 添加箱线图
            box_plot = ax.boxplot([df[metric].dropna()], positions=[0], widths=0.3,
                                patch_artist=True, showfliers=True)
            box_plot['boxes'][0].set_facecolor('#ffd92f')
            box_plot['boxes'][0].set_alpha(0.8)
            
            # 设置标题和标签
            ax.set_title(f'{label} Distribution', fontsize=14, fontweight='bold')
            ax.set_ylabel(label, fontsize=12)
            ax.set_xticks([])
            
            # 添加统计信息
            mean_val = df[metric].mean()
            median_val = df[metric].median()
            std_val = df[metric].std()
            
            stats_text = f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
            
            ax.grid(True, alpha=0.3)
    
    # 删除多余的子图
    if len(metrics) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_PATH}/metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_individual_violin_plots(df):
    """为每个指标单独绘制小提琴图"""
    # 主要分析指标 - 按性能好坏的方向定义 - 使用Diffold的列名
    metrics_info = {
        'best_rmsd': {'label': 'RMSD (Angstrom)', 'better': 'lower', 'unit': 'Å'},
        'avg_tm_score': {'label': 'TM-Score', 'better': 'higher', 'unit': ''},
        'avg_lddt': {'label': 'LDDT Score', 'better': 'higher', 'unit': ''},
        'avg_clash_score': {'label': 'Clash Score', 'better': 'lower', 'unit': ''},
        'sequence_length': {'label': 'Sequence Length', 'better': 'neutral', 'unit': 'residues'}
    }
    
    # 颜色调色板
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']
    
    for i, (metric, info) in enumerate(metrics_info.items()):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        data = df[metric].dropna()
        
        # 对于'better=lower'的指标（RMSD, Clash Score），翻转Y轴让好的值在上方
        if info['better'] == 'lower':
            # 创建翻转的数据用于显示
            display_data = -data  # 翻转数据
            violin_parts = ax.violinplot([display_data], positions=[0], 
                                       showmeans=True, showmedians=True, widths=0.8)
            # 但坐标轴标签仍使用原始数据
            y_ticks = ax.get_yticks()
            ax.set_yticklabels([f'{-tick:.2f}' for tick in y_ticks])
            y_label_suffix = " (Better →)"
        else:
            # 对于'better=higher'的指标，正常显示
            violin_parts = ax.violinplot([data], positions=[0], 
                                       showmeans=True, showmedians=True, widths=0.8)
            if info['better'] == 'higher':
                y_label_suffix = " (Better →)"
            else:  # neutral
                y_label_suffix = ""
        
        # 设置小提琴图颜色
        for pc in violin_parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # 设置其他元素颜色
        violin_parts['cmeans'].set_color('red')
        violin_parts['cmeans'].set_linewidth(2)
        violin_parts['cmedians'].set_color('darkblue')
        violin_parts['cmedians'].set_linewidth(2)
        violin_parts['cbars'].set_color('black')
        violin_parts['cmaxes'].set_color('black')
        violin_parts['cmins'].set_color('black')
        
        # 添加数据点（使用抖动避免重叠）
        np.random.seed(42)  # 保证可重复性
        jittered_x = np.random.normal(0, 0.04, size=len(data))
        if info['better'] == 'lower':
            ax.scatter(jittered_x, -data, alpha=0.3, s=20, color='black')
        else:
            ax.scatter(jittered_x, data, alpha=0.3, s=20, color='black')
        
        # 设置标题和标签
        ax.set_title(f'{info["label"]} Distribution', fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel(f'{info["label"]}{y_label_suffix}', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        
        # 计算统计信息（始终用原始数据）
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        min_val = data.min()
        max_val = data.max()
        
        # 统计信息文本
        unit_str = f" {info['unit']}" if info['unit'] else ""
        stats_text = f'''Statistics:
            Mean: {mean_val:.3f}{unit_str}
            Median: {median_val:.3f}{unit_str}
            Std: {std_val:.3f}{unit_str}
            Q25: {q25:.3f}{unit_str}
            Q75: {q75:.3f}{unit_str}
            Min: {min_val:.3f}{unit_str}
            Max: {max_val:.3f}{unit_str}
            Count: {len(data)}'''
        
        # 添加均值和中位数线
        if info['better'] == 'lower':
            mean_line = ax.axhline(y=-mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.3f}{unit_str}')
            median_line = ax.axhline(y=-median_val, color='darkblue', linestyle='--', alpha=0.8, linewidth=2, label=f'Median: {median_val:.3f}{unit_str}')
        else:
            mean_line = ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.3f}{unit_str}')
            median_line = ax.axhline(y=median_val, color='darkblue', linestyle='--', alpha=0.8, linewidth=2, label=f'Median: {median_val:.3f}{unit_str}')
        
        # 将图例放在正下方中央
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10)
        
        # 将统计信息放在小提琴图内部的右下角
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
               verticalalignment='bottom', horizontalalignment='right', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#f8f9fa')
        
        # 调整布局，为底部图例留出适当空间
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # 保存图片到文件夹
        filename = f'{SAVE_PATH}/{metric}_violin_plot.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   保存: {filename}")
        plt.show()

def plot_sequence_length_analysis(df, figsize=(15, 10)):
    """分析序列长度对预测质量的影响"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 创建序列长度分组
    df['length_group'] = pd.cut(df['sequence_length'], 
                               bins=[0, 30, 50, 80, 120, np.inf],
                               labels=['Very Short(≤30)', 'Short(31-50)', 'Medium(51-80)', 'Long(81-120)', 'Very Long(>120)'])
    
    # RMSD vs Sequence Length
    axes[0,0].scatter(df['sequence_length'], df['best_rmsd'], alpha=0.6, s=50)
    axes[0,0].set_xlabel('Sequence Length')
    axes[0,0].set_ylabel('RMSD (Angstrom)')
    axes[0,0].set_title('RMSD vs Sequence Length')
    
    # 添加趋势线
    z = np.polyfit(df['sequence_length'], df['best_rmsd'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(df['sequence_length'], p(df['sequence_length']), "r--", alpha=0.8)
    
    # TM-Score vs Sequence Length  
    axes[0,1].scatter(df['sequence_length'], df['avg_tm_score'], alpha=0.6, s=50, color='green')
    axes[0,1].set_xlabel('Sequence Length')
    axes[0,1].set_ylabel('TM-Score')
    axes[0,1].set_title('TM-Score vs Sequence Length')
    
    # RMSD Distribution by Length Groups
    sns.boxplot(data=df, x='length_group', y='best_rmsd', ax=axes[1,0])
    axes[1,0].set_title('RMSD Distribution by Length Groups')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # TM-Score Distribution by Length Groups
    sns.boxplot(data=df, x='length_group', y='avg_tm_score', ax=axes[1,1])
    axes[1,1].set_title('TM-Score Distribution by Length Groups')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_PATH}/sequence_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(df, figsize=(12, 10)):
    """绘制指标相关性热力图"""
    # 选择数值型指标
    numeric_cols = ['sequence_length', 'best_rmsd', 'avg_tm_score', 'avg_lddt', 'avg_clash_score']
    correlation_data = df[numeric_cols].corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(correlation_data, dtype=bool))
    
    sns.heatmap(correlation_data, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Metrics Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_PATH}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_quality_assessment(df, figsize=(15, 10)):
    """质量评估可视化 - 使用直方图替代饼图"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 定义质量阈值
    rmsd_excellent = 2.0  # RMSD < 2.0 Å 认为是优秀
    rmsd_good = 4.0       # RMSD < 4.0 Å 认为是良好
    tm_excellent = 0.95   # TM-Score > 0.95 认为是优秀
    tm_good = 0.90        # TM-Score > 0.90 认为是良好
    
    total_samples = len(df)
    
    # RMSD质量分布 - 直方图形式
    rmsd_counts = [
        len(df[df['best_rmsd'] < rmsd_excellent]),
        len(df[(df['best_rmsd'] >= rmsd_excellent) & (df['best_rmsd'] < rmsd_good)]),
        len(df[df['best_rmsd'] >= rmsd_good])
    ]
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    labels = [f'Excellent (<{rmsd_excellent}A)', f'Good ({rmsd_excellent}-{rmsd_good}A)', f'Fair (>={rmsd_good}A)']
    
    # RMSD质量分布直方图
    bars1 = axes[0,0].bar(labels, rmsd_counts, color=colors, alpha=0.8, edgecolor='black')
    axes[0,0].set_title('RMSD Quality Distribution', fontweight='bold')
    axes[0,0].set_ylabel('Number of Samples')
    
    # 在柱子上标注数量和百分比
    for bar, count in zip(bars1, rmsd_counts):
        percentage = (count / total_samples) * 100
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                      f'{count}\n({percentage:.1f}%)', 
                      ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3, axis='y')
    
    # TM-Score质量分布 - 直方图形式
    tm_counts = [
        len(df[df['avg_tm_score'] > tm_excellent]),
        len(df[(df['avg_tm_score'] <= tm_excellent) & (df['avg_tm_score'] > tm_good)]),
        len(df[df['avg_tm_score'] <= tm_good])
    ]
    
    tm_labels = [f'Excellent (>{tm_excellent})', f'Good ({tm_good}-{tm_excellent})', f'Fair (<={tm_good})']
    
    bars2 = axes[0,1].bar(tm_labels, tm_counts, color=colors, alpha=0.8, edgecolor='black')
    axes[0,1].set_title('TM-Score Quality Distribution', fontweight='bold')
    axes[0,1].set_ylabel('Number of Samples')
    
    # 在柱子上标注数量和百分比
    for bar, count in zip(bars2, tm_counts):
        percentage = (count / total_samples) * 100
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                      f'{count}\n({percentage:.1f}%)', 
                      ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # RMSD Distribution Histogram
    axes[1,0].hist(df['best_rmsd'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,0].axvline(rmsd_excellent, color='green', linestyle='--', linewidth=2, label=f'Excellent Threshold ({rmsd_excellent}A)')
    axes[1,0].axvline(rmsd_good, color='orange', linestyle='--', linewidth=2, label=f'Good Threshold ({rmsd_good}A)')
    axes[1,0].set_xlabel('RMSD (Angstrom)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('RMSD Distribution Histogram')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # TM-Score Distribution Histogram
    axes[1,1].hist(df['avg_tm_score'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1,1].axvline(tm_excellent, color='green', linestyle='--', linewidth=2, label=f'Excellent Threshold ({tm_excellent})')
    axes[1,1].axvline(tm_good, color='orange', linestyle='--', linewidth=2, label=f'Good Threshold ({tm_good})')
    axes[1,1].set_xlabel('TM-Score')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('TM-Score Distribution Histogram')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_PATH}/quality_assessment.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_summary(df, figsize=(16, 6)):
    """性能总结图表"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 总体成功率
    total_samples = len(df)
    success_rate = len(df) / total_samples * 100  # 这里df已经是成功的样本
    
    # 创建成功率饼图
    sizes = [success_rate, 100-success_rate] if success_rate < 100 else [100]
    labels = ['Success', 'Failed'] if success_rate < 100 else ['Success']
    colors = ['#2ecc71', '#e74c3c'] if success_rate < 100 else ['#2ecc71']
    
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Overall Success Rate', fontweight='bold')
    
    # Performance metrics data
    metrics_means = {
        'RMSD (norm.)': 1 - (df['best_rmsd'].mean() / df['best_rmsd'].max()),  # 反向归一化，值越大越好
        'TM-Score': df['avg_tm_score'].mean(),
        'LDDT': df['avg_lddt'].mean() / 100,  # 归一化到0-1
        'Clash Score (norm.)': 1 - (df['avg_clash_score'].mean() / df['avg_clash_score'].max())  # 反向归一化
    }
    
    # 性能指标条形图
    metrics_names = list(metrics_means.keys())
    metrics_values = list(metrics_means.values())
    
    bars = axes[1].bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    axes[1].set_title('Normalized Performance Metrics', fontweight='bold')
    axes[1].set_ylabel('Normalized Score')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    
    # 在柱子上添加数值标签
    for bar, value in zip(bars, metrics_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 样本分布统计
    length_stats = df['sequence_length'].describe()
    
    stats_text = f"""
Sample Statistics:
• Total Samples: {len(df)}
• Sequence Length:
  - Min: {int(length_stats['min'])}
  - Max: {int(length_stats['max'])}
  - Mean: {length_stats['mean']:.1f}
  - Median: {length_stats['50%']:.1f}

• RMSD Statistics:
  - Mean: {df['best_rmsd'].mean():.3f} A
  - Median: {df['best_rmsd'].median():.3f} A
  - Best: {df['best_rmsd'].min():.3f} A

• TM-Score Statistics:
  - Mean: {df['avg_tm_score'].mean():.3f}
  - Median: {df['avg_tm_score'].median():.3f}
  - Best: {df['avg_tm_score'].max():.3f}
"""
    
    axes[2].text(0.05, 0.95, stats_text, transform=axes[2].transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].axis('off')
    axes[2].set_title('Statistical Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_PATH}/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    # 确保保存目录存在
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # 加载数据
    csv_file = LOAD_PATH
    df_all, df_success = load_and_clean_data(csv_file)
    
    print("\n开始生成可视化图表...")
    
    # 生成各种可视化
    print("1. 生成指标分布图（小提琴图+箱线图）...")
    plot_metric_distributions(df_success)
    
    print("1b. 生成单独的小提琴图...")
    plot_individual_violin_plots(df_success)
    
    print("2. 生成序列长度分析图...")
    plot_sequence_length_analysis(df_success)
    
    print("3. 生成相关性热力图...")
    plot_correlation_matrix(df_success)
    
    print("4. 生成质量评估图...")
    plot_quality_assessment(df_success)
    
    print("5. 生成性能总结图...")
    plot_performance_summary(df_success)
    
    print("\n✅ 所有可视化图表已生成完成！")
    print("📁 生成的图片文件（位于 plots/ 文件夹）:")
    print("   • metric_distributions.png - 指标分布图（综合）")
    print("   • rmsd_violin_plot.png - RMSD分布（单独小提琴图）")
    print("   • tm_score_violin_plot.png - TM-Score分布（单独小提琴图）")
    print("   • lddt_violin_plot.png - LDDT分布（单独小提琴图）")
    print("   • clash_score_violin_plot.png - Clash Score分布（单独小提琴图）")
    print("   • evaluated_atoms_violin_plot.png - 评估原子数分布（单独小提琴图）")
    print("   • sequence_length_analysis.png - 原子数量分析")
    print("   • correlation_matrix.png - 相关性热力图")
    print("   • quality_assessment.png - 质量评估图")
    print("   • performance_summary.png - 性能总结图")

if __name__ == "__main__":
    main()
