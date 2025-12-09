#!/usr/bin/env python3
"""
绘制多个模型的TM-score box plot脚本

支持两种CSV格式：
1. evaluation_results.csv格式：包含tm_score列（小写）
2. tm_*.csv格式：包含TM_Score列（大写）

用法：
    python plot_tm_score_boxplot.py file1.csv file2.csv ... [--output output.png]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def extract_tm_scores(csv_path):
    """
    从CSV文件中提取TM-score值
    
    参数:
        csv_path: CSV文件路径
        
    返回:
        (model_name, tm_scores): 模型名称和TM-score值列表
    """
    try:
        df = pd.read_csv(csv_path)
        
        # 检测格式并提取TM-score列
        if 'tm_score' in df.columns:
            # 格式1: evaluation_results.csv格式
            tm_scores = df['tm_score'].dropna().tolist()
        elif 'TM_Score' in df.columns:
            # 格式2: tm_*.csv格式
            tm_scores = df['TM_Score'].dropna().tolist()
        else:
            raise ValueError(f"无法在文件 {csv_path} 中找到tm_score或TM_Score列")
        
        # 从路径提取模型名称：
        #  - 对单模型评估: results/single_performance/af3/evaluation_results.csv -> af3
        #  - 对DiffFold:   results/single_diffold_output/evaluation_results_d0=5/evaluation_results.csv -> evaluation_results_d0=5
        path = Path(csv_path)
        model_name = path.parent.name or path.stem
        
        return model_name, tm_scores
    
    except Exception as e:
        print(f"错误：读取文件 {csv_path} 时出错: {e}")
        raise


def beautify_model_name(model_name):
    """
    美化模型名称显示
    """
    name_mapping = {
        # 单模型评估目录名
        'af3': 'AF3',
        'boltz': 'Boltz-1',
        'chai': 'Chai',
        'hf3': 'HF3',
        'nufold': 'NuFold',
        'rf2na': 'RF2NA',
        'rhofold': 'RhoFold+',
        'trrosetta': 'trRosettaRNA',
        # DiffFold 单模型评估目录
        'evaluation_results_d0=5': 'DiffFold',
        # 兼容旧的 tm_* 命名
        'tm_af3': 'AF3',
        'tm_boltz': 'Boltz-1',
        'tm_chai': 'Chai',
        'tm_hf3': 'HF3',
        'tm_nufold': 'NuFold',
        'tm_rf2na': 'RF2NA',
        'tm_rhofold': 'RhoFold+',
        'tm_trrosetta': 'trRosettaRNA',
    }
    return name_mapping.get(model_name, model_name.replace('tm_', '').replace('_', ' ').title())


def plot_boxplot(data_dict, output_path=None, title="Performance on Single RNA"):
    """
    绘制TM-score的box plot（参考专业论文样式）
    
    参数:
        data_dict: 字典，键为模型名称，值为TM-score列表
        output_path: 输出图片路径（可选）
        title: 图表标题
    """
    # 准备数据
    plot_data = []
    labels = []
    
    for model_name, scores in data_dict.items():
        plot_data.extend(scores)
        labels.extend([beautify_model_name(model_name)] * len(scores))
    
    df_plot = pd.DataFrame({
        'Model': labels,
        'TM-score': plot_data
    })
    
    # 设置matplotlib参数以获得更好的样式
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'figure.dpi': 300,
    })
    
    # 创建图形，使用更大的尺寸以获得更好的视觉效果
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 绘制box plot，使用更专业的样式（类似参考图的浅蓝色）
    box_plot = sns.boxplot(
        data=df_plot, 
        x='Model', 
        y='TM-score', 
        ax=ax,
        width=0.6,
        linewidth=1.5,
        fliersize=4,
        flierprops=dict(
            marker='o',
            markerfacecolor='gray',
            markeredgecolor='gray',
            markersize=4,
            alpha=0.6
        ),
        boxprops=dict(
            facecolor='#87CEEB',  # Sky blue
            edgecolor='#4682B4',  # Steel blue
            linewidth=1.5,
            alpha=0.8
        ),
        medianprops=dict(
            color='#1E90FF',  # Dodger blue
            linewidth=2
        ),
        whiskerprops=dict(
            color='#4682B4',
            linewidth=1.5
        ),
        capprops=dict(
            color='#4682B4',
            linewidth=1.5
        )
    )
    
    # 设置标题和标签
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=13, fontweight='medium')
    ax.set_ylabel('TM-Score', fontsize=13, fontweight='medium')
    ax.set_ylim(0, 1.05)
    
    # 设置y轴刻度
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11)
    
    # 旋转x轴标签以避免重叠，并设置字体大小
    plt.xticks(rotation=45, ha='right', fontsize=11)
    
    # 添加水平网格线（仅y轴方向）
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 设置背景色为白色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 调整布局，增加底部边距以容纳旋转的标签
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"图表已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_statistics(data_dict):
    """
    打印每个模型的统计信息
    
    参数:
        data_dict: 字典，键为模型名称，值为TM-score列表
    """
    print("\n" + "="*60)
    print("TM-score 统计信息")
    print("="*60)
    
    for model_name, scores in sorted(data_dict.items()):
        if len(scores) == 0:
            continue
        
        mean_score = sum(scores) / len(scores)
        median_score = sorted(scores)[len(scores) // 2]
        min_score = min(scores)
        max_score = max(scores)
        
        print(f"\n{model_name}:")
        print(f"  样本数: {len(scores)}")
        print(f"  均值:   {mean_score:.4f}")
        print(f"  中位数: {median_score:.4f}")
        print(f"  最小值: {min_score:.4f}")
        print(f"  最大值: {max_score:.4f}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='绘制多个模型的TM-score box plot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python plot_tm_score_boxplot.py file1.csv file2.csv file3.csv
  python plot_tm_score_boxplot.py *.csv --output tm_scores_boxplot.png
  python plot_tm_score_boxplot.py results/*/evaluation_results.csv --output comparison.png
        """
    )
    
    parser.add_argument(
        'csv_files',
        nargs='+',
        help='CSV文件路径（支持多个文件）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出图片路径（默认：显示图表）'
    )
    
    parser.add_argument(
        '--title',
        type=str,
        default='TM-score Box Plot',
        help='图表标题（默认：TM-score Box Plot）'
    )
    
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='不打印统计信息'
    )
    
    args = parser.parse_args()
    
    # 收集所有文件的TM-score数据
    data_dict = {}
    
    for csv_file in args.csv_files:
        if not os.path.exists(csv_file):
            print(f"警告：文件不存在，跳过: {csv_file}")
            continue
        
        try:
            model_name, tm_scores = extract_tm_scores(csv_file)
            if len(tm_scores) > 0:
                data_dict[model_name] = tm_scores
                print(f"成功读取 {csv_file}: {len(tm_scores)} 个TM-score值")
            else:
                print(f"警告：文件 {csv_file} 中没有有效的TM-score数据")
        except Exception as e:
            print(f"错误：处理文件 {csv_file} 时出错: {e}")
            continue
    
    if len(data_dict) == 0:
        print("错误：没有成功读取任何数据，无法绘制图表")
        return
    
    # 打印统计信息
    if not args.no_stats:
        print_statistics(data_dict)
    
    # 绘制box plot
    plot_boxplot(data_dict, output_path=args.output, title=args.title)


if __name__ == '__main__':
    main()

