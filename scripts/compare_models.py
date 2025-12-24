#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffold vs RhoFold Performance Comparison Visualization
Comparing RMSD, TM-Score, LDDT and Clash Score metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import argparse
warnings.filterwarnings('ignore')

# 设置字体和样式 - 学术论文标准
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.style.use('seaborn-v0_8-white')

# 统一的颜色配置 - 专业学术配色
MODEL_COLORS = {
    'Diffold': {
        'violin': '#4472C4',      # 专业蓝色
        'points': '#2E5090',       # 深蓝色
        'edge': '#1F3864'          # 深边框
    },
    'Rhofold+(unrelaxed)': {
        'violin': '#ED7D31',       # 专业橙色
        'points': '#C65911',       # 深橙色
        'edge': '#833C0B'          # 深边框
    }
}

def load_and_prepare_data(diffold_path, rhofold_path):
    """Load and prepare data for both models"""
    
    # Load Diffold data
    print(f"Loading Diffold data from: {diffold_path}")
    diffold_df = pd.read_csv(diffold_path)
    diffold_df = diffold_df[diffold_df['status'] == 'success'].copy()
    
    print(f"Available columns in Diffold CSV: {list(diffold_df.columns)}")
    
    # Select appropriate columns for Diffold data
    # Try new column names first (rmsd, tm_score), then fallback to old names (avg_*)
    rmsd_col = 'rmsd' if 'rmsd' in diffold_df.columns else 'avg_rmsd' if 'avg_rmsd' in diffold_df.columns else 'best_rmsd'
    tm_col = 'tm_score' if 'tm_score' in diffold_df.columns else 'avg_tm_score'
    lddt_col = 'lddt' if 'lddt' in diffold_df.columns else 'avg_lddt'
    clash_col = 'clash_score' if 'clash_score' in diffold_df.columns else 'avg_clash_score'
    
    print(f"Using columns for Diffold: RMSD={rmsd_col}, TM-score={tm_col}, lDDT={lddt_col}, Clash={clash_col}")
    
    diffold_data = {
        'rmsd': diffold_df[rmsd_col].values,
        'tm_score': diffold_df[tm_col].values,
        'lddt': diffold_df[lddt_col].values,
        'clash_score': diffold_df[clash_col].values,
        'sample_name': diffold_df['sample_name'].values,
        'model': ['Diffold'] * len(diffold_df)
    }
    
    # Load RhoFold data
    print(f"Loading RhoFold data from: {rhofold_path}")
    rhofold_df = pd.read_csv(rhofold_path)
    rhofold_df = rhofold_df[rhofold_df['status'] == 'success'].copy()
    
    print(f"Available columns in RhoFold CSV: {list(rhofold_df.columns)}")
    
    # Select appropriate columns for RhoFold data
    # Try new column names first (rmsd, tm_score), then fallback to old names (avg_*)
    rmsd_col = 'rmsd' if 'rmsd' in rhofold_df.columns else 'avg_rmsd'
    tm_col = 'tm_score' if 'tm_score' in rhofold_df.columns else 'avg_tm_score'
    lddt_col = 'lddt' if 'lddt' in rhofold_df.columns else 'avg_lddt'
    clash_col = 'clash_score' if 'clash_score' in rhofold_df.columns else 'avg_clash_score'
    
    print(f"Using columns for RhoFold: RMSD={rmsd_col}, TM-score={tm_col}, lDDT={lddt_col}, Clash={clash_col}")
    
    rhofold_data = {
        'rmsd': rhofold_df[rmsd_col].values,
        'tm_score': rhofold_df[tm_col].values,
        'lddt': rhofold_df[lddt_col].values,
        'clash_score': rhofold_df[clash_col].values,
        'sample_name': rhofold_df['sample_name'].values,
        'model': ['Rhofold+(unrelaxed)'] * len(rhofold_df)
    }
    
    # Check for new metrics (GDT-TS, GDT-HA, MaxSub)
    new_metrics = ['gdt_ts', 'gdt_ha', 'maxsub']
    available_new_metrics = []
    
    for metric in new_metrics:
        if metric in diffold_df.columns and metric in rhofold_df.columns:
            available_new_metrics.append(metric)
            diffold_data[metric] = diffold_df[metric].values
            rhofold_data[metric] = rhofold_df[metric].values
    
    # Combine data
    combined_data = {}
    base_keys = ['rmsd', 'tm_score', 'lddt', 'clash_score', 'sample_name']
    all_keys = base_keys + available_new_metrics
    
    for key in all_keys:
        if key in diffold_data and key in rhofold_data:
            combined_data[key] = np.concatenate([diffold_data[key], rhofold_data[key]])
    combined_data['model'] = diffold_data['model'] + rhofold_data['model']
    
    combined_df = pd.DataFrame(combined_data)
    
    if available_new_metrics:
        print(f"Found new metrics: {', '.join(available_new_metrics)}")
    
    print(f"Diffold samples: {len(diffold_df)}")
    print(f"Rhofold+(unrelaxed) samples: {len(rhofold_df)}")
    print(f"Total combined samples: {len(combined_df)}")
    
    return combined_df, diffold_df, rhofold_df

def create_comparison_violin_plots(df, save_path="plots"):
    """Create comparison violin plots"""
    
    # Metrics information - 只选择关键指标用于综合图
    selected_metrics = ['rmsd', 'tm_score', 'lddt', 'clash_score', 'gdt_ts']
    
    all_metrics_info = {
        'rmsd': {
            'label': 'RMSD (Å)',
            'better': 'lower'
        },
        'tm_score': {
            'label': 'TM-score',
            'better': 'higher'
        },
        'lddt': {
            'label': 'lDDT',
            'better': 'higher'
        },
        'clash_score': {
            'label': 'Clash score',
            'better': 'lower'
        },
        'gdt_ts': {
            'label': 'GDT-TS',
            'better': 'higher'
        }
    }
    
    # Filter to only available and selected metrics
    metrics_info = {k: v for k in selected_metrics for k, v in all_metrics_info.items() 
                   if k == k and k in df.columns}
    
    num_metrics = len(metrics_info)
    print(f"Creating comparison plots for {num_metrics} key metrics")
    
    # Create save directory
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Fixed layout for 5 metrics: 2 rows x 3 columns
    nrows, ncols = 2, 3
    figsize = (18, 10)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Panel labels
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    
    for i, (metric, info) in enumerate(metrics_info.items()):
        ax = axes[i]
        
        # Prepare data
        diffold_data = df[df['model'] == 'Diffold'][metric].dropna()
        rhofold_data = df[df['model'] == 'Rhofold+(unrelaxed)'][metric].dropna()
        
        # Create violin plot
        violin_parts = ax.violinplot(
            [diffold_data, rhofold_data],
            positions=[0, 1],
            widths=0.65,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        
        # Set colors using unified color scheme
        colors = [MODEL_COLORS['Diffold']['violin'], MODEL_COLORS['Rhofold+(unrelaxed)']['violin']]
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.0)
        
        # Add box plot overlay for better statistics visualization
        bp = ax.boxplot(
            [diffold_data, rhofold_data],
            positions=[0, 1],
            widths=0.15,
            patch_artist=False,
            showfliers=False,
            showcaps=False,
            whiskerprops=dict(linewidth=1.5, color='black'),
            boxprops=dict(linewidth=1.5, color='black'),
            medianprops=dict(linewidth=2.5, color='darkred')
        )
        
        # Set labels and title (更简洁)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Diffold', 'RhoFold+'], fontsize=12)
        ax.set_ylabel(info['label'], fontsize=13)
        ax.set_title(info["label"], fontsize=14, pad=15)
        
        # Add panel label
        if i < len(panel_labels):
            ax.text(-0.15, 1.05, panel_labels[i], transform=ax.transAxes,
                   fontsize=16, fontweight='bold', va='top', ha='right')
        
        # Keep y-axis ticks and values
        ax.tick_params(axis='y', labelsize=11)
        ax.tick_params(axis='x', labelsize=12)
        
        ax.grid(True, alpha=0.25, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/model_comparison_violin_plots.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {save_path}/model_comparison_violin_plots.png")
    plt.close()

def create_individual_comparison_plots(df, save_path="plots"):
    """Create individual comparison plots for each metric"""
    
    # All possible metrics
    all_metrics_info = {
        'rmsd': {
            'label': 'RMSD (Å)', 
            'better': 'lower'
        },
        'tm_score': {
            'label': 'TM-score', 
            'better': 'higher'
        },
        'gdt_ts': {
            'label': 'GDT-TS',
            'better': 'higher'
        },
        'gdt_ha': {
            'label': 'GDT-HA',
            'better': 'higher'
        },
        'maxsub': {
            'label': 'MaxSub',
            'better': 'higher'
        },
        'lddt': {
            'label': 'lDDT', 
            'better': 'higher'
        },
        'clash_score': {
            'label': 'Clash score', 
            'better': 'lower'
        }
    }
    
    # Filter to only available metrics
    metrics_info = {k: v for k, v in all_metrics_info.items() if k in df.columns}
    
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    for metric, info in metrics_info.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data
        diffold_data = df[df['model'] == 'Diffold'][metric].dropna()
        rhofold_data = df[df['model'] == 'Rhofold+(unrelaxed)'][metric].dropna()
        
        # Create violin plot
        violin_parts = ax.violinplot(
            [diffold_data, rhofold_data],
            positions=[0, 1],
            widths=0.7,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        
        # Set colors using unified color scheme
        colors = [MODEL_COLORS['Diffold']['violin'], MODEL_COLORS['Rhofold+(unrelaxed)']['violin']]
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.2)
        
        # Add box plot overlay for better statistics visualization
        bp = ax.boxplot(
            [diffold_data, rhofold_data],
            positions=[0, 1],
            widths=0.2,
            patch_artist=False,
            showfliers=False,
            showcaps=False,
            whiskerprops=dict(linewidth=2.0, color='black'),
            boxprops=dict(linewidth=2.0, color='black'),
            medianprops=dict(linewidth=3.0, color='darkred')
        )
        
        # Set labels (更学术化)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Diffold', 'RhoFold+'], fontsize=16)
        ax.set_ylabel(info["label"], fontsize=17)
        ax.set_title(info["label"], fontsize=19, pad=20)
        
        # Keep y-axis ticks and values
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=16)
        
        ax.grid(True, alpha=0.25, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Save figure
        filename = f'{save_path}/{metric}_comparison.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {filename}")
        plt.close()

def create_box_plot_comparison(df, save_path="plots"):
    """Create box plot comparison (论文标准格式)"""
    
    # 只选择关键指标用于综合图
    selected_metrics = ['rmsd', 'tm_score', 'lddt', 'clash_score', 'gdt_ts']
    
    all_metrics_info = {
        'rmsd': {
            'label': 'RMSD',
            'unit': 'Å',
            'better': 'lower'
        },
        'tm_score': {
            'label': 'TM-score',
            'unit': '',
            'better': 'higher'
        },
        'lddt': {
            'label': 'lDDT',
            'unit': '',
            'better': 'higher'
        },
        'clash_score': {
            'label': 'Clash score',
            'unit': '',
            'better': 'lower'
        },
        'gdt_ts': {
            'label': 'GDT-TS',
            'unit': '',
            'better': 'higher'
        }
    }
    
    # Filter to only available and selected metrics
    metrics_info = {k: v for k in selected_metrics for k, v in all_metrics_info.items() 
                   if k == k and k in df.columns}
    
    num_metrics = len(metrics_info)
    print(f"Creating box plot for {num_metrics} key metrics")
    
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Fixed layout for 5 metrics: 2 rows x 3 columns
    nrows, ncols = 2, 3
    figsize = (18, 10)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Panel labels
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    
    for i, (metric, info) in enumerate(metrics_info.items()):
        ax = axes[i]
        
        # Prepare data for box plot
        data_to_plot = [
            df[df['model'] == 'Diffold'][metric].dropna(),
            df[df['model'] == 'Rhofold+(unrelaxed)'][metric].dropna()
        ]
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, 
                       labels=['Diffold', 'RhoFold+'],
                       patch_artist=True,
                       widths=0.6,
                       showfliers=True,
                       flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, 
                                      linestyle='none', markeredgecolor='black', alpha=0.5, linewidth=0.5))
        
        # Set colors using unified color scheme
        colors = [MODEL_COLORS['Diffold']['violin'], MODEL_COLORS['Rhofold+(unrelaxed)']['violin']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
        
        # Style median lines
        for median in bp['medians']:
            median.set(color='black', linewidth=2.0)
        
        # Style whiskers and caps
        for whisker in bp['whiskers']:
            whisker.set(color='black', linewidth=1.2, linestyle='-')
        for cap in bp['caps']:
            cap.set(color='black', linewidth=1.2)
        
        # Set labels and title (更简洁)
        unit_str = f" ({info['unit']})" if info['unit'] else ""
        ax.set_ylabel(f'{info["label"]}{unit_str}', fontsize=13)
        ax.set_title(info["label"], fontsize=14, pad=15)
        
        # Add panel label
        if i < len(panel_labels):
            ax.text(-0.15, 1.05, panel_labels[i], transform=ax.transAxes,
                   fontsize=16, fontweight='bold', va='top', ha='right')
        
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, alpha=0.25, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/model_comparison_box_plots.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {save_path}/model_comparison_box_plots.png")
    plt.close()

def create_individual_box_plots(df, save_path="plots"):
    """Create individual box plots for each metric (论文标准格式)"""
    
    all_metrics_info = {
        'rmsd': {'label': 'RMSD', 'unit': 'Å', 'better': 'lower'},
        'tm_score': {'label': 'TM-score', 'unit': '', 'better': 'higher'},
        'gdt_ts': {'label': 'GDT-TS', 'unit': '', 'better': 'higher'},
        'gdt_ha': {'label': 'GDT-HA', 'unit': '', 'better': 'higher'},
        'maxsub': {'label': 'MaxSub', 'unit': '', 'better': 'higher'},
        'lddt': {'label': 'lDDT', 'unit': '', 'better': 'higher'},
        'clash_score': {'label': 'Clash score', 'unit': '', 'better': 'lower'}
    }
    
    metrics_info = {k: v for k, v in all_metrics_info.items() if k in df.columns}
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    for metric, info in metrics_info.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Prepare data
        data_to_plot = [
            df[df['model'] == 'Diffold'][metric].dropna(),
            df[df['model'] == 'Rhofold+(unrelaxed)'][metric].dropna()
        ]
        
        # Create box plot
        bp = ax.boxplot(data_to_plot,
                       labels=['Diffold', 'RhoFold+'],
                       patch_artist=True,
                       widths=0.55,
                       showfliers=True,
                       flierprops=dict(marker='o', markerfacecolor='gray', markersize=5,
                                      linestyle='none', markeredgecolor='black', alpha=0.5, linewidth=0.8))
        
        # Set colors
        colors = [MODEL_COLORS['Diffold']['violin'], MODEL_COLORS['Rhofold+(unrelaxed)']['violin']]
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        # Style median lines (黑色更专业)
        for median in bp['medians']:
            median.set(color='black', linewidth=2.5)
        
        # Style whiskers and caps
        for whisker in bp['whiskers']:
            whisker.set(color='black', linewidth=1.5, linestyle='-')
        for cap in bp['caps']:
            cap.set(color='black', linewidth=1.5)
        
        # Add grid
        ax.grid(True, alpha=0.25, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        
        # Labels (更简洁)
        unit_str = f" ({info['unit']})" if info['unit'] else ""
        ax.set_ylabel(f'{info["label"]}{unit_str}', fontsize=16)
        ax.set_title(info["label"], fontsize=18, pad=20)
        ax.tick_params(axis='both', labelsize=14)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        filename = f'{save_path}/{metric}_box_plot.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {filename}")
        plt.close()

def generate_summary_statistics(df):
    """Generate comparison statistics summary"""
    
    print("\n" + "="*70)
    print("      DIFFOLD vs RHOFOLD+(UNRELAXED) PERFORMANCE COMPARISON")
    print("="*70)
    
    # All possible metrics
    all_metrics = {
        'rmsd': ('RMSD (Å)', 'lower'),
        'tm_score': ('TM-score', 'higher'),
        'gdt_ts': ('GDT-TS', 'higher'),
        'gdt_ha': ('GDT-HA', 'higher'),
        'maxsub': ('MaxSub', 'higher'),
        'lddt': ('lDDT', 'higher'),
        'clash_score': ('Clash score', 'lower')
    }
    
    # Filter to available metrics
    metrics = []
    metric_names = []
    better_direction = []
    
    for metric, (name, direction) in all_metrics.items():
        if metric in df.columns:
            metrics.append(metric)
            metric_names.append(name)
            better_direction.append(direction)
    
    for metric, name, direction in zip(metrics, metric_names, better_direction):
        diffold_values = df[df['model'] == 'Diffold'][metric].dropna()
        rhofold_values = df[df['model'] == 'Rhofold+(unrelaxed)'][metric].dropna()
        
        diffold_mean = diffold_values.mean()
        rhofold_mean = rhofold_values.mean()
        
        print(f"\n{name}:")
        print(f"  Diffold               - Mean: {diffold_mean:.4f}, Median: {diffold_values.median():.4f}")
        print(f"  Rhofold+(unrelaxed)   - Mean: {rhofold_mean:.4f}, Median: {rhofold_values.median():.4f}")
        
        if direction == 'lower':
            improvement = ((diffold_mean - rhofold_mean) / diffold_mean) * 100
            winner = "Rhofold+(unrelaxed)" if rhofold_mean < diffold_mean else "Diffold"
        else:
            improvement = ((rhofold_mean - diffold_mean) / diffold_mean) * 100
            winner = "Rhofold+(unrelaxed)" if rhofold_mean > diffold_mean else "Diffold"
        
        print(f"  Winner: {winner}")
        print(f"  Improvement: {abs(improvement):.2f}%")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Compare Diffold and RhoFold+ performance on RNA benchmark dataset'
    )
    parser.add_argument(
        '--diffold-path',
        type=str,
        required=True,
        help='Path to Diffold evaluation results CSV file'
    )
    parser.add_argument(
        '--rhofold-path',
        type=str,
        required=True,
        help='Path to RhoFold evaluation results CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save output plots'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("  DIFFOLD vs RHOFOLD+ PERFORMANCE COMPARISON")
    print("="*70)
    print(f"\nDiffold data: {args.diffold_path}")
    print(f"RhoFold data: {args.rhofold_path}")
    print(f"Output directory: {args.output_dir}")
    print("\nLoading data...")
    
    df, diffold_df, rhofold_df = load_and_prepare_data(args.diffold_path, args.rhofold_path)
    
    print("\nGenerating comparison visualizations...")
    
    print("1. Creating box plots (论文标准格式)...")
    create_box_plot_comparison(df, save_path=args.output_dir)
    
    print("2. Creating individual box plots...")
    create_individual_box_plots(df, save_path=args.output_dir)
    
    print("3. Creating violin plots (详细分布)...")
    create_comparison_violin_plots(df, save_path=args.output_dir)
    
    print("4. Creating individual violin plots...")
    create_individual_comparison_plots(df, save_path=args.output_dir)
    
    print("5. Generating statistical summary...")
    generate_summary_statistics(df)
    
    print("\n✅ Comparison analysis completed!")
    print(f"📁 Generated image files in '{args.output_dir}/':")
    
    available_metrics = [col for col in df.columns if col in [
        'rmsd', 'tm_score', 'gdt_ts', 'gdt_ha', 'maxsub', 'lddt', 'clash_score'
    ]]
    
    print("\n  📊 Box Plots (论文标准格式):")
    print("   • model_comparison_box_plots.png - 综合箱线图 ⭐ 推荐")
    for metric in available_metrics:
        print(f"   • {metric}_box_plot.png")
    
    print("\n  🎻 Violin Plots (详细分布分析):")
    print("   • model_comparison_violin_plots.png - 综合小提琴图")
    for metric in available_metrics:
        print(f"   • {metric}_comparison.png")

if __name__ == "__main__":
    main()
