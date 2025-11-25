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

# 设置字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 统一的颜色配置
MODEL_COLORS = {
    'Diffold': {
        'violin': '#6baed6',      # 浅蓝色 (violin plot)
        'points': '#08519c',       # 深蓝色 (数据点)
        'edge': '#08306b'          # 更深蓝色 (边框)
    },
    'Rhofold+(unrelaxed)': {
        'violin': '#fd8d3c',       # 浅橙色 (violin plot)
        'points': '#d94801',       # 深橙色 (数据点)
        'edge': '#7f2704'          # 更深橙色 (边框)
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
    # Try 'best_rmsd' first, then 'avg_rmsd', fallback to 'rmsd' if not available
    if 'best_rmsd' in diffold_df.columns:
        rmsd_col = 'best_rmsd'
    elif 'avg_rmsd' in diffold_df.columns:
        rmsd_col = 'avg_rmsd'
    else:
        rmsd_col = 'rmsd'
    
    print(f"Using RMSD column for Diffold: {rmsd_col}")
    
    diffold_data = {
        'rmsd': diffold_df[rmsd_col].values,
        'tm_score': diffold_df['avg_tm_score'].values,
        'lddt': diffold_df['avg_lddt'].values,
        'clash_score': diffold_df['avg_clash_score'].values,
        'sample_name': diffold_df['sample_name'].values,
        'model': ['Diffold'] * len(diffold_df)
    }
    
    # Load RhoFold data
    print(f"Loading RhoFold data from: {rhofold_path}")
    rhofold_df = pd.read_csv(rhofold_path)
    rhofold_df = rhofold_df[rhofold_df['status'] == 'success'].copy()
    
    print(f"Available columns in RhoFold CSV: {list(rhofold_df.columns)}")
    
    # Select appropriate columns for RhoFold data
    # Try 'rmsd' first, then 'avg_rmsd'
    if 'rmsd' in rhofold_df.columns:
        rmsd_col = 'rmsd'
    else:
        rmsd_col = 'avg_rmsd'
    
    print(f"Using RMSD column for RhoFold: {rmsd_col}")
    
    rhofold_data = {
        'rmsd': rhofold_df[rmsd_col].values,
        'tm_score': rhofold_df['avg_tm_score'].values,
        'lddt': rhofold_df['avg_lddt'].values,
        'clash_score': rhofold_df['avg_clash_score'].values,
        'sample_name': rhofold_df['sample_name'].values,
        'model': ['Rhofold+(unrelaxed)'] * len(rhofold_df)
    }
    
    # Combine data
    combined_data = {}
    for key in ['rmsd', 'tm_score', 'lddt', 'clash_score', 'sample_name']:
        combined_data[key] = np.concatenate([diffold_data[key], rhofold_data[key]])
    combined_data['model'] = diffold_data['model'] + rhofold_data['model']
    
    combined_df = pd.DataFrame(combined_data)
    
    print(f"Diffold samples: {len(diffold_df)}")
    print(f"Rhofold+(unrelaxed) samples: {len(rhofold_df)}")
    print(f"Total combined samples: {len(combined_df)}")
    
    return combined_df, diffold_df, rhofold_df

def create_comparison_violin_plots(df, save_path="plots"):
    """Create comparison violin plots"""
    
    # Metrics information
    metrics_info = {
        'rmsd': {
            'label': 'RMSD (Angstrom)',
            'better': 'lower'
        },
        'tm_score': {
            'label': 'TM-Score',
            'better': 'higher'
        },
        'lddt': {
            'label': 'lDDT Score',
            'better': 'higher'
        },
        'clash_score': {
            'label': 'Clash Score',
            'better': 'lower'
        }
    }
    
    # Create save directory
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (metric, info) in enumerate(metrics_info.items()):
        ax = axes[i]
        
        # Prepare data
        diffold_data = df[df['model'] == 'Diffold'][metric].dropna()
        rhofold_data = df[df['model'] == 'Rhofold+(unrelaxed)'][metric].dropna()
        
        # Create violin plot
        violin_parts = ax.violinplot(
            [diffold_data, rhofold_data],
            positions=[0, 1],
            widths=0.6,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        
        # Set colors using unified color scheme
        colors = [MODEL_COLORS['Diffold']['violin'], MODEL_COLORS['Rhofold+(unrelaxed)']['violin']]
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add data points with jitter (more concentrated around center)
        np.random.seed(42)  # For reproducible jitter
        jitter_width = 0.08  # Reduced jitter width for better centering
        
        # Add Diffold data points with unified colors
        diffold_jitter = np.random.normal(0, jitter_width, size=len(diffold_data))
        ax.scatter(diffold_jitter, diffold_data, 
                  color=MODEL_COLORS['Diffold']['points'], 
                  s=60, alpha=0.8, 
                  edgecolors=MODEL_COLORS['Diffold']['edge'], 
                  linewidths=1.8, zorder=3)
        
        # Add RhoFold+ data points with unified colors
        rhofold_jitter = np.random.normal(1, jitter_width, size=len(rhofold_data))
        ax.scatter(rhofold_jitter, rhofold_data, 
                  color=MODEL_COLORS['Rhofold+(unrelaxed)']['points'], 
                  s=60, alpha=0.8, 
                  edgecolors=MODEL_COLORS['Rhofold+(unrelaxed)']['edge'], 
                  linewidths=1.8, zorder=3)
        
        # Set labels and title
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Diffold', 'RhoFold+'], fontsize=14, fontweight='bold')
        ax.set_ylabel(info['label'], fontsize=14, fontweight='bold')
        ax.set_title(f'{info["label"]} Comparison', fontsize=16, fontweight='bold', pad=20)
        
        # Keep y-axis ticks and values
        ax.tick_params(axis='y', labelsize=10)
        
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/model_comparison_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_individual_comparison_plots(df, save_path="plots"):
    """Create individual comparison plots for each metric"""
    
    metrics_info = {
        'rmsd': {
            'label': 'RMSD (Angstrom)', 
            'better': 'lower'
        },
        'tm_score': {
            'label': 'TM-Score', 
            'better': 'higher'
        },
        'lddt': {
            'label': 'lDDT Score', 
            'better': 'higher'
        },
        'clash_score': {
            'label': 'Clash Score', 
            'better': 'lower'
        }
    }
    
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
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(2)
        
        # Add data points with jitter (more concentrated around center)
        np.random.seed(42)  # For reproducible jitter
        jitter_width = 0.08  # Reduced jitter width for better centering
        
        # Add Diffold data points with unified colors
        diffold_jitter = np.random.normal(0, jitter_width, size=len(diffold_data))
        ax.scatter(diffold_jitter, diffold_data, 
                  color=MODEL_COLORS['Diffold']['points'], 
                  s=80, alpha=0.8, 
                  edgecolors=MODEL_COLORS['Diffold']['edge'], 
                  linewidths=2, zorder=3)
        
        # Add RhoFold+ data points with unified colors
        rhofold_jitter = np.random.normal(1, jitter_width, size=len(rhofold_data))
        ax.scatter(rhofold_jitter, rhofold_data, 
                  color=MODEL_COLORS['Rhofold+(unrelaxed)']['points'], 
                  s=80, alpha=0.8, 
                  edgecolors=MODEL_COLORS['Rhofold+(unrelaxed)']['edge'], 
                  linewidths=2, zorder=3)
        
        # Set labels
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Diffold', 'RhoFold+'], fontsize=16, fontweight='bold')
        ax.set_ylabel(f'{info["label"]}', fontsize=16, fontweight='bold')
        ax.set_title(f'{info["label"]} Comparison', fontsize=18, fontweight='bold', pad=25)
        
        # Keep y-axis ticks and values
        ax.tick_params(axis='y', labelsize=12)
        
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save figure
        filename = f'{save_path}/{metric}_comparison.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {filename}")
        plt.show()

def generate_summary_statistics(df):
    """Generate comparison statistics summary"""
    
    print("\n" + "="*60)
    print("      DIFFOLD vs RHOFOLD+(UNRELAXED) PERFORMANCE COMPARISON")
    print("="*60)
    
    metrics = ['rmsd', 'tm_score', 'lddt', 'clash_score']
    metric_names = ['RMSD (Angstrom)', 'TM-Score', 'lDDT Score', 'Clash Score']
    better_direction = ['lower', 'higher', 'higher', 'lower']
    
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
        description='Compare Diffold and RhoFold+ performance on CASP16 dataset'
    )
    parser.add_argument(
        '--diffold-path',
        type=str,
        default='casp16_eval_results/merged_results.csv',
        help='Path to Diffold results CSV file (default: casp16_eval_results/merged_results.csv)'
    )
    parser.add_argument(
        '--rhofold-path',
        type=str,
        default='casp16_rhofold_parallel_output/rhofold_merged_results.csv',
        help='Path to RhoFold results CSV file (default: casp16_rhofold_parallel_output/rhofold_merged_results.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='comparison_plots',
        help='Directory to save output plots (default: comparison_plots)'
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
    
    print("1. Creating comprehensive comparison violin plots...")
    create_comparison_violin_plots(df, save_path=args.output_dir)
    
    print("2. Creating individual metric comparison plots...")
    create_individual_comparison_plots(df, save_path=args.output_dir)
    
    print("3. Generating statistical summary...")
    generate_summary_statistics(df)
    
    print("\n✅ Comparison analysis completed!")
    print(f"📁 Generated image files in '{args.output_dir}/':")
    print("   • model_comparison_violin_plots.png - Comprehensive comparison")
    print("   • rmsd_comparison.png - RMSD comparison")
    print("   • tm_score_comparison.png - TM-Score comparison")
    print("   • lddt_comparison.png - lDDT comparison")
    print("   • clash_score_comparison.png - Clash Score comparison")

if __name__ == "__main__":
    main()
