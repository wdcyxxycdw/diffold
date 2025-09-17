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
warnings.filterwarnings('ignore')

# 设置字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare data for both models"""
    
    # Load Diffold data
    diffold_path = "batch_inference_output/batch_inference_results.csv"
    diffold_df = pd.read_csv(diffold_path)
    diffold_df = diffold_df[diffold_df['status'] == 'success'].copy()
    
    # Select appropriate columns for Diffold data
    diffold_data = {
        'rmsd': diffold_df['best_rmsd'].values,
        'tm_score': diffold_df['avg_tm_score'].values,
        'lddt': diffold_df['avg_lddt'].values,
        'clash_score': diffold_df['avg_clash_score'].values,
        'model': ['Diffold'] * len(diffold_df)
    }
    
    # Load RhoFold data
    rhofold_path = "rhofold_test_output/rhofold_test_results.csv"
    rhofold_df = pd.read_csv(rhofold_path)
    rhofold_df = rhofold_df[rhofold_df['status'] == 'success'].copy()
    
    # Select columns for RhoFold data
    rhofold_data = {
        'rmsd': rhofold_df['rmsd'].values,
        'tm_score': rhofold_df['tm_score'].values,
        'lddt': rhofold_df['lddt'].values,
        'clash_score': rhofold_df['clash_score'].values,
        'model': ['RhoFold+'] * len(rhofold_df)
    }
    
    # Combine data
    combined_data = {}
    for key in ['rmsd', 'tm_score', 'lddt', 'clash_score']:
        combined_data[key] = np.concatenate([diffold_data[key], rhofold_data[key]])
    combined_data['model'] = diffold_data['model'] + rhofold_data['model']
    
    combined_df = pd.DataFrame(combined_data)
    
    print(f"Diffold samples: {len(diffold_df)}")
    print(f"RhoFold+ samples: {len(rhofold_df)}")
    print(f"Total combined samples: {len(combined_df)}")
    
    return combined_df, diffold_df, rhofold_df

def create_comparison_violin_plots(df, save_path="plots"):
    """Create comparison violin plots"""
    
    # Metrics information
    metrics_info = {
        'rmsd': {
            'label': 'RMSD (Angstrom)',
            'better': 'lower',
            'color_diffold': '#2ecc71',
            'color_rhofold': '#e74c3c'
        },
        'tm_score': {
            'label': 'TM-Score',
            'better': 'higher', 
            'color_diffold': '#3498db',
            'color_rhofold': '#f39c12'
        },
        'lddt': {
            'label': 'lDDT Score',
            'better': 'higher',
            'color_diffold': '#9b59b6',
            'color_rhofold': '#1abc9c'
        },
        'clash_score': {
            'label': 'Clash Score',
            'better': 'lower',
            'color_diffold': '#e67e22',
            'color_rhofold': '#34495e'
        }
    }
    
    # Create save directory
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (metric, info) in enumerate(metrics_info.items()):
        ax = axes[i]
        
        # Create violin plot
        violin_parts = ax.violinplot(
            [df[df['model'] == 'Diffold'][metric].dropna(), 
             df[df['model'] == 'RhoFold+'][metric].dropna()],
            positions=[0, 1],
            widths=0.6,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        
        # Set colors
        colors = [info['color_diffold'], info['color_rhofold']]
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.8)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
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
            'better': 'lower',
            'color_diffold': '#2ecc71',
            'color_rhofold': '#e74c3c'
        },
        'tm_score': {
            'label': 'TM-Score', 
            'better': 'higher',
            'color_diffold': '#3498db', 
            'color_rhofold': '#f39c12'
        },
        'lddt': {
            'label': 'lDDT Score', 
            'better': 'higher',
            'color_diffold': '#9b59b6',
            'color_rhofold': '#1abc9c'
        },
        'clash_score': {
            'label': 'Clash Score', 
            'better': 'lower',
            'color_diffold': '#e67e22',
            'color_rhofold': '#34495e'
        }
    }
    
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    for metric, info in metrics_info.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data
        diffold_data = df[df['model'] == 'Diffold'][metric].dropna()
        rhofold_data = df[df['model'] == 'RhoFold+'][metric].dropna()
        
        # Create violin plot
        violin_parts = ax.violinplot(
            [diffold_data, rhofold_data],
            positions=[0, 1],
            widths=0.7,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        
        # Set colors
        colors = [info['color_diffold'], info['color_rhofold']]
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.8)
            pc.set_edgecolor('black')
            pc.set_linewidth(2)
        
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
    print("           DIFFOLD vs RHOFOLD+ PERFORMANCE COMPARISON")
    print("="*60)
    
    metrics = ['rmsd', 'tm_score', 'lddt', 'clash_score']
    metric_names = ['RMSD (Angstrom)', 'TM-Score', 'lDDT Score', 'Clash Score']
    better_direction = ['lower', 'higher', 'higher', 'lower']
    
    for metric, name, direction in zip(metrics, metric_names, better_direction):
        diffold_values = df[df['model'] == 'Diffold'][metric].dropna()
        rhofold_values = df[df['model'] == 'RhoFold+'][metric].dropna()
        
        diffold_mean = diffold_values.mean()
        rhofold_mean = rhofold_values.mean()
        
        print(f"\n{name}:")
        print(f"  Diffold    - Mean: {diffold_mean:.4f}, Median: {diffold_values.median():.4f}")
        print(f"  RhoFold+   - Mean: {rhofold_mean:.4f}, Median: {rhofold_values.median():.4f}")
        
        if direction == 'lower':
            improvement = ((diffold_mean - rhofold_mean) / diffold_mean) * 100
            winner = "RhoFold+" if rhofold_mean < diffold_mean else "Diffold"
        else:
            improvement = ((rhofold_mean - diffold_mean) / diffold_mean) * 100
            winner = "RhoFold+" if rhofold_mean > diffold_mean else "Diffold"
        
        print(f"  Winner: {winner}")
        print(f"  Improvement: {abs(improvement):.2f}%")

def main():
    """Main function"""
    print("Loading data...")
    df, diffold_df, rhofold_df = load_and_prepare_data()
    
    print("\nGenerating comparison visualizations...")
    
    print("1. Creating comprehensive comparison violin plots...")
    create_comparison_violin_plots(df)
    
    print("2. Creating individual metric comparison plots...")
    create_individual_comparison_plots(df)
    
    print("3. Generating statistical summary...")
    generate_summary_statistics(df)
    
    print("\n✅ Comparison analysis completed!")
    print("📁 Generated image files:")
    print("   • plots/model_comparison_violin_plots.png - Comprehensive comparison")
    print("   • plots/rmsd_comparison.png - RMSD comparison")
    print("   • plots/tm_score_comparison.png - TM-Score comparison")
    print("   • plots/lddt_comparison.png - lDDT comparison")
    print("   • plots/clash_score_comparison.png - Clash Score comparison")

if __name__ == "__main__":
    main()
