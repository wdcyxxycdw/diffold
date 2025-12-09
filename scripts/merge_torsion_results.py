#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并所有模型的扭转角分析结果
"""

import pandas as pd
from pathlib import Path
import sys

def merge_torsion_results(results_dir: Path, output_file: Path):
    """合并所有模型的扭转角结果"""
    
    models = ['af3', 'boltz', 'chai', 'diffold', 'hf3', 'nufold', 'rf2na', 'rhofold', 'trrosetta']
    
    all_results = []
    
    for model in models:
        csv_file = results_dir / f"torsion_angles_{model}_mae_rmse.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df['Model'] = model
            all_results.append(df)
        else:
            print(f"警告: 文件不存在: {csv_file}")
    
    if not all_results:
        print("错误: 没有找到任何结果文件")
        sys.exit(1)
    
    # 合并所有结果
    merged_df = pd.concat(all_results, ignore_index=True)
    
    # 重新排列列的顺序
    merged_df = merged_df[['Model', 'Torsion_Angle', 'Count', 'MAE', 'RMSE']]
    
    # 保存结果
    merged_df.to_csv(output_file, index=False, float_format='%.2f')
    print(f"合并结果已保存到: {output_file}")
    
    # 打印汇总统计
    print("\n" + "="*80)
    print("各模型扭转角MAE汇总 (度)")
    print("="*80)
    
    # 创建透视表
    mae_pivot = merged_df.pivot_table(
        index='Model', 
        columns='Torsion_Angle', 
        values='MAE',
        aggfunc='mean'
    )
    print("\nMAE (平均绝对误差):")
    print(mae_pivot.to_string())
    
    print("\n" + "="*80)
    print("各模型扭转角RMSE汇总 (度)")
    print("="*80)
    
    rmse_pivot = merged_df.pivot_table(
        index='Model', 
        columns='Torsion_Angle', 
        values='RMSE',
        aggfunc='mean'
    )
    print("\nRMSE (均方根误差):")
    print(rmse_pivot.to_string())
    
    # 计算每个模型的平均MAE和RMSE
    print("\n" + "="*80)
    print("各模型平均MAE和RMSE")
    print("="*80)
    model_avg = merged_df.groupby('Model')[['MAE', 'RMSE']].mean().sort_values('MAE')
    print(model_avg.to_string())
    
    return merged_df

if __name__ == '__main__':
    results_dir = Path('results/torsion_angle_analysis')
    output_file = results_dir / 'torsion_angles_all_models_merged.csv'
    
    merge_torsion_results(results_dir, output_file)






