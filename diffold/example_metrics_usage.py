"""
RNA结构评估指标示例代码
演示如何使用RNA专用的TM-score、lDDT和clash score等指标
"""

import torch
import numpy as np
from advanced_optimizers import RNAEvaluationMetrics

def generate_sample_rna_coordinates(batch_size=2, n_atoms=100, noise_level=0.5):
    """生成示例RNA坐标数据"""
    # 生成目标坐标（模拟真实RNA结构）
    target_coords = torch.randn(batch_size, n_atoms, 3) * 10.0
    
    # 生成预测坐标（添加一些噪声模拟预测误差）
    noise = torch.randn_like(target_coords) * noise_level
    predicted_coords = target_coords + noise
    
    return predicted_coords, target_coords

def demo_rna_evaluation_metrics():
    """演示RNA评估指标的使用"""
    print("=== RNA结构评估指标演示 ===\n")
    
    # 创建RNA专用评估器
    evaluator = RNAEvaluationMetrics()
    
    # 生成示例数据
    batch_size = 3
    n_atoms = 50
    
    print(f"生成示例RNA数据: batch_size={batch_size}, n_atoms={n_atoms}")
    
    # 模拟多个batch的评估
    for i in range(3):
        pred_coords, target_coords = generate_sample_rna_coordinates(
            batch_size=batch_size, 
            n_atoms=n_atoms, 
            noise_level=0.3 + i * 0.2  # 逐渐增加噪声
        )
        
        print(f"\n--- Batch {i+1} (噪声水平: {0.3 + i * 0.2:.1f}) ---")
        print(f"预测坐标形状: {pred_coords.shape}")
        print(f"目标坐标形状: {target_coords.shape}")
        
        # 更新指标
        evaluator.update(
            loss=0.5 + i * 0.1,  # 模拟损失
            batch_size=batch_size,
            predicted_coords=pred_coords,
            target_coords=target_coords,
            confidence_scores=torch.rand(batch_size, n_atoms) * 100  # 模拟置信度分数
        )
    
    # 计算最终指标
    metrics = evaluator.compute_metrics()
    
    print("\n=== RNA结构评估结果 ===")
    
    # 打印各种指标
    rna_metrics = [
        ('RMSD', 'rmsd', 'Å', '越低越好'),
        ('RNA TM-score', 'tm_score', '0-1', '越高越好 (≥0.45为同家族)'),
        ('RNA lDDT', 'lddt', '0-100', '越高越好 (≥70为高质量)'),
        ('RNA Clash Score', 'clash_score', '%', '越低越好 (≤5%为低冲突)')
    ]
    
    for metric_name, key, unit, direction in rna_metrics:
        if f'avg_{key}' in metrics:
            avg_val = metrics[f'avg_{key}']
            median_val = metrics.get(f'median_{key}', 'N/A')
            std_val = metrics.get(f'std_{key}', 'N/A')
            
            print(f"\n{metric_name}:")
            print(f"  平均值: {avg_val:.3f} {unit}")
            if median_val != 'N/A':
                print(f"  中位数: {median_val:.3f} {unit}")
            if std_val != 'N/A':
                print(f"  标准差: {std_val:.3f} {unit}")
            print(f"  评价: {direction}")
    
    # RNA特有的质量统计
    print(f"\n=== RNA质量统计 ===")
    if 'tm_score_good_ratio' in metrics:
        print(f"TM-score ≥ 0.45 (同家族质量): {metrics['tm_score_good_ratio']:.1%}")
    if 'tm_score_excellent_ratio' in metrics:
        print(f"TM-score ≥ 0.6 (优秀质量): {metrics['tm_score_excellent_ratio']:.1%}")
    if 'lddt_high_quality_ratio' in metrics:
        print(f"lDDT ≥ 70 (高质量): {metrics['lddt_high_quality_ratio']:.1%}")
    if 'lddt_good_quality_ratio' in metrics:
        print(f"lDDT ≥ 50 (良好质量): {metrics['lddt_good_quality_ratio']:.1%}")
    if 'clash_low_ratio' in metrics:
        print(f"冲突率 ≤ 5% (低冲突): {metrics['clash_low_ratio']:.1%}")
    
    # 打印其他信息
    print(f"\n其他信息:")
    print(f"  结构类型: {metrics['structure_type']}")
    print(f"  总样本数: {metrics['total_samples']}")
    print(f"  批次数: {metrics['batch_count']}")
    print(f"  平均损失: {metrics['avg_loss']:.3f}")
    
    if 'avg_confidence' in metrics:
        print(f"  平均置信度: {metrics['avg_confidence']:.1f}")

def compare_different_rna_predictions():
    """比较不同质量RNA预测的指标差异"""
    print("\n\n=== 不同质量RNA预测的比较 ===")
    
    batch_size = 2
    n_atoms = 100
    
    # 生成目标坐标
    target_coords = torch.randn(batch_size, n_atoms, 3) * 10.0
    
    # 创建不同质量的预测
    predictions = {
        "高质量预测": target_coords + torch.randn_like(target_coords) * 0.1,
        "中等质量预测": target_coords + torch.randn_like(target_coords) * 0.5,
        "低质量预测": target_coords + torch.randn_like(target_coords) * 2.0,
    }
    
    results = {}
    
    for pred_name, pred_coords in predictions.items():
        evaluator = RNAEvaluationMetrics()
        evaluator.update(
            loss=1.0,
            batch_size=batch_size,
            predicted_coords=pred_coords,
            target_coords=target_coords
        )
        results[pred_name] = evaluator.compute_metrics()
    
    # 打印比较结果
    print(f"{'指标':<15} {'高质量':<12} {'中等质量':<12} {'低质量':<12}")
    print("-" * 60)
    
    metrics_to_compare = [
        'avg_rmsd', 'avg_tm_score', 'avg_lddt', 'avg_clash_score'
    ]
    
    for metric in metrics_to_compare:
        if metric in results["高质量预测"]:
            high = results["高质量预测"][metric]
            medium = results["中等质量预测"][metric]
            low = results["低质量预测"][metric]
            
            metric_name = metric.replace('avg_', '').upper()
            print(f"{metric_name:<15} {high:<12.3f} {medium:<12.3f} {low:<12.3f}")
    
    # 打印质量比例
    print(f"\n{'质量比例':<15} {'高质量':<12} {'中等质量':<12} {'低质量':<12}")
    print("-" * 60)
    
    quality_metrics = [
        'tm_score_good_ratio', 'lddt_high_quality_ratio', 'clash_low_ratio'
    ]
    
    for metric in quality_metrics:
        if metric in results["高质量预测"]:
            high = results["高质量预测"][metric]
            medium = results["中等质量预测"][metric]
            low = results["低质量预测"][metric]
            
            metric_name = metric.replace('_ratio', '').replace('_', ' ').title()
            print(f"{metric_name:<15} {high:<12.1%} {medium:<12.1%} {low:<12.1%}")

def show_rna_specific_features():
    """展示RNA专用的评估特性"""
    print("\n\n=== RNA专用评估特性 ===")
    
    # 创建一个RNA评估器来显示默认参数
    evaluator = RNAEvaluationMetrics()
    
    print("RNA专用参数设置:")
    print("  TM-score:")
    print("    - d0公式: 针对RNA的松散结构优化")
    print("    - 短序列平滑过渡")
    print("  lDDT:")
    print("    - inclusion_radius: 20.0Å (vs 蛋白质15.0Å)")
    print("    - cutoffs: [1.0, 2.0, 4.0, 8.0]Å (vs 蛋白质[0.5, 1.0, 2.0, 4.0]Å)")
    print("  Clash Score:")
    print("    - threshold: 2.5Å (vs 蛋白质2.0Å)")
    print("\nRNA质量阈值:")
    print("  TM-score ≥ 0.45: 同一Rfam家族相似性")
    print("  TM-score ≥ 0.6:  优秀预测质量")
    print("  lDDT ≥ 70:       高质量预测")
    print("  lDDT ≥ 50:       良好质量预测")
    print("  Clash ≤ 5%:      低冲突，物理合理")

if __name__ == "__main__":
    # 运行演示
    demo_rna_evaluation_metrics()
    compare_different_rna_predictions()
    show_rna_specific_features()
    
    print("\n=== RNA评估指标说明 ===")
    print("1. RMSD: Root Mean Square Deviation - 结构偏差的均方根")
    print("2. RNA TM-score: RNA专用Template Modeling score - RNA结构相似性评分")
    print("   - 使用RNA优化的d0归一化公式")
    print("   - ≥0.45: 同一Rfam家族的结构相似性")
    print("   - ≥0.6: 优秀的结构预测质量")
    print("3. RNA lDDT: RNA专用local Distance Difference Test - 局部距离差异测试")
    print("   - 使用更大的inclusion radius (20Å) 和距离阈值")
    print("   - ≥70: 高质量结构预测")
    print("   - ≥50: 良好质量结构预测")
    print("4. RNA Clash Score: RNA专用原子冲突分数 - 检测过于接近的原子对")
    print("   - 使用RNA优化的冲突阈值 (2.5Å)")
    print("   - ≤5%: 低冲突率，物理上合理的结构")
    print("\n✨ 所有指标均专门为RNA结构特性设计，无蛋白质相关代码") 