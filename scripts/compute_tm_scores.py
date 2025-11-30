#!/usr/bin/env python3
"""
计算测试集与训练集之间的TM-score相似度

使用USalign比对测试集中每个PDB与训练集中所有PDB，
找出每个测试样本与训练集的最大TM-score（用于检测数据泄露）
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


def run_usalign(pdb1: str, pdb2: str, usalign_path: str = "USalign") -> Optional[float]:
    """
    运行USalign并返回TM-score
    
    参数:
        pdb1: 第一个PDB文件路径（测试集）
        pdb2: 第二个PDB文件路径（训练集）
        usalign_path: USalign可执行文件路径
    
    返回:
        float: TM-score（相对于第一个结构），如果失败返回None
    """
    try:
        # 运行USalign
        result = subprocess.run(
            [usalign_path, pdb1, pdb2],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return None
        
        # 解析输出获取TM-score
        # USalign输出格式示例：
        # TM-score= 0.xxxxx (normalized by length of Structure_1: L=xxx)
        output = result.stdout
        
        # 查找TM-score行（相对于第一个结构）
        for line in output.split('\n'):
            if 'TM-score=' in line and 'normalized by length of Structure_1' in line:
                match = re.search(r'TM-score=\s*([\d.]+)', line)
                if match:
                    return float(match.group(1))
        
        return None
        
    except subprocess.TimeoutExpired:
        print(f"  警告: USalign超时 {os.path.basename(pdb1)} vs {os.path.basename(pdb2)}", 
              file=sys.stderr)
        return None
    except Exception as e:
        print(f"  错误: {e}", file=sys.stderr)
        return None


def get_pdb_files(directory: Path) -> List[Path]:
    """获取目录中所有PDB文件"""
    pdb_files = list(directory.glob("*.pdb"))
    return sorted(pdb_files)


def compute_max_tm_scores(
    test_dir: Path,
    train_dir: Path,
    usalign_path: str = "USalign",
    output_file: Optional[str] = None
) -> Dict[str, Dict]:
    """
    计算测试集中每个样本与训练集的最大TM-score
    
    参数:
        test_dir: 测试集目录
        train_dir: 训练集目录
        usalign_path: USalign可执行文件路径
        output_file: 输出文件路径（可选）
    
    返回:
        Dict: 每个测试样本的统计信息
    """
    # 获取所有PDB文件
    test_pdbs = get_pdb_files(test_dir)
    train_pdbs = get_pdb_files(train_dir)
    
    if not test_pdbs:
        print(f"错误: 在 {test_dir} 中未找到PDB文件", file=sys.stderr)
        return {}
    
    if not train_pdbs:
        print(f"错误: 在 {train_dir} 中未找到PDB文件", file=sys.stderr)
        return {}
    
    print("=" * 80)
    print("TM-score 相似度分析")
    print("=" * 80)
    print(f"测试集目录: {test_dir}")
    print(f"训练集目录: {train_dir}")
    print(f"测试集样本数: {len(test_pdbs)}")
    print(f"训练集样本数: {len(train_pdbs)}")
    print(f"USalign路径: {usalign_path}")
    print("=" * 80)
    print()
    
    results = {}
    
    # 对每个测试样本
    for i, test_pdb in enumerate(test_pdbs, 1):
        test_name = test_pdb.stem
        print(f"[{i}/{len(test_pdbs)}] 处理: {test_name}")
        
        max_tm_score = 0.0
        max_tm_train = None
        all_scores = []
        
        # 与所有训练样本比对
        for j, train_pdb in enumerate(train_pdbs, 1):
            train_name = train_pdb.stem
            
            # 跳过同名文件（如果测试集和训练集有重叠）
            if test_name == train_name:
                print(f"  [{j}/{len(train_pdbs)}] 跳过 {train_name} (同名)")
                continue
            
            # 计算TM-score
            tm_score = run_usalign(str(test_pdb), str(train_pdb), usalign_path)
            
            if tm_score is not None:
                all_scores.append((train_name, tm_score))
                
                if tm_score > max_tm_score:
                    max_tm_score = tm_score
                    max_tm_train = train_name
                
                # 实时显示进度
                if j % 10 == 0 or j == len(train_pdbs):
                    print(f"  进度: {j}/{len(train_pdbs)}, 当前最大TM: {max_tm_score:.4f}", 
                          end='\r', flush=True)
        
        print()  # 换行
        
        if max_tm_train:
            print(f"  ✓ 最大TM-score: {max_tm_score:.4f} (与 {max_tm_train})")
            
            # 统计TM-score分布
            high_similarity = [s for s in all_scores if s[1] >= 0.5]
            if high_similarity:
                print(f"  ! 发现 {len(high_similarity)} 个高相似度样本 (TM>=0.5):")
                for name, score in sorted(high_similarity, key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    - {name}: {score:.4f}")
        else:
            print(f"  ✗ 未能计算TM-score")
        
        print()
        
        # 保存结果
        results[test_name] = {
            'max_tm_score': max_tm_score,
            'max_tm_train': max_tm_train,
            'all_scores': all_scores,
            'num_comparisons': len(all_scores),
            'high_similarity_count': len([s for s in all_scores if s[1] >= 0.5])
        }
    
    # 打印总结
    print("=" * 80)
    print("总结报告")
    print("=" * 80)
    print(f"{'测试样本':<20} {'最大TM-score':<15} {'最相似训练样本':<30}")
    print("-" * 80)
    
    for test_name in sorted(results.keys()):
        info = results[test_name]
        max_tm = info['max_tm_score']
        max_train = info['max_tm_train'] or 'N/A'
        
        # 根据TM-score添加警告标记
        warning = ""
        if max_tm >= 0.5:
            warning = " ⚠️ 高相似度"
        elif max_tm >= 0.3:
            warning = " ⚡ 中等相似度"
        
        print(f"{test_name:<20} {max_tm:<15.4f} {max_train:<30}{warning}")
    
    print("=" * 80)
    
    # 统计信息
    valid_results = [r for r in results.values() if r['max_tm_score'] > 0]
    if valid_results:
        avg_max_tm = sum(r['max_tm_score'] for r in valid_results) / len(valid_results)
        high_sim_count = sum(1 for r in valid_results if r['max_tm_score'] >= 0.5)
        medium_sim_count = sum(1 for r in valid_results if 0.3 <= r['max_tm_score'] < 0.5)
        
        print(f"\n统计信息:")
        print(f"  平均最大TM-score: {avg_max_tm:.4f}")
        print(f"  高相似度样本 (TM>=0.5): {high_sim_count}/{len(valid_results)}")
        print(f"  中等相似度样本 (0.3<=TM<0.5): {medium_sim_count}/{len(valid_results)}")
        print(f"  低相似度样本 (TM<0.3): {len(valid_results)-high_sim_count-medium_sim_count}/{len(valid_results)}")
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w') as f:
            f.write("测试样本\t最大TM-score\t最相似训练样本\t高相似度计数\n")
            for test_name in sorted(results.keys()):
                info = results[test_name]
                f.write(f"{test_name}\t{info['max_tm_score']:.4f}\t"
                       f"{info['max_tm_train'] or 'N/A'}\t"
                       f"{info['high_similarity_count']}\n")
        print(f"\n结果已保存到: {output_file}")
    
    return results


def check_usalign(usalign_path: str) -> bool:
    """检查USalign是否可用"""
    try:
        result = subprocess.run(
            [usalign_path, "-h"],
            capture_output=True,
            timeout=5
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description='计算测试集与训练集之间的TM-score相似度',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python compute_tm_scores.py --test_dir benchmark_data/casp16/pdb --train_dir data/training_set/pdb
  
  # 指定USalign路径
  python compute_tm_scores.py --test_dir test/ --train_dir train/ --usalign /path/to/USalign
  
  # 保存结果到文件
  python compute_tm_scores.py --test_dir test/ --train_dir train/ --output results.tsv

说明:
  - TM-score范围: 0-1，越高表示结构越相似
  - TM-score >= 0.5: 通常认为是相同的折叠类型
  - TM-score >= 0.3: 可能具有相似的拓扑结构
  - TM-score < 0.3: 结构不相似
"""
    )
    
    parser.add_argument(
        '--test_dir',
        type=str,
        required=True,
        help='测试集目录（包含PDB文件）'
    )
    
    parser.add_argument(
        '--train_dir',
        type=str,
        required=True,
        help='训练集目录（包含PDB文件）'
    )
    
    parser.add_argument(
        '--usalign',
        type=str,
        default='USalign',
        help='USalign可执行文件路径（默认: USalign，需在PATH中）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径（TSV格式）'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查USalign是否可用
    print("检查USalign...", end=' ', flush=True)
    if not check_usalign(args.usalign):
        print("✗")
        print(f"\n错误: USalign不可用 ({args.usalign})", file=sys.stderr)
        print("请确保USalign已安装并在PATH中，或使用 --usalign 指定路径", file=sys.stderr)
        print("\nUSalign下载地址: https://zhanggroup.org/US-align/", file=sys.stderr)
        return 1
    print("✓")
    
    # 检查目录
    test_dir = Path(args.test_dir)
    train_dir = Path(args.train_dir)
    
    if not test_dir.exists():
        print(f"错误: 测试集目录不存在: {test_dir}", file=sys.stderr)
        return 1
    
    if not train_dir.exists():
        print(f"错误: 训练集目录不存在: {train_dir}", file=sys.stderr)
        return 1
    
    # 计算TM-scores
    results = compute_max_tm_scores(
        test_dir=test_dir,
        train_dir=train_dir,
        usalign_path=args.usalign,
        output_file=args.output
    )
    
    if not results:
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

