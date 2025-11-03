#!/usr/bin/env python3
"""
创建样本列表文件的辅助脚本
支持从目录、多个fold、或自定义模式创建样本列表
"""

import argparse
from pathlib import Path
import os


def create_sample_list_from_fold(data_dir: str, fold: int, output_file: str):
    """从指定的fold创建样本列表"""
    fold_file = Path(data_dir) / "list" / f"valid_fold-{fold}"
    
    if not fold_file.exists():
        raise FileNotFoundError(f"Fold文件不存在: {fold_file}")
    
    with open(fold_file, 'r') as f:
        samples = [line.strip() for line in f if line.strip()]
    
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(f"{sample}\n")
    
    print(f"✅ 从 fold-{fold} 创建样本列表: {output_file}")
    print(f"   样本数量: {len(samples)}")


def create_sample_list_from_multiple_folds(data_dir: str, folds: list, output_file: str):
    """从多个fold创建样本列表"""
    all_samples = []
    
    for fold in folds:
        fold_file = Path(data_dir) / "list" / f"valid_fold-{fold}"
        if not fold_file.exists():
            print(f"⚠️  警告: Fold文件不存在，跳过: {fold_file}")
            continue
        
        with open(fold_file, 'r') as f:
            samples = [line.strip() for line in f if line.strip()]
            all_samples.extend(samples)
            print(f"   从 fold-{fold} 读取 {len(samples)} 个样本")
    
    # 去重
    all_samples = list(dict.fromkeys(all_samples))  # 保持顺序的去重
    
    with open(output_file, 'w') as f:
        for sample in all_samples:
            f.write(f"{sample}\n")
    
    print(f"✅ 从 {len(folds)} 个fold创建样本列表: {output_file}")
    print(f"   总样本数量: {len(all_samples)}")


def create_sample_list_from_dir(data_dir: str, output_file: str, pattern: str = "*.pkl"):
    """从数据目录自动发现样本"""
    data_path = Path(data_dir)
    
    # 查找所有匹配的文件
    sample_files = list(data_path.glob(pattern))
    
    if not sample_files:
        raise ValueError(f"在 {data_dir} 中未找到匹配 {pattern} 的文件")
    
    # 提取样本名称（去掉扩展名）
    samples = sorted([f.stem for f in sample_files])
    
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(f"{sample}\n")
    
    print(f"✅ 从目录创建样本列表: {output_file}")
    print(f"   扫描目录: {data_dir}")
    print(f"   样本数量: {len(samples)}")


def create_sample_list_from_pdb_dir(pdb_dir: str, output_file: str):
    """从PDB文件目录创建样本列表"""
    pdb_path = Path(pdb_dir)
    
    # 查找所有PDB文件
    pdb_files = list(pdb_path.glob("*.pdb"))
    
    if not pdb_files:
        raise ValueError(f"在 {pdb_dir} 中未找到PDB文件")
    
    # 提取样本名称（去掉 .pdb 扩展名）
    samples = sorted([f.stem for f in pdb_files])
    
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(f"{sample}\n")
    
    print(f"✅ 从PDB目录创建样本列表: {output_file}")
    print(f"   PDB目录: {pdb_dir}")
    print(f"   样本数量: {len(samples)}")


def merge_sample_lists(input_files: list, output_file: str, deduplicate: bool = True):
    """合并多个样本列表文件"""
    all_samples = []
    
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"⚠️  警告: 文件不存在，跳过: {input_file}")
            continue
        
        with open(input_file, 'r') as f:
            samples = [line.strip() for line in f if line.strip()]
            all_samples.extend(samples)
            print(f"   从 {input_file} 读取 {len(samples)} 个样本")
    
    if deduplicate:
        all_samples = list(dict.fromkeys(all_samples))  # 保持顺序的去重
        print(f"   去重后: {len(all_samples)} 个样本")
    
    with open(output_file, 'w') as f:
        for sample in all_samples:
            f.write(f"{sample}\n")
    
    print(f"✅ 合并样本列表: {output_file}")
    print(f"   总样本数量: {len(all_samples)}")


def main():
    parser = argparse.ArgumentParser(
        description="创建样本列表文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 从单个fold创建:
   python create_sample_list.py --mode fold --data_dir ./fine_tuning_data --fold 2 --output samples.txt

2. 从多个fold创建:
   python create_sample_list.py --mode multi_fold --data_dir ./fine_tuning_data --folds 0 1 2 3 4 --output all_samples.txt

3. 从数据目录自动发现:
   python create_sample_list.py --mode dir --data_dir ./fine_tuning_data --output samples.txt

4. 从PDB目录创建:
   python create_sample_list.py --mode pdb_dir --pdb_dir ./benchmark_data/casp15/pdb --output casp15_samples.txt

5. 合并多个样本列表:
   python create_sample_list.py --mode merge --input_files list1.txt list2.txt --output merged.txt
        """
    )
    
    parser.add_argument("--mode", required=True,
                       choices=['fold', 'multi_fold', 'dir', 'pdb_dir', 'merge'],
                       help="创建模式")
    parser.add_argument("--output", required=True,
                       help="输出文件路径")
    
    # Fold模式参数
    parser.add_argument("--data_dir", 
                       help="数据目录路径（fold, multi_fold, dir模式需要）")
    parser.add_argument("--fold", type=int,
                       help="验证集fold编号（fold模式需要）")
    parser.add_argument("--folds", type=int, nargs='+',
                       help="多个fold编号（multi_fold模式需要）")
    
    # 目录模式参数
    parser.add_argument("--pattern", default="*.pkl",
                       help="文件匹配模式（dir模式，默认: *.pkl）")
    
    # PDB目录模式参数
    parser.add_argument("--pdb_dir",
                       help="PDB文件目录路径（pdb_dir模式需要）")
    
    # 合并模式参数
    parser.add_argument("--input_files", nargs='+',
                       help="要合并的样本列表文件（merge模式需要）")
    parser.add_argument("--deduplicate", action='store_true', default=True,
                       help="合并时去重（默认: True）")
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'fold':
            if not args.data_dir or args.fold is None:
                parser.error("fold模式需要 --data_dir 和 --fold 参数")
            create_sample_list_from_fold(args.data_dir, args.fold, args.output)
        
        elif args.mode == 'multi_fold':
            if not args.data_dir or not args.folds:
                parser.error("multi_fold模式需要 --data_dir 和 --folds 参数")
            create_sample_list_from_multiple_folds(args.data_dir, args.folds, args.output)
        
        elif args.mode == 'dir':
            if not args.data_dir:
                parser.error("dir模式需要 --data_dir 参数")
            create_sample_list_from_dir(args.data_dir, args.output, args.pattern)
        
        elif args.mode == 'pdb_dir':
            if not args.pdb_dir:
                parser.error("pdb_dir模式需要 --pdb_dir 参数")
            create_sample_list_from_pdb_dir(args.pdb_dir, args.output)
        
        elif args.mode == 'merge':
            if not args.input_files:
                parser.error("merge模式需要 --input_files 参数")
            merge_sample_lists(args.input_files, args.output, args.deduplicate)
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

