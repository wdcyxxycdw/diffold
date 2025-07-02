#!/usr/bin/env python3
"""
MSA长度检查脚本
检查RNA3D_DATA数据集中MSA文件与对应序列文件的长度是否匹配
将长度不匹配的MSA文件移动到指定文件夹
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import defaultdict


def read_fasta_sequence(filepath: str) -> Optional[str]:
    """
    读取FASTA文件中的第一个序列
    
    Args:
        filepath: FASTA文件路径
        
    Returns:
        序列字符串，如果文件不存在或格式错误则返回None
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        sequence = ""
        for line in lines:
            line = line.strip()
            if line and not line.startswith('>'):
                sequence += line
        
        return sequence.upper() if sequence else None
        
    except Exception as e:
        print(f"❌ 读取文件 {filepath} 时出错: {e}")
        return None


def read_msa_query_sequence(filepath: str) -> Optional[str]:
    """
    读取MSA文件中的查询序列（第一个序列）
    
    Args:
        filepath: MSA文件路径
        
    Returns:
        查询序列字符串，如果文件不存在或格式错误则返回None
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        sequence = ""
        found_query = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if found_query:
                    # 已经找到第一个序列，退出
                    break
                elif 'query' in line.lower() or not found_query:
                    # 找到查询序列或第一个序列
                    found_query = True
            elif found_query and line:
                # 移除gap字符（-）来获取原始序列长度
                sequence += line.replace('-', '')
        
        return sequence.upper() if sequence else None
        
    except Exception as e:
        print(f"❌ 读取MSA文件 {filepath} 时出错: {e}")
        return None


def get_sequence_length(sequence: str) -> int:
    """获取序列长度（忽略非字母字符）"""
    if not sequence:
        return 0
    
    # 只计算字母字符
    return len([c for c in sequence if c.isalpha()])


def find_paired_files(data_dir: str) -> Dict[str, Tuple[str, str]]:
    """
    找到配对的seq和MSA文件
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        字典，键为文件basename，值为(seq_path, msa_path)元组
    """
    seq_dir = Path(data_dir) / "sequences"
    msa_dir = Path(data_dir) / "rMSA"
    
    if not seq_dir.exists():
        raise FileNotFoundError(f"序列目录不存在: {seq_dir}")
    if not msa_dir.exists():
        raise FileNotFoundError(f"MSA目录不存在: {msa_dir}")
    
    # 找到所有seq文件
    seq_files = {}
    for seq_file in seq_dir.glob("*.fasta"):
        basename = seq_file.stem  # 去掉扩展名
        seq_files[basename] = str(seq_file)
    
    # 找到所有MSA文件
    msa_files = {}
    for msa_file in msa_dir.glob("*.a3m"):
        basename = msa_file.stem  # 去掉扩展名
        msa_files[basename] = str(msa_file)
    
    # 找到配对的文件
    paired_files = {}
    for basename in seq_files:
        if basename in msa_files:
            paired_files[basename] = (seq_files[basename], msa_files[basename])
        else:
            print(f"⚠️  未找到对应的MSA文件: {basename}")
    
    # 检查孤立的MSA文件
    for basename in msa_files:
        if basename not in seq_files:
            print(f"⚠️  未找到对应的序列文件: {basename}")
    
    return paired_files


def check_length_mismatch(paired_files: Dict[str, Tuple[str, str]], 
                         tolerance: int = 0) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    """
    检查长度不匹配的文件对
    
    Args:
        paired_files: 配对文件字典
        tolerance: 允许的长度差异容忍度
        
    Returns:
        (mismatched_basenames, mismatch_details) 元组
        mismatched_basenames: 不匹配的文件basename列表
        mismatch_details: 详细信息列表，每项为(basename, seq_len, msa_len)
    """
    mismatched = []
    mismatch_details = []
    
    print("🔍 开始检查长度匹配...")
    print("-" * 80)
    
    for basename, (seq_path, msa_path) in paired_files.items():
        # 读取序列
        seq_sequence = read_fasta_sequence(seq_path)
        msa_sequence = read_msa_query_sequence(msa_path)
        
        if seq_sequence is None:
            print(f"❌ 无法读取序列文件: {basename}")
            continue
            
        if msa_sequence is None:
            print(f"❌ 无法读取MSA文件: {basename}")
            continue
        
        seq_len = get_sequence_length(seq_sequence)
        msa_len = get_sequence_length(msa_sequence)
        
        length_diff = abs(seq_len - msa_len)
        
        if length_diff > tolerance:
            print(f"❌ 长度不匹配: {basename} (seq: {seq_len}, msa: {msa_len}, diff: {length_diff})")
            mismatched.append(basename)
            mismatch_details.append((basename, seq_len, msa_len))
        else:
            print(f"✅ 长度匹配: {basename} (seq: {seq_len}, msa: {msa_len})")
    
    print("-" * 80)
    print(f"📊 检查完成: 总计 {len(paired_files)} 对文件，{len(mismatched)} 对不匹配")
    
    return mismatched, mismatch_details


def move_mismatched_files(data_dir: str, mismatched_basenames: List[str], 
                         target_dir: str = "mismatched_msa") -> int:
    """
    移动长度不匹配的MSA文件到目标目录
    
    Args:
        data_dir: 数据目录路径
        mismatched_basenames: 不匹配的文件basename列表
        target_dir: 目标目录名称
        
    Returns:
        成功移动的文件数量
    """
    if not mismatched_basenames:
        print("✅ 没有需要移动的文件")
        return 0
    
    msa_dir = Path(data_dir) / "rMSA"
    target_path = Path(data_dir) / target_dir
    
    # 创建目标目录
    target_path.mkdir(exist_ok=True)
    print(f"📁 创建目标目录: {target_path}")
    
    moved_count = 0
    
    print("\n🚚 开始移动文件...")
    print("-" * 80)
    
    for basename in mismatched_basenames:
        source_file = msa_dir / f"{basename}.a3m"
        target_file = target_path / f"{basename}.a3m"
        
        if source_file.exists():
            try:
                shutil.move(str(source_file), str(target_file))
                print(f"✅ 已移动: {basename}.a3m")
                moved_count += 1
            except Exception as e:
                print(f"❌ 移动失败: {basename}.a3m - {e}")
        else:
            print(f"⚠️  源文件不存在: {source_file}")
    
    print("-" * 80)
    print(f"📊 移动完成: 成功移动 {moved_count} 个文件到 {target_path}")
    
    return moved_count


def generate_report(mismatch_details: List[Tuple[str, int, int]], 
                   output_file: str = "mismatch_report.txt"):
    """生成详细的不匹配报告"""
    if not mismatch_details:
        print("✅ 没有长度不匹配的文件，无需生成报告")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("MSA长度不匹配报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总计不匹配文件数: {len(mismatch_details)}\n\n")
        f.write("详细信息:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'文件名':<20} {'序列长度':<10} {'MSA长度':<10} {'差异':<10}\n")
        f.write("-" * 60 + "\n")
        
        for basename, seq_len, msa_len in sorted(mismatch_details):
            diff = abs(seq_len - msa_len)
            f.write(f"{basename:<20} {seq_len:<10} {msa_len:<10} {diff:<10}\n")
        
        f.write("-" * 60 + "\n")
        
        # 统计信息
        diffs = [abs(seq_len - msa_len) for _, seq_len, msa_len in mismatch_details]
        f.write(f"\n统计信息:\n")
        f.write(f"平均差异: {sum(diffs)/len(diffs):.2f}\n")
        f.write(f"最大差异: {max(diffs)}\n")
        f.write(f"最小差异: {min(diffs)}\n")
    
    print(f"📄 已生成详细报告: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="检查MSA文件与序列文件的长度匹配")
    
    parser.add_argument("--data_dir", type=str, default="./RNA3D_DATA", 
                       help="数据目录路径 (默认: ./RNA3D_DATA)")
    parser.add_argument("--target_dir", type=str, default="mismatched_msa",
                       help="不匹配文件的目标目录名 (默认: mismatched_msa)")
    parser.add_argument("--tolerance", type=int, default=0,
                       help="允许的长度差异容忍度 (默认: 0)")
    parser.add_argument("--dry_run", action="store_true",
                       help="只检查不移动文件")
    parser.add_argument("--report", type=str, default="mismatch_report.txt",
                       help="报告文件名 (默认: mismatch_report.txt)")
    
    args = parser.parse_args()
    
    print("🧬 MSA长度检查工具")
    print("=" * 60)
    print(f"📁 数据目录: {args.data_dir}")
    print(f"📁 目标目录: {args.target_dir}")
    print(f"🔧 容忍度: {args.tolerance}")
    print(f"🚫 仅检查模式: {args.dry_run}")
    print("=" * 60)
    
    try:
        # 检查数据目录是否存在
        if not Path(args.data_dir).exists():
            raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")
        
        # 找到配对的文件
        print("🔍 查找配对文件...")
        paired_files = find_paired_files(args.data_dir)
        print(f"📊 找到 {len(paired_files)} 对配对文件")
        
        # 检查长度匹配
        mismatched_basenames, mismatch_details = check_length_mismatch(
            paired_files, args.tolerance
        )
        
        # 生成报告
        if mismatch_details:
            generate_report(mismatch_details, args.report)
        
        # 移动文件（如果不是dry run模式）
        if not args.dry_run and mismatched_basenames:
            moved_count = move_mismatched_files(
                args.data_dir, mismatched_basenames, args.target_dir
            )
            
            print("\n" + "=" * 60)
            print("✅ 处理完成!")
            print(f"📊 共检查 {len(paired_files)} 对文件")
            print(f"❌ 发现 {len(mismatched_basenames)} 个不匹配文件")
            print(f"🚚 成功移动 {moved_count} 个文件")
            if mismatch_details:
                print(f"📄 详细报告已保存至: {args.report}")
        elif args.dry_run:
            print("\n" + "=" * 60)
            print("🚫 仅检查模式 - 未移动文件")
            print(f"📊 共检查 {len(paired_files)} 对文件")
            print(f"❌ 发现 {len(mismatched_basenames)} 个不匹配文件")
            if mismatch_details:
                print(f"📄 详细报告已保存至: {args.report}")
        else:
            print("\n" + "=" * 60)
            print("✅ 所有文件长度匹配，无需处理!")
    
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 