#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理数据泄露：删除 fine_tuning_data 中与 benchmark_data/casp15 重复的文件
"""

import os
import sys


def main():
    # 需要删除的文件列表
    files_to_remove = [
        # 序列文件
        "fine_tuning_data/seq/7qr3_1_d_A.fasta",
        "fine_tuning_data/seq/8fza_1_a_A.fasta",
        "fine_tuning_data/seq/8s95_1_c_A.fasta",
        # MSA 文件
        "fine_tuning_data/rMSA/7qr3_1_d_A.a3m",
        "fine_tuning_data/rMSA/8fza_1_a_A.a3m",
        "fine_tuning_data/rMSA/8s95_1_c_A.a3m",
        # PDB 文件
        "fine_tuning_data/pdb/7qr3_1_d_A.pdb",
        "fine_tuning_data/pdb/8fza_1_a_A.pdb",
        "fine_tuning_data/pdb/8s95_1_c_A.pdb",
    ]
    
    print("=" * 80)
    print("数据泄露清理工具")
    print("=" * 80)
    print()
    print("将要删除以下文件：")
    print()
    
    # 检查文件是否存在
    existing_files = []
    missing_files = []
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            file_size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({file_size:,} bytes)")
        else:
            missing_files.append(file_path)
            print(f"  ⊘ {file_path} (不存在)")
    
    print()
    print("-" * 80)
    print(f"找到 {len(existing_files)} 个文件需要删除")
    if missing_files:
        print(f"有 {len(missing_files)} 个文件已经不存在")
    print("-" * 80)
    print()
    
    if not existing_files:
        print("没有文件需要删除。")
        return 0
    
    # 确认删除
    response = input("确定要删除这些文件吗？(yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("操作已取消。")
        return 1
    
    # 删除文件
    print()
    print("开始删除文件...")
    print()
    
    deleted_count = 0
    failed_count = 0
    
    for file_path in existing_files:
        try:
            os.remove(file_path)
            print(f"  ✓ 已删除: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ 删除失败: {file_path} - {e}")
            failed_count += 1
    
    print()
    print("=" * 80)
    print("清理完成")
    print("=" * 80)
    print(f"成功删除: {deleted_count} 个文件")
    if failed_count > 0:
        print(f"删除失败: {failed_count} 个文件")
    print("=" * 80)
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

