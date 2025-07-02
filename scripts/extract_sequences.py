#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从PDB文件中提取RNA序列信息
"""

import os
import glob
from collections import OrderedDict
import re

def extract_sequence_from_pdb(pdb_file):
    """
    从PDB文件中提取RNA序列
    """
    sequences = {}
    
    try:
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        
        current_chain = None
        residue_dict = {}
        
        for line in lines:
            if line.startswith('ATOM'):
                parts = line.split()
                if len(parts) >= 6:
                    # 提取残基信息
                    residue_name = parts[3]  # 核苷酸类型 (A, U, G, C)
                    chain_id = parts[4] if len(parts) > 4 else 'A'  # 链ID
                    residue_num = parts[5] if len(parts) > 5 else parts[4]  # 残基编号
                    
                    # 尝试从不同位置提取残基编号
                    try:
                        residue_num = int(residue_num)
                    except ValueError:
                        # 如果解析失败，尝试从其他位置提取
                        try:
                            residue_num = int(parts[6])
                        except (ValueError, IndexError):
                            continue
                    
                    # 验证是否为有效的核苷酸
                    if residue_name in ['A', 'U', 'G', 'C', 'T']:
                        # T转换为U（DNA->RNA）
                        if residue_name == 'T':
                            residue_name = 'U'
                        
                        if chain_id not in residue_dict:
                            residue_dict[chain_id] = {}
                        
                        residue_dict[chain_id][residue_num] = residue_name
        
        # 构建序列
        for chain_id, residues in residue_dict.items():
            if residues:
                # 按残基编号排序
                sorted_residues = sorted(residues.items())
                sequence = ''.join([res[1] for res in sorted_residues])
                
                if sequence:  # 确保序列不为空
                    sequences[chain_id] = sequence
    
    except Exception as e:
        print(f"解析文件 {pdb_file} 时出错: {e}")
        return {}
    
    return sequences

def write_individual_fasta(pdb_name, sequences, output_dir):
    """
    为单个PDB文件写入FASTA文件
    """
    if not sequences:
        return None
    
    # 创建输出文件名
    output_file = os.path.join(output_dir, f"{pdb_name}.fasta")
    
    with open(output_file, 'w') as f:
        for chain_id, sequence in sequences.items():
            if len(sequences) == 1:
                # 如果只有一条链，使用PDB名称作为标识符
                header = f">{pdb_name}"
            else:
                # 如果有多条链，使用PDB名称_链ID作为标识符
                header = f">{pdb_name}_{chain_id}"
            f.write(f"{header}\n")
            f.write(f"{sequence}\n")
    
    return output_file

def main():
    # PDB文件目录
    pdb_dir = "/home/wdcyx/rhofold/RNA3D_DATA/pdb"
    output_dir = "/home/wdcyx/rhofold/RNA3D_DATA/sequences"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有PDB文件
    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    print(f"找到 {len(pdb_files)} 个PDB文件")
    print(f"输出目录: {output_dir}")
    
    processed_count = 0
    total_sequences = 0
    
    for pdb_file in sorted(pdb_files):
        pdb_name = os.path.basename(pdb_file).replace('.pdb', '')
        print(f"正在处理: {pdb_name}")
        
        sequences = extract_sequence_from_pdb(pdb_file)
        
        if sequences:
            output_file = write_individual_fasta(pdb_name, sequences, output_dir)
            if output_file:
                processed_count += 1
                total_sequences += len(sequences)
                print(f"  提取到 {len(sequences)} 条链:")
                for chain_id, seq in sequences.items():
                    print(f"    链 {chain_id}: {len(seq)} 个核苷酸")
                print(f"  保存到: {os.path.basename(output_file)}")
        else:
            print(f"  未能从 {pdb_name} 提取到序列")
    
    print(f"\n处理完成!")
    print(f"成功处理 {processed_count} 个PDB文件")
    print(f"总计提取 {total_sequences} 条序列链")
    print(f"序列文件保存在: {output_dir}")

if __name__ == "__main__":
    main() 