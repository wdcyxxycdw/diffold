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
    从PDB文件中提取RNA序列（使用固定列宽格式，支持多模型PDB）
    """
    sequences = {}
    
    try:
        residue_dict = {}
        in_model = False
        model_count = 0
        
        with open(pdb_file, 'r') as f:
            for line in f:
                # 处理多模型PDB文件（NMR/EM ensemble）
                if line.startswith('MODEL'):
                    model_count += 1
                    if model_count == 1:
                        in_model = True
                    continue
                
                if line.startswith('ENDMDL'):
                    if model_count == 1:
                        in_model = False
                        # 第一个模型结束后直接退出
                        break
                    continue
                
                # 如果有MODEL标记，只处理第一个模型
                if model_count > 0 and not in_model:
                    continue
                
                # 只处理ATOM和HETATM记录
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                
                if len(line) < 54:  # 确保行足够长
                    continue
                
                # 使用固定列宽格式解析（PDB标准格式）
                try:
                    residue_name = line[17:20].strip()  # 残基名称
                    chain_id = line[21:22].strip()       # 链ID
                    residue_num_str = line[22:26].strip()  # 残基序号
                    
                    # 如果链ID为空，使用默认值
                    if not chain_id:
                        chain_id = 'A'
                    
                    # 解析残基序号
                    try:
                        residue_num = int(residue_num_str)
                    except ValueError:
                        continue
                    
                    # 验证是否为有效的核苷酸（RNA/DNA）
                    # 标准RNA碱基
                    valid_bases = ['A', 'C', 'G', 'U', 'T', 'I']
                    # 也支持一些常见修饰残基的简写
                    valid_modified = ['PSU', '1MA', '2MG', '5MC', '5MU', '7MG', 'H2U', 'M2G', 'OMC', 'OMG']
                    
                    if residue_name in valid_bases or residue_name in valid_modified:
                        # 标准化碱基名称
                        if residue_name == 'T':
                            residue_name = 'U'  # DNA的T转换为RNA的U
                        elif residue_name == 'PSU':
                            residue_name = 'U'  # 假尿嘧啶视为U
                        elif residue_name == 'I':
                            residue_name = 'G'  # 次黄嘌呤视为G
                        elif residue_name in valid_modified:
                            # 其他修饰残基根据前缀判断
                            if 'A' in residue_name or residue_name.startswith('A'):
                                residue_name = 'A'
                            elif 'C' in residue_name or residue_name.startswith('C'):
                                residue_name = 'C'
                            elif 'G' in residue_name or residue_name.startswith('G'):
                                residue_name = 'G'
                            elif 'U' in residue_name or residue_name.startswith('U'):
                                residue_name = 'U'
                            else:
                                continue  # 无法识别的修饰残基
                        
                        if chain_id not in residue_dict:
                            residue_dict[chain_id] = {}
                        
                        residue_dict[chain_id][residue_num] = residue_name
                
                except (IndexError, ValueError):
                    continue
        
        # 提示多模型信息
        if model_count > 1:
            print(f"  检测到多模型PDB（共{model_count}个模型），只提取第一个模型")
        
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
    # PDB文件目录（使用原始raw文件，会自动处理多模型）
    pdb_dir = "benchmark_data/casp15/pdb_raw"
    output_dir = "benchmark_data/casp15/seq"
    
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