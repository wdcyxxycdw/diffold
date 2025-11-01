#!/usr/bin/env python3
"""
RNA数据长度筛选脚本
根据序列长度（16-256nt）筛选RNA3D_DATA中的数据，并使用CD-HIT进行序列去冗余
将符合条件的数据复制到processed_data文件夹，保持原有的目录结构
新增功能：根据缺失原子比例进行筛选
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import argparse
from collections import defaultdict
import re
from rhofold.utils.constants import ATOM_NAMES_PER_RESD


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


def read_seq_file_sequence(filepath: str) -> Optional[str]:
    """
    读取.seq文件中的序列
    
    Args:
        filepath: .seq文件路径
        
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
        print(f"❌ 读取seq文件 {filepath} 时出错: {e}")
        return None


def calculate_atom_completeness(pdb_filepath: str, sequence: str) -> Tuple[float, int, int]:
    """
    计算PDB文件中的原子完整性比例
    
    Args:
        pdb_filepath: PDB文件路径
        sequence: RNA序列字符串
        
    Returns:
        (完整性比例, 实际原子数, 预期原子数)
    """
    try:
        if not os.path.exists(pdb_filepath):
            return 0.0, 0, 0
        
        # 计算预期原子数（根据序列和每个核苷酸的原子定义）
        # 同时构建每个残基的标准原子集合
        expected_atoms = 0
        residue_standard_atoms = {}  # {残基索引: set(标准原子名)}
        for idx, residue in enumerate(sequence, start=1):
            if residue in ATOM_NAMES_PER_RESD:
                expected_atoms += len(ATOM_NAMES_PER_RESD[residue])
                residue_standard_atoms[idx] = set(ATOM_NAMES_PER_RESD[residue])
        
        if expected_atoms == 0:
            return 0.0, 0, 0
        
        # 读取PDB文件，统计实际存在的原子数
        with open(pdb_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        actual_atoms = 0
        seen_atoms = set()  # 用于记录已见过的原子（残基编号+原子名）
        
        for line in lines:
            line = line.strip()
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # 检查原子坐标是否完整（不是缺失的）
                if len(line) >= 54:  # 确保行足够长
                    try:
                        # 获取原子名称（第13-16列，不包括第17列的alternate location）
                        atom_name = line[12:16].strip()
                        
                        # 跳过氢原子（以H开头或包含H的原子名）
                        # 常见氢原子名：H, H1, H2, H3, H5', H5'', HO2', etc.
                        if atom_name.startswith('H') or atom_name.startswith('1H') or atom_name.startswith('2H') or atom_name.startswith('3H'):
                            continue
                        
                        # 获取残基序号（第23-26列）
                        try:
                            residue_num = int(line[22:26].strip())
                        except ValueError:
                            continue
                        
                        # 移除原子名中可能包含的A/B占位标记（处理alternate location）
                        clean_atom_name = atom_name.rstrip('AB').rstrip()  # 移除末尾的A或B和空格
                        
                        # 🔥 只统计标准原子：检查这个原子是否在该残基的标准原子列表中
                        if residue_num in residue_standard_atoms:
                            if clean_atom_name not in residue_standard_atoms[residue_num]:
                                # 跳过非标准原子（如OP3等）
                                continue
                        else:
                            # 残基编号不在序列范围内，跳过
                            continue
                        
                        # 获取残基信息（第18-26列：残基名+链ID+残基序号）
                        residue_info = line[17:26].strip()
                        atom_id = f"{residue_info}_{clean_atom_name}"
                        
                        # 如果这个原子已经记录过，跳过（处理alternate location的情况）
                        if atom_id in seen_atoms:
                            continue
                        
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        
                        # 检查坐标是否合理（不是缺失值）
                        # 缺失的原子通常用0.000或999.999表示
                        if (x != 0.0 or y != 0.0 or z != 0.0) and \
                           abs(x) < 999.0 and abs(y) < 999.0 and abs(z) < 999.0:
                            seen_atoms.add(atom_id)
                            actual_atoms += 1
                    except ValueError:
                        # 如果无法解析坐标，认为是缺失的
                        pass
        
        completeness = actual_atoms / expected_atoms if expected_atoms > 0 else 0.0
        return completeness, actual_atoms, expected_atoms
        
    except Exception as e:
        print(f"❌ 计算原子完整性时出错 {pdb_filepath}: {e}")
        return 0.0, 0, 0


def get_sequence_length(sequence: str) -> int:
    """获取序列长度（忽略非字母字符）"""
    if not sequence:
        return 0
    
    # 只计算字母字符
    return len([c for c in sequence if c.isalpha()])


def scan_all_sequences(data_dir: str, min_completeness: float = 0.0) -> Dict[str, Tuple[str, int, str, float, int, int]]:
    """
    扫描所有序列文件并获取长度信息和原子完整性
    
    Args:
        data_dir: 数据目录路径
        min_completeness: 最小原子完整性阈值
        
    Returns:
        字典，键为文件basename，值为(file_path, length, file_type, completeness, valid_atoms, total_atoms)元组
        file_type可能是'fasta'或'seq'
    """
    sequences_dir = Path(data_dir) / "sequences"
    seq_dir = Path(data_dir) / "seq"
    pdb_dir = Path(data_dir) / "pdb"
    
    all_sequences = {}
    
    # 处理sequences目录中的.fasta文件
    if sequences_dir.exists():
        print(f"🔍 扫描 {sequences_dir}...")
        for fasta_file in sequences_dir.glob("*.fasta"):
            basename = fasta_file.stem
            sequence = read_fasta_sequence(str(fasta_file))
            if sequence:
                length = get_sequence_length(sequence)
                
                # 检查对应的PDB文件
                pdb_file = pdb_dir / f"{basename}.pdb"
                completeness, valid_atoms, total_atoms = calculate_atom_completeness(str(pdb_file), sequence)
                
                all_sequences[basename] = (str(fasta_file), length, 'fasta', completeness, valid_atoms, total_atoms)
                print(f"  📄 {basename}: {length}nt, 完整性: {completeness:.1%} ({valid_atoms}/{total_atoms} 原子)")
            else:
                print(f"  ❌ 无法读取: {basename}")
    
    # 处理seq目录中的.seq和.fasta文件
    if seq_dir.exists():
        print(f"🔍 扫描 {seq_dir}...")
        # 先处理.seq文件
        for seq_file in seq_dir.glob("*.seq"):
            basename = seq_file.stem
            # 如果已经在fasta中找到了，跳过
            if basename in all_sequences:
                continue
                
            sequence = read_seq_file_sequence(str(seq_file))
            if sequence:
                length = get_sequence_length(sequence)
                
                # 检查对应的PDB文件
                pdb_file = pdb_dir / f"{basename}.pdb"
                completeness, valid_atoms, total_atoms = calculate_atom_completeness(str(pdb_file), sequence)
                
                all_sequences[basename] = (str(seq_file), length, 'seq', completeness, valid_atoms, total_atoms)
                print(f"  📄 {basename}: {length}nt, 完整性: {completeness:.1%} ({valid_atoms}/{total_atoms} 原子)")
            else:
                print(f"  ❌ 无法读取: {basename}")
        
        # 再处理.fasta文件
        for fasta_file in seq_dir.glob("*.fasta"):
            basename = fasta_file.stem
            # 如果已经在sequences或seq中找到了，跳过
            if basename in all_sequences:
                continue
                
            sequence = read_fasta_sequence(str(fasta_file))
            if sequence:
                length = get_sequence_length(sequence)
                
                # 检查对应的PDB文件
                pdb_file = pdb_dir / f"{basename}.pdb"
                completeness, valid_atoms, total_atoms = calculate_atom_completeness(str(pdb_file), sequence)
                
                all_sequences[basename] = (str(fasta_file), length, 'fasta', completeness, valid_atoms, total_atoms)
                print(f"  📄 {basename}: {length}nt, 完整性: {completeness:.1%} ({valid_atoms}/{total_atoms} 原子)")
            else:
                print(f"  ❌ 无法读取: {basename}")
    
    return all_sequences


def filter_by_length_and_completeness(sequences: Dict[str, Tuple[str, int, str, float, int, int]], 
                                    min_length: int = 16, 
                                    max_length: int = 256,
                                    min_completeness: float = 0.8,
                                    max_completeness: float = 1.0) -> Dict[str, Tuple[str, int, str, float, int, int]]:
    """
    根据长度范围和原子完整性筛选序列
    
    Args:
        sequences: 序列信息字典
        min_length: 最小长度
        max_length: 最大长度
        min_completeness: 最小原子完整性阈值
        max_completeness: 最大原子完整性阈值
        
    Returns:
        筛选后的序列字典
    """
    filtered = {}
    
    print(f"🔍 筛选条件:")
    print(f"  - 长度范围: {min_length}-{max_length}nt")
    print(f"  - 原子完整性: {min_completeness:.1%}-{max_completeness:.1%}")
    print("-" * 80)
    
    for basename, (file_path, length, file_type, completeness, actual_atoms, expected_atoms) in sequences.items():
        length_ok = min_length <= length <= max_length
        completeness_ok = min_completeness <= completeness <= max_completeness
        
        if length_ok and completeness_ok:
            filtered[basename] = (file_path, length, file_type, completeness, actual_atoms, expected_atoms)
            print(f"✅ {basename}: {length}nt, 完整性: {completeness:.1%} ({actual_atoms}/{expected_atoms}) - 符合条件")
        else:
            reasons = []
            if not length_ok:
                reasons.append(f"长度{length}nt超出范围")
            if not completeness_ok:
                reasons.append(f"完整性{completeness:.1%}低于阈值")
            print(f"❌ {basename}: {length}nt, 完整性: {completeness:.1%} ({actual_atoms}/{expected_atoms}) - {'; '.join(reasons)}")
    
    print("-" * 80)
    print(f"📊 筛选结果: 原始 {len(sequences)} 个，符合条件 {len(filtered)} 个")
    
    return filtered


def create_temp_fasta(filtered_sequences: Dict[str, Tuple[str, int, str, float, int, int]]) -> str:
    """
    创建临时FASTA文件，包含所有筛选后的序列
    
    Args:
        filtered_sequences: 筛选后的序列字典
        
    Returns:
        临时FASTA文件路径
    """
    temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
    
    print("📝 创建临时FASTA文件用于CD-HIT聚类...")
    
    for basename, (file_path, length, file_type, completeness, actual_atoms, expected_atoms) in filtered_sequences.items():
        # 读取序列内容
        if file_type == 'fasta':
            sequence = read_fasta_sequence(file_path)
        else:
            sequence = read_seq_file_sequence(file_path)
        
        if sequence:
            # 写入临时文件
            temp_fasta.write(f">{basename}\n{sequence}\n")
    
    temp_fasta.close()
    print(f"📝 临时FASTA文件已创建: {temp_fasta.name}")
    
    return temp_fasta.name


def run_cdhit_clustering(input_fasta: str, similarity_threshold: float = 0.8, 
                        threads: int = 4) -> Tuple[str, str]:
    """
    运行CD-HIT进行序列聚类
    
    Args:
        input_fasta: 输入FASTA文件路径
        similarity_threshold: 相似度阈值
        threads: 使用的线程数
        
    Returns:
        (输出文件路径, 聚类信息文件路径)
    """
    output_fasta = input_fasta + ".cdhit"
    cluster_file = output_fasta + ".clstr"
    
    print(f"🔬 开始CD-HIT序列聚类 (相似度阈值: {similarity_threshold*100}%)...")
    
    # 根据相似度阈值设置词长
    if similarity_threshold >= 0.9:
        word_length = 8
    elif similarity_threshold >= 0.85:
        word_length = 7
    elif similarity_threshold >= 0.8:
        word_length = 5
    elif similarity_threshold >= 0.7:
        word_length = 4
    else:
        word_length = 3
    
    cmd = [
        "cd-hit-est",
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(similarity_threshold),
        "-n", str(word_length),
        "-M", "2000",  # 内存限制2GB
        "-T", str(threads),
        "-d", "0"  # 输出完整的序列描述
    ]
    
    try:
        print(f"🔧 运行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ CD-HIT聚类完成!")
        print(f"📊 代表性序列文件: {output_fasta}")
        print(f"📊 聚类信息文件: {cluster_file}")
        
        return output_fasta, cluster_file
        
    except subprocess.CalledProcessError as e:
        print(f"❌ CD-HIT运行失败: {e}")
        print(f"错误输出: {e.stderr}")
        raise


def parse_cdhit_output(output_fasta: str) -> Set[str]:
    """
    解析CD-HIT输出文件，获取代表性序列的ID列表
    
    Args:
        output_fasta: CD-HIT输出的FASTA文件
        
    Returns:
        代表性序列ID的集合
    """
    representative_ids = set()
    
    print("📋 解析CD-HIT输出，获取代表性序列...")
    
    try:
        with open(output_fasta, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    # 提取序列ID
                    seq_id = line.strip()[1:].split()[0]
                    representative_ids.add(seq_id)
        
        print(f"📊 找到 {len(representative_ids)} 个代表性序列")
        
    except Exception as e:
        print(f"❌ 解析CD-HIT输出失败: {e}")
        raise
    
    return representative_ids


def filter_by_similarity(filtered_sequences: Dict[str, Tuple[str, int, str, float, int, int]], 
                        similarity_threshold: float = 0.8,
                        threads: int = 4) -> Dict[str, Tuple[str, int, str, float, int, int]]:
    """
    使用CD-HIT进行序列相似度筛选
    
    Args:
        filtered_sequences: 长度筛选后的序列字典
        similarity_threshold: 相似度阈值
        threads: 线程数
        
    Returns:
        去冗余后的序列字典
    """
    if not filtered_sequences:
        return filtered_sequences
    
    print(f"\n🧬 第四步: 序列去冗余 (相似度阈值: {similarity_threshold*100}%)")
    print("-" * 60)
    
    # 创建临时FASTA文件
    temp_fasta = create_temp_fasta(filtered_sequences)
    
    try:
        # 运行CD-HIT
        output_fasta, cluster_file = run_cdhit_clustering(
            temp_fasta, similarity_threshold, threads
        )
        
        # 解析输出获取代表性序列
        representative_ids = parse_cdhit_output(output_fasta)
        
        # 筛选出代表性序列
        deduplicated_sequences = {}
        for seq_id in representative_ids:
            if seq_id in filtered_sequences:
                deduplicated_sequences[seq_id] = filtered_sequences[seq_id]
        
        print("-" * 60)
        print(f"📊 去冗余结果: 输入 {len(filtered_sequences)} 个，输出 {len(deduplicated_sequences)} 个")
        
        # 清理临时文件
        for temp_file in [temp_fasta, output_fasta, cluster_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return deduplicated_sequences
        
    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_fasta):
            os.remove(temp_fasta)
        raise


def copy_related_files(data_dir: str, filtered_sequences: Dict[str, Tuple[str, int, str, float, int, int]], 
                      output_dir: str) -> Dict[str, List[str]]:
    """
    复制符合条件的序列相关的所有文件到输出目录
    
    Args:
        data_dir: 源数据目录
        filtered_sequences: 筛选后的序列字典
        output_dir: 输出目录
        
    Returns:
        复制文件的统计信息
    """
    source_path = Path(data_dir)
    target_path = Path(output_dir)
    
    # 创建输出目录结构
    subdirs = ["sequences", "seq", "pdb", "rMSA", "cache", "list"]
    for subdir in subdirs:
        (target_path / subdir).mkdir(parents=True, exist_ok=True)
    
    copy_stats = defaultdict(list)
    
    print("🚚 开始复制文件...")
    print("-" * 80)
    
    for basename, (file_path, length, file_type, completeness, actual_atoms, expected_atoms) in filtered_sequences.items():
        print(f"\n📁 处理 {basename} ({length}nt, 完整性: {completeness:.1%}):")
        
        # 复制原始序列文件
        if file_type == 'fasta':
            source_file = source_path / "sequences" / f"{basename}.fasta"
            target_file = target_path / "sequences" / f"{basename}.fasta"
            if source_file.exists():
                shutil.copy2(str(source_file), str(target_file))
                print(f"  ✅ 复制 sequences/{basename}.fasta")
                copy_stats["sequences"].append(basename)
        elif file_type == 'seq':
            source_file = source_path / "seq" / f"{basename}.seq"
            target_file = target_path / "seq" / f"{basename}.seq"
            if source_file.exists():
                shutil.copy2(str(source_file), str(target_file))
                print(f"  ✅ 复制 seq/{basename}.seq")
                copy_stats["seq"].append(basename)
        
        # 复制相关的PDB文件
        pdb_source = source_path / "pdb" / f"{basename}.pdb"
        if pdb_source.exists():
            pdb_target = target_path / "pdb" / f"{basename}.pdb"
            shutil.copy2(str(pdb_source), str(pdb_target))
            print(f"  ✅ 复制 pdb/{basename}.pdb")
            copy_stats["pdb"].append(basename)
        else:
            print(f"  ⚠️  未找到 pdb/{basename}.pdb")
        
        # 复制MSA文件
        msa_source = source_path / "rMSA" / f"{basename}.a3m"
        if msa_source.exists():
            msa_target = target_path / "rMSA" / f"{basename}.a3m"
            shutil.copy2(str(msa_source), str(msa_target))
            print(f"  ✅ 复制 rMSA/{basename}.a3m")
            copy_stats["rMSA"].append(basename)
        else:
            print(f"  ⚠️  未找到 rMSA/{basename}.a3m")
        
        # 复制cache文件（如果存在）
        for cache_file in (source_path / "cache").glob(f"*{basename}*"):
            cache_target = target_path / "cache" / cache_file.name
            shutil.copy2(str(cache_file), str(cache_target))
            print(f"  ✅ 复制 cache/{cache_file.name}")
            copy_stats["cache"].append(cache_file.name)
    
    # 复制list文件夹的内容
    list_source = source_path / "list"
    list_target = target_path / "list"
    if list_source.exists():
        print(f"\n📁 复制list目录内容...")
        for list_file in list_source.glob("*"):
            if list_file.is_file():
                shutil.copy2(str(list_file), str(list_target / list_file.name))
                print(f"  ✅ 复制 list/{list_file.name}")
                copy_stats["list"].append(list_file.name)
    
    return dict(copy_stats)


def generate_summary_report(original_count: int, length_filtered_count: int,
                           final_sequences: Dict[str, Tuple[str, int, str, float, int, int]], 
                           copy_stats: Dict[str, List[str]], 
                           similarity_threshold: float,
                           min_completeness: float,
                           max_completeness: float,
                           output_file: str = "filter_summary.txt"):
    """生成筛选和复制的汇总报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("RNA数据筛选汇总报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"筛选条件:\n")
        f.write(f"  - 长度范围: 16-256nt\n")
        f.write(f"  - 原子完整性: {min_completeness:.1%}-{max_completeness:.1%}\n")
        f.write(f"  - 序列相似度: <{similarity_threshold*100}%\n\n")
        
        f.write(f"筛选流程:\n")
        f.write(f"  1. 原始序列数: {original_count}\n")
        f.write(f"  2. 长度+完整性筛选后: {length_filtered_count}\n")
        f.write(f"  3. 去冗余后: {len(final_sequences)}\n\n")
        
        f.write("长度分布:\n")
        f.write("-" * 40 + "\n")
        
        # 统计长度分布
        length_counts = defaultdict(int)
        for _, (_, length, _, _, _, _) in final_sequences.items():
            length_range = f"{(length//10)*10}-{(length//10)*10+9}"
            length_counts[length_range] += 1
        
        for length_range in sorted(length_counts.keys(), key=lambda x: int(x.split('-')[0])):
            f.write(f"{length_range}nt: {length_counts[length_range]} 个\n")
        
        f.write("\n完整性分布:\n")
        f.write("-" * 40 + "\n")
        
        # 统计完整性分布
        completeness_counts = defaultdict(int)
        for _, (_, _, _, completeness, _, _) in final_sequences.items():
            completeness_range = f"{int(completeness*10)*10}-{(int(completeness*10)*10+9)}%"
            completeness_counts[completeness_range] += 1
        
        for completeness_range in sorted(completeness_counts.keys(), key=lambda x: int(x.split('-')[0])):
            f.write(f"{completeness_range}: {completeness_counts[completeness_range]} 个\n")
        
        f.write("\n复制文件统计:\n")
        f.write("-" * 40 + "\n")
        for file_type, files in copy_stats.items():
            f.write(f"{file_type}: {len(files)} 个文件\n")
        
        f.write("\n最终保留的序列列表:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'序列名':<20} {'长度':<8} {'完整性':<10} {'实际原子':<10} {'预期原子':<10} {'类型':<8}\n")
        f.write("-" * 80 + "\n")
        
        for basename, (_, length, file_type, completeness, actual_atoms, expected_atoms) in sorted(final_sequences.items()):
            f.write(f"{basename:<20} {length:<8} {completeness:.1%} {actual_atoms:<10} {expected_atoms:<10} {file_type:<8}\n")
    
    print(f"📄 已生成汇总报告: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="根据序列长度和原子完整性筛选RNA数据并进行去冗余")
    
    parser.add_argument("--input_dir", type=str, 
                       help="输入数据目录路径")
    parser.add_argument("--output_dir", type=str,
                       help="输出数据目录路径")
    parser.add_argument("--min_length", type=int, default=16,
                       help="最小序列长度 (默认: 16)")
    parser.add_argument("--max_length", type=int, default=256,
                       help="最大序列长度 (默认: 256)")
    parser.add_argument("--min_completeness", type=float, default=0.8,
                       help="最小原子完整性阈值 (默认: 0.8, 即80%%)")
    parser.add_argument("--max_completeness", type=float, default=1.0,
                       help="最大原子完整性阈值 (默认: 1.0, 即100%%)")
    parser.add_argument("--similarity", type=float, default=0.8,
                       help="序列相似度阈值 (默认: 0.8, 即80%%)")
    parser.add_argument("--threads", type=int, default=4,
                       help="CD-HIT使用的线程数 (默认: 4)")
    parser.add_argument("--dry_run", action="store_true",
                       help="只分析不复制文件")
    parser.add_argument("--skip_similarity", action="store_true",
                       help="跳过相似度筛选，只进行长度和完整性筛选")
    parser.add_argument("--report", type=str, default="filter_summary.txt",
                       help="汇总报告文件名 (默认: filter_summary.txt)")
    
    args = parser.parse_args()
    
    print("🧬 RNA数据筛选工具 (长度 + 完整性 + 相似度)")
    print("=" * 60)
    print(f"📁 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📏 长度范围: {args.min_length}-{args.max_length}nt")
    print(f"🔬 原子完整性: {args.min_completeness:.1%}-{args.max_completeness:.1%}")
    print(f"🔬 相似度阈值: <{args.similarity*100}% {'(跳过)' if args.skip_similarity else ''}")
    print(f"🧵 线程数: {args.threads}")
    print(f"🚫 仅分析模式: {args.dry_run}")
    print("=" * 60)
    
    try:
        # 检查输入目录是否存在
        if not Path(args.input_dir).exists():
            raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")
        
        # 扫描所有序列文件
        print("\n🔍 第一步: 扫描所有序列文件和PDB文件...")
        all_sequences = scan_all_sequences(args.input_dir, args.min_completeness)
        print(f"📊 找到 {len(all_sequences)} 个序列文件")
        original_count = len(all_sequences)
        
        # 根据长度和完整性筛选
        print("\n🔍 第二步: 根据长度和原子完整性筛选...")
        length_filtered = filter_by_length_and_completeness(
            all_sequences, args.min_length, args.max_length, args.min_completeness, args.max_completeness
        )
        length_filtered_count = len(length_filtered)
        
        # 序列相似度筛选
        if not args.skip_similarity and length_filtered:
            final_sequences = filter_by_similarity(
                length_filtered, args.similarity, args.threads
            )
        else:
            final_sequences = length_filtered
            if args.skip_similarity:
                print("\n⏭️  跳过相似度筛选")
        
        if not args.dry_run and final_sequences:
            # 复制文件
            print(f"\n🚚 第{'四' if not args.skip_similarity else '三'}步: 复制相关文件...")
            copy_stats = copy_related_files(
                args.input_dir, final_sequences, args.output_dir
            )
            
            # 生成报告
            generate_summary_report(
                original_count, length_filtered_count, 
                final_sequences, copy_stats, args.similarity, args.min_completeness, args.max_completeness, args.report
            )
            
            print("\n" + "=" * 60)
            print("✅ 数据筛选和复制完成!")
            print(f"📊 原始序列: {original_count} 个")
            print(f"📊 长度+完整性筛选后: {length_filtered_count} 个")
            print(f"📊 去冗余后: {len(final_sequences)} 个")
            print(f"📁 输出目录: {args.output_dir}")
            print(f"📄 汇总报告: {args.report}")
            
        elif args.dry_run:
            print("\n" + "=" * 60)
            print("🚫 仅分析模式 - 未复制文件")
            print(f"📊 原始序列: {original_count} 个")
            print(f"📊 长度+完整性筛选后: {length_filtered_count} 个")
            print(f"📊 去冗余后: {len(final_sequences)} 个")
            
            # 显示一些统计信息
            if final_sequences:
                lengths = [length for _, length, _, _, _, _ in final_sequences.values()]
                completeness_values = [completeness for _, _, _, completeness, _, _ in final_sequences.values()]
                actual_atoms_list = [actual_atoms for _, _, _, _, actual_atoms, _ in final_sequences.values()]
                expected_atoms_list = [expected_atoms for _, _, _, _, _, expected_atoms in final_sequences.values()]
                print(f"📏 长度范围: {min(lengths)}-{max(lengths)}nt")
                print(f"📏 平均长度: {sum(lengths)/len(lengths):.1f}nt")
                print(f"🔬 完整性范围: {min(completeness_values):.1%}-{max(completeness_values):.1%}")
                print(f"🔬 平均完整性: {sum(completeness_values)/len(completeness_values):.1%}")
                print(f"🔬 实际原子数范围: {min(actual_atoms_list)}-{max(actual_atoms_list)}")
                print(f"🔬 预期原子数范围: {min(expected_atoms_list)}-{max(expected_atoms_list)}")
        else:
            print("\n" + "=" * 60)
            print("⚠️  没有找到符合条件的序列")
    
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

