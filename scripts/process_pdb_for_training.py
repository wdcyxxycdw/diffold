#!/usr/bin/env python3
"""
处理下载的PDB文件，提取RNA链并转换为训练格式

功能：
1. 提取ATOM记录（去除HEADER、REMARK等元数据）
2. 只保留RNA残基（去除蛋白质、DNA等）
3. 按链分离，保存为 pdbID_chainID.pdb 格式
4. 支持批量处理
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional
import sys


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# RNA标准残基（包括一些常见的修饰残基）
RNA_RESIDUES = {
    # 标准RNA碱基
    'A', 'C', 'G', 'U',
    # 修饰的RNA残基（常见的）
    'ADE', 'CYT', 'GUA', 'URA',  # 全名
    '1MA', '2MG', '5MC', '5MU', '7MG',  # 甲基化
    'PSU', 'H2U', 'M2G', 'OMC', 'OMG', 'YYG',  # 其他修饰
    'I', 'T',  # 次黄嘌呤、胸腺嘧啶（在某些tRNA中）
    # 更多修饰残基
    '4SU', '6MZ', 'A2M', 'A23', 'A3P', 'A44', 'A5M', 'A5O', 'A6A', 'ABV',
    'ACL', 'AD2', 'AET', 'AF2', 'ALY', 'AMP', 'AVC', 'BGM', 'C25', 'C2L',
    'C31', 'C43', 'C5L', 'CAR', 'CCC', 'CFL', 'CFZ', 'CMS', 'CSF', 'CZZ',
    'D3P', 'DOC', 'DRT', 'EDA', 'F3N', 'FA2', 'FHU', 'FMU', 'FOE', 'FRU',
    'G2L', 'G46', 'G48', 'G7M', 'GAO', 'GDP', 'GMS', 'GOM', 'GRB', 'GTP',
    'GUN', 'H2U', 'HPA', 'I', 'I5C', 'IC', 'IGU', 'IMP', 'IPN', 'IU',
    'LCA', 'LCC', 'LCG', 'M1G', 'M2G', 'M5M', 'M5U', 'M7A', 'MA6', 'MA7',
    'MAD', 'MG5', 'MIA', 'MMT', 'MNU', 'OMG', 'OMU', 'ONE', 'P', 'P2U',
    'P5P', 'PGP', 'PPU', 'PPZ', 'PRN', 'PSU', 'PU', 'QUO', 'RIA', 'S4C',
    'S4U', 'S6G', 'SMP', 'T', 'T23', 'T2S', 'T2T', 'T31', 'T32', 'T36',
    'T37', 'T38', 'T39', 'T3P', 'T41', 'T48', 'T49', 'T4S', 'T5O', 'T5S',
    'T64', 'T6A', 'TA3', 'TAF', 'TCP', 'TFE', 'TFO', 'TGP', 'THM', 'TLC',
    'TME', 'TPG', 'TSB', 'TSP', 'TTE', 'U23', 'U25', 'U2L', 'U2P', 'U31',
    'U33', 'U34', 'U36', 'U37', 'U3H', 'U8U', 'UAR', 'UBB', 'UBD', 'UD5',
    'UMP', 'UMS', 'UPE', 'UR3', 'URD', 'US1', 'US2', 'US3', 'US5', 'UTP',
    'UVX', 'VGN', 'XAN', 'XTS', 'XUA', 'YG', 'YYG', 'ZAD', 'ZBC', 'ZBU',
    'ZCY', 'ZDU', 'ZGU'
}

# DNA残基（用于排除）
DNA_RESIDUES = {
    'DA', 'DC', 'DG', 'DT',  # 标准DNA碱基
    'DI', 'DU',  # 其他DNA
    'ADE', 'CYT', 'GUA', 'THY',  # DNA全名
}

# 蛋白质残基（用于排除）
PROTEIN_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    # 单字母也要排除（虽然与RNA的A、C、G、U、T、I重叠，但在context下可以判断）
}


def is_rna_residue(residue_name: str) -> bool:
    """
    判断残基是否为RNA
    
    参数:
        residue_name: 残基名称（去除空格）
    
    返回:
        bool: 是否为RNA残基
    """
    residue_name = residue_name.strip().upper()
    
    # 先检查是否明确是DNA或蛋白质
    if residue_name in DNA_RESIDUES:
        return False
    if residue_name in PROTEIN_RESIDUES and len(residue_name) == 3:
        return False
    
    # 检查是否是RNA
    return residue_name in RNA_RESIDUES


def parse_pdb_file(pdb_file: Path) -> Dict[str, List[str]]:
    """
    解析PDB文件，按链分组提取RNA的ATOM记录
    
    参数:
        pdb_file: PDB文件路径
    
    返回:
        Dict[chain_id, List[atom_lines]]: 每条链的ATOM记录
    """
    chains = {}
    chain_residue_counts = {}  # 统计每条链的残基类型
    in_model = False  # 是否在模型内
    model_count = 0   # 模型计数
    
    try:
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
                        # 第一个模型结束后直接退出，不再处理其他模型
                        break
                    continue
                
                # 如果文件有MODEL标记，只处理第一个模型内的原子
                # 如果没有MODEL标记（单模型），则处理所有原子
                if model_count > 0 and not in_model:
                    continue
                
                # 只处理ATOM和HETATM记录
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                
                # PDB格式解析（固定列宽）
                # ATOM record format:
                # COLUMNS        DATA TYPE       FIELD         DEFINITION
                # 1-6            Record name     "ATOM  "
                # 7-11           Integer         serial        Atom serial number
                # 13-16          Atom            name          Atom name
                # 17             Character       altLoc        Alternate location indicator
                # 18-20          Residue name    resName       Residue name
                # 22             Character       chainID       Chain identifier
                # 23-26          Integer         resSeq        Residue sequence number
                # 27             AChar           iCode         Code for insertion of residues
                # 31-38          Real(8.3)       x             Orthogonal coordinates for X
                # 39-46          Real(8.3)       y             Orthogonal coordinates for Y
                # 47-54          Real(8.3)       z             Orthogonal coordinates for Z
                
                if len(line) < 54:  # 确保行足够长
                    continue
                
                # 提取残基名称和链ID
                try:
                    residue_name = line[17:20].strip()
                    chain_id = line[21:22].strip()
                except IndexError:
                    continue
                
                # 如果链ID为空，使用默认值
                if not chain_id:
                    chain_id = 'A'
                
                # 检查是否为RNA残基
                if not is_rna_residue(residue_name):
                    continue
                
                # 添加到对应链
                if chain_id not in chains:
                    chains[chain_id] = []
                    chain_residue_counts[chain_id] = set()
                
                chains[chain_id].append(line)
                chain_residue_counts[chain_id].add(residue_name)
        
        # 记录每条链的信息
        if model_count > 1:
            logger.info(f"  检测到多模型PDB文件（共{model_count}个模型），只提取第一个模型")
        
        for chain_id, residues in chain_residue_counts.items():
            logger.debug(f"  链 {chain_id}: {len(chains[chain_id])} 个原子, "
                        f"残基类型: {', '.join(sorted(residues))}")
        
        return chains
    
    except Exception as e:
        logger.error(f"解析文件 {pdb_file} 时出错: {e}")
        return {}


def save_chain(
    output_file: Path,
    atom_lines: List[str],
    renumber: bool = True
) -> bool:
    """
    保存单条链的ATOM记录到文件
    
    参数:
        output_file: 输出文件路径
        atom_lines: ATOM记录行列表
        renumber: 是否重新编号原子序号和残基序号
    
    返回:
        bool: 是否成功保存
    """
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            if renumber:
                # 重新编号
                atom_num = 1
                prev_res_num = None
                new_res_num = 0
                
                for line in atom_lines:
                    # 提取原残基序号
                    try:
                        old_res_num = int(line[22:26].strip())
                    except ValueError:
                        old_res_num = prev_res_num
                    
                    # 如果残基序号变化，递增新序号
                    if old_res_num != prev_res_num:
                        new_res_num += 1
                        prev_res_num = old_res_num
                    
                    # 重新格式化行（更新原子序号和残基序号）
                    new_line = (
                        f"{line[0:6]}"           # Record name
                        f"{atom_num:5d} "        # Atom serial
                        f"{line[12:17]}"         # Atom name + altLoc
                        f"{line[17:22]}"         # Residue name + chainID
                        f"{new_res_num:4d}"      # Residue sequence number
                        f"{line[26:]}"           # Rest of the line
                    )
                    f.write(new_line)
                    atom_num += 1
            else:
                # 直接写入
                for line in atom_lines:
                    f.write(line)
        
        logger.debug(f"  保存成功: {output_file.name} ({len(atom_lines)} 个原子)")
        return True
    
    except Exception as e:
        logger.error(f"保存文件 {output_file} 时出错: {e}")
        return False


def process_single_pdb(
    pdb_file: Path,
    output_dir: Path,
    renumber: bool = True,
    min_atoms: int = 10
) -> Dict[str, int]:
    """
    处理单个PDB文件
    
    参数:
        pdb_file: 输入PDB文件
        output_dir: 输出目录
        renumber: 是否重新编号
        min_atoms: 最小原子数（少于此数的链会被跳过）
    
    返回:
        Dict: 统计信息 {'chains': N, 'atoms': M}
    """
    # 获取PDB ID（文件名去除.pdb后缀）
    pdb_id = pdb_file.stem.lower()
    
    logger.info(f"处理: {pdb_file.name}")
    
    # 解析PDB文件
    chains = parse_pdb_file(pdb_file)
    
    if not chains:
        logger.warning(f"  未找到RNA链: {pdb_file.name}")
        return {'chains': 0, 'atoms': 0}
    
    logger.info(f"  找到 {len(chains)} 条RNA链")
    
    # 保存每条链
    total_atoms = 0
    saved_chains = 0
    
    for chain_id, atom_lines in chains.items():
        if len(atom_lines) < min_atoms:
            logger.info(f"  跳过链 {chain_id}: 原子数太少 ({len(atom_lines)} < {min_atoms})")
            continue
        
        # 生成输出文件名
        output_file = output_dir / f"{pdb_id}_{chain_id}.pdb"
        
        # 保存链
        if save_chain(output_file, atom_lines, renumber):
            saved_chains += 1
            total_atoms += len(atom_lines)
            logger.info(f"  ✓ 链 {chain_id}: {len(atom_lines)} 个原子 -> {output_file.name}")
    
    return {'chains': saved_chains, 'atoms': total_atoms}


def batch_process(
    input_path: Path,
    output_dir: Path,
    renumber: bool = True,
    min_atoms: int = 10,
    recursive: bool = False
) -> Dict[str, int]:
    """
    批量处理PDB文件
    
    参数:
        input_path: 输入路径（文件或目录）
        output_dir: 输出目录
        renumber: 是否重新编号
        min_atoms: 最小原子数
        recursive: 是否递归搜索子目录
    
    返回:
        Dict: 总体统计信息
    """
    # 收集PDB文件
    pdb_files = []
    
    if input_path.is_file():
        if input_path.suffix.lower() == '.pdb':
            pdb_files = [input_path]
    elif input_path.is_dir():
        if recursive:
            pdb_files = list(input_path.rglob('*.pdb'))
        else:
            pdb_files = list(input_path.glob('*.pdb'))
    else:
        logger.error(f"输入路径不存在: {input_path}")
        return {'files': 0, 'chains': 0, 'atoms': 0}
    
    if not pdb_files:
        logger.error(f"未找到PDB文件: {input_path}")
        return {'files': 0, 'chains': 0, 'atoms': 0}
    
    logger.info("=" * 80)
    logger.info("批量处理PDB文件")
    logger.info("=" * 80)
    logger.info(f"输入路径: {input_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"文件数量: {len(pdb_files)}")
    logger.info(f"重新编号: {'是' if renumber else '否'}")
    logger.info(f"最小原子数: {min_atoms}")
    logger.info("=" * 80)
    
    # 处理每个文件
    total_stats = {'files': 0, 'chains': 0, 'atoms': 0}
    
    for i, pdb_file in enumerate(pdb_files, 1):
        logger.info(f"\n[{i}/{len(pdb_files)}] {pdb_file.name}")
        
        stats = process_single_pdb(pdb_file, output_dir, renumber, min_atoms)
        
        if stats['chains'] > 0:
            total_stats['files'] += 1
            total_stats['chains'] += stats['chains']
            total_stats['atoms'] += stats['atoms']
    
    return total_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='处理PDB文件，提取RNA链并转换为训练格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个PDB文件
  python process_pdb_for_training.py input.pdb --output ./processed
  
  # 批量处理目录中的所有PDB文件
  python process_pdb_for_training.py ./pdb_downloads/pdb --output ./processed_data/pdb
  
  # 递归处理子目录
  python process_pdb_for_training.py ./downloads --output ./processed --recursive
  
  # 保留原始编号（不重新编号）
  python process_pdb_for_training.py input.pdb --output ./processed --no-renumber
  
  # 设置最小原子数阈值
  python process_pdb_for_training.py ./pdbs --output ./processed --min-atoms 20

输出格式:
  - 文件名: pdbID_chainID.pdb (例如: 1ehz_A.pdb)
  - 只包含ATOM/HETATM记录
  - 只包含RNA残基
  - 每个文件对应一条链
"""
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='输入PDB文件或目录'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录'
    )
    
    parser.add_argument(
        '--no-renumber',
        action='store_true',
        help='不重新编号原子和残基序号（保留原始编号）'
    )
    
    parser.add_argument(
        '--min-atoms',
        type=int,
        default=10,
        help='最小原子数，少于此数的链会被跳过（默认: 10）'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='递归搜索输入目录的子目录'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式（只显示警告和错误）'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='调试模式（显示详细信息）'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # 批量处理
    stats = batch_process(
        input_path=input_path,
        output_dir=output_dir,
        renumber=not args.no_renumber,
        min_atoms=args.min_atoms,
        recursive=args.recursive
    )
    
    # 打印总结
    logger.info("\n" + "=" * 80)
    logger.info("处理完成")
    logger.info("=" * 80)
    logger.info(f"处理的文件: {stats['files']}")
    logger.info(f"提取的链: {stats['chains']}")
    logger.info(f"总原子数: {stats['atoms']}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 80)
    
    return 0 if stats['chains'] > 0 else 1


if __name__ == '__main__':
    sys.exit(main())

