#!/usr/bin/env python3
"""
下载CASP和RNA-Puzzles基准数据集
包括PDB结构文件和序列文件
"""

import argparse
import os
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error
from typing import List, Dict, Optional


# CASP RNA targets (示例，需要根据实际情况更新)
CASP_RNA_TARGETS = {
    'CASP15': [
        # CASP15 RNA targets
        'R1107', 'R1108', 'R1116', 'R1117', 'R1126', 'R1136'
    ],
    'CASP14': [],  # 如有需要可添加
}

# RNA-Puzzles targets (常见的puzzle编号和对应的PDB ID)
RNA_PUZZLES_TARGETS = {
    'puzzle1': {'pdb_ids': ['2l8f'], 'description': 'GlmS riboswitch'},
    'puzzle2': {'pdb_ids': ['2lc8'], 'description': 'SAH riboswitch'},
    'puzzle3': {'pdb_ids': ['2lhp'], 'description': 'Lysine riboswitch'},
    'puzzle4': {'pdb_ids': ['2m8k'], 'description': 'SAM-I riboswitch'},
    'puzzle5': {'pdb_ids': ['2n3r'], 'description': 'FMN riboswitch'},
    'puzzle6': {'pdb_ids': ['2m4q'], 'description': 'Adenine riboswitch'},
    'puzzle7': {'pdb_ids': ['2m21'], 'description': 'T-box riboswitch'},
    'puzzle8': {'pdb_ids': ['2m24'], 'description': 'preQ1 riboswitch'},
    'puzzle9': {'pdb_ids': ['2m18'], 'description': 'c-di-GMP riboswitch'},
    'puzzle10': {'pdb_ids': ['2m22'], 'description': 'THF riboswitch'},
    'puzzle11': {'pdb_ids': ['3q50'], 'description': 'Guanine riboswitch'},
    'puzzle12': {'pdb_ids': ['3sux'], 'description': 'Cyclic-di-GMP riboswitch'},
    'puzzle13': {'pdb_ids': ['3u4m'], 'description': 'SAM-II riboswitch'},
    'puzzle14': {'pdb_ids': ['4k31'], 'description': 'Twister ribozyme'},
    'puzzle15': {'pdb_ids': ['4nio'], 'description': 'Hatchet ribozyme'},
    'puzzle16': {'pdb_ids': ['4p5j'], 'description': 'Pistol ribozyme'},
    'puzzle17': {'pdb_ids': ['4q9q'], 'description': 'ZTP riboswitch'},
    'puzzle18': {'pdb_ids': ['4xw7'], 'description': 'Twister sister ribozyme'},
    'puzzle19': {'pdb_ids': ['5btp'], 'description': 'YdaO riboswitch'},
    'puzzle20': {'pdb_ids': ['5di1'], 'description': 'Glycine riboswitch'},
    'puzzle21': {'pdb_ids': ['5kx9'], 'description': 'ZMP riboswitch'},
    'puzzle22': {'pdb_ids': ['6d90'], 'description': 'NMT1 ligase ribozyme'},
    'puzzle23': {'pdb_ids': ['6qn3'], 'description': 'Twister ribozyme variant'},
    'puzzle24': {'pdb_ids': ['6r47'], 'description': 'Pistol ribozyme variant'},
}


def download_file(url: str, output_path: str, max_retries: int = 3) -> bool:
    """
    下载文件，支持重试
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            print(f"  下载中 [{attempt + 1}/{max_retries}]: {url}", end=" ... ", flush=True)
            urllib.request.urlretrieve(url, str(output_path))
            print("✓")
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"✗ (404 Not Found)")
                return False
            print(f"✗ HTTP Error {e.code}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
        except Exception as e:
            print(f"✗ Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return False


def download_pdb_file(pdb_id: str, output_dir: str, file_format: str = 'pdb') -> bool:
    """
    从RCSB PDB下载结构文件
    
    参数:
        pdb_id: PDB ID (例如 '2l8f')
        output_dir: 输出目录
        file_format: 文件格式 ('pdb' 或 'cif')
    """
    pdb_id = pdb_id.lower()
    
    if file_format == 'pdb':
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_file = Path(output_dir) / f"{pdb_id}.pdb"
    elif file_format == 'cif':
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        output_file = Path(output_dir) / f"{pdb_id}.cif"
    else:
        raise ValueError(f"不支持的文件格式: {file_format}")
    
    # 如果文件已存在，跳过
    if output_file.exists():
        print(f"  ⊘ 跳过 {pdb_id}.{file_format} (已存在)")
        return True
    
    return download_file(url, str(output_file))


def download_pdb_fasta(pdb_id: str, output_dir: str) -> bool:
    """
    从RCSB PDB下载FASTA序列文件
    """
    pdb_id = pdb_id.lower()
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    output_file = Path(output_dir) / f"{pdb_id}.fasta"
    
    # 如果文件已存在，跳过
    if output_file.exists():
        print(f"  ⊘ 跳过 {pdb_id}.fasta (已存在)")
        return True
    
    return download_file(url, str(output_file))


def download_rna_puzzles(
    output_dir: str,
    puzzles: Optional[List[str]] = None,
    download_pdb: bool = True,
    download_fasta: bool = True,
    file_format: str = 'pdb',
) -> Dict[str, int]:
    """
    下载RNA-Puzzles数据集
    
    返回统计信息: {'success': N, 'failed': M, 'skipped': K}
    """
    output_dir = Path(output_dir)
    pdb_dir = output_dir / "pdb"
    fasta_dir = output_dir / "fasta"
    
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    # 确定要下载的puzzles
    if puzzles is None:
        puzzles_to_download = RNA_PUZZLES_TARGETS.keys()
    else:
        puzzles_to_download = [p for p in puzzles if p in RNA_PUZZLES_TARGETS]
    
    print(f"\n{'='*80}")
    print(f"下载 RNA-Puzzles 数据集")
    print(f"{'='*80}")
    print(f"目标puzzle数量: {len(puzzles_to_download)}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")
    
    for puzzle_name in sorted(puzzles_to_download):
        puzzle_info = RNA_PUZZLES_TARGETS[puzzle_name]
        pdb_ids = puzzle_info['pdb_ids']
        description = puzzle_info['description']
        
        print(f"{puzzle_name}: {description}")
        
        for pdb_id in pdb_ids:
            success = True
            
            # 下载PDB文件
            if download_pdb:
                if not download_pdb_file(pdb_id, str(pdb_dir), file_format):
                    success = False
            
            # 下载FASTA序列
            if download_fasta:
                if not download_pdb_fasta(pdb_id, str(fasta_dir)):
                    success = False
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        print()
    
    return stats


def download_casp(
    output_dir: str,
    casp_version: str = 'CASP15',
    targets: Optional[List[str]] = None,
    download_pdb: bool = True,
    download_fasta: bool = True,
    file_format: str = 'pdb',
) -> Dict[str, int]:
    """
    下载CASP RNA targets数据
    
    注意: CASP数据通常需要从官网手动下载或通过API获取
    这里提供基本框架，实际使用需要根据CASP具体要求调整
    """
    output_dir = Path(output_dir)
    
    print(f"\n{'='*80}")
    print(f"下载 {casp_version} RNA targets")
    print(f"{'='*80}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")
    
    if casp_version not in CASP_RNA_TARGETS:
        print(f"警告: {casp_version} 没有预定义的target列表")
        print("请手动从CASP官网下载: https://predictioncenter.org/")
        return {'success': 0, 'failed': 0, 'skipped': 0}
    
    targets_list = CASP_RNA_TARGETS[casp_version]
    
    if not targets_list:
        print(f"注意: {casp_version} RNA targets 列表为空")
        print("请访问 https://predictioncenter.org/ 获取最新的target列表")
        return {'success': 0, 'failed': 0, 'skipped': 0}
    
    print(f"找到 {len(targets_list)} 个targets")
    print("注意: CASP数据通常需要从官网手动下载")
    print("官网: https://predictioncenter.org/")
    print()
    
    # 这里可以添加实际的下载逻辑
    # 例如通过CASP API或从已知的PDB映射下载
    
    return {'success': 0, 'failed': 0, 'skipped': 0}


def download_custom_pdb_list(
    pdb_list_file: str,
    output_dir: str,
    download_pdb: bool = True,
    download_fasta: bool = True,
    file_format: str = 'pdb',
) -> Dict[str, int]:
    """
    从自定义PDB ID列表文件下载数据
    
    列表文件格式: 每行一个PDB ID
    """
    output_dir = Path(output_dir)
    pdb_dir = output_dir / "pdb"
    fasta_dir = output_dir / "fasta"
    
    # 读取PDB ID列表
    pdb_ids = []
    with open(pdb_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            # 只取第一个单词作为PDB ID（忽略行内注释）
            pdb_id = line.split()[0]
            pdb_ids.append(pdb_id)
    
    print(f"\n{'='*80}")
    print(f"从自定义列表下载PDB数据")
    print(f"{'='*80}")
    print(f"列表文件: {pdb_list_file}")
    print(f"PDB数量: {len(pdb_ids)}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")
    
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    for i, pdb_id in enumerate(pdb_ids, 1):
        print(f"[{i}/{len(pdb_ids)}] {pdb_id}")
        
        success = True
        
        # 下载PDB文件
        if download_pdb:
            if not download_pdb_file(pdb_id, str(pdb_dir), file_format):
                success = False
        
        # 下载FASTA序列
        if download_fasta:
            if not download_pdb_fasta(pdb_id, str(fasta_dir)):
                success = False
        
        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1
    
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='下载CASP和RNA-Puzzles基准数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载所有RNA-Puzzles数据
  python download_benchmark_data.py --rna-puzzles --output ./benchmark_data
  
  # 只下载特定的puzzles
  python download_benchmark_data.py --rna-puzzles --puzzles puzzle1 puzzle2 puzzle5 --output ./data
  
  # 下载CASP15 RNA targets
  python download_benchmark_data.py --casp --casp-version CASP15 --output ./casp_data
  
  # 从自定义PDB列表下载
  python download_benchmark_data.py --pdb-list my_pdb_list.txt --output ./my_data
  
  # 只下载PDB文件，不下载FASTA
  python download_benchmark_data.py --rna-puzzles --no-fasta --output ./data
  
  # 下载CIF格式而不是PDB格式
  python download_benchmark_data.py --rna-puzzles --format cif --output ./data
"""
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录'
    )
    
    # 数据集选择
    dataset = parser.add_argument_group('数据集选择')
    dataset.add_argument(
        '--rna-puzzles',
        action='store_true',
        help='下载RNA-Puzzles数据集'
    )
    dataset.add_argument(
        '--casp',
        action='store_true',
        help='下载CASP RNA targets'
    )
    dataset.add_argument(
        '--pdb-list',
        type=str,
        help='从自定义PDB ID列表文件下载（每行一个PDB ID）'
    )
    
    # RNA-Puzzles选项
    rp_opts = parser.add_argument_group('RNA-Puzzles选项')
    rp_opts.add_argument(
        '--puzzles',
        nargs='+',
        help='指定要下载的puzzles (例如: puzzle1 puzzle2 puzzle5)'
    )
    
    # CASP选项
    casp_opts = parser.add_argument_group('CASP选项')
    casp_opts.add_argument(
        '--casp-version',
        type=str,
        default='CASP15',
        help='CASP版本 (default: CASP15)'
    )
    casp_opts.add_argument(
        '--targets',
        nargs='+',
        help='指定要下载的targets (例如: R1107 R1108)'
    )
    
    # 下载选项
    dl_opts = parser.add_argument_group('下载选项')
    dl_opts.add_argument(
        '--no-pdb',
        action='store_true',
        help='不下载PDB结构文件'
    )
    dl_opts.add_argument(
        '--no-fasta',
        action='store_true',
        help='不下载FASTA序列文件'
    )
    dl_opts.add_argument(
        '--format',
        choices=['pdb', 'cif'],
        default='pdb',
        help='结构文件格式 (default: pdb)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查是否至少选择了一个数据集
    if not (args.rna_puzzles or args.casp or args.pdb_list):
        print("错误: 请至少选择一个数据集 (--rna-puzzles, --casp, 或 --pdb-list)")
        return 1
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    # 下载RNA-Puzzles
    if args.rna_puzzles:
        rp_dir = output_dir / "RNA-Puzzles"
        stats = download_rna_puzzles(
            output_dir=str(rp_dir),
            puzzles=args.puzzles,
            download_pdb=not args.no_pdb,
            download_fasta=not args.no_fasta,
            file_format=args.format,
        )
        for k in total_stats:
            total_stats[k] += stats[k]
    
    # 下载CASP
    if args.casp:
        casp_dir = output_dir / args.casp_version
        stats = download_casp(
            output_dir=str(casp_dir),
            casp_version=args.casp_version,
            targets=args.targets,
            download_pdb=not args.no_pdb,
            download_fasta=not args.no_fasta,
            file_format=args.format,
        )
        for k in total_stats:
            total_stats[k] += stats[k]
    
    # 下载自定义列表
    if args.pdb_list:
        custom_dir = output_dir / "custom"
        stats = download_custom_pdb_list(
            pdb_list_file=args.pdb_list,
            output_dir=str(custom_dir),
            download_pdb=not args.no_pdb,
            download_fasta=not args.no_fasta,
            file_format=args.format,
        )
        for k in total_stats:
            total_stats[k] += stats[k]
    
    # 打印总结
    print(f"\n{'='*80}")
    print("下载完成")
    print(f"{'='*80}")
    print(f"成功: {total_stats['success']}")
    print(f"失败: {total_stats['failed']}")
    print(f"跳过: {total_stats['skipped']}")
    print(f"{'='*80}\n")
    
    return 0 if total_stats['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

