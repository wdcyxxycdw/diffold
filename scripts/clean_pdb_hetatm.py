#!/usr/bin/env python3
"""
清理PDB文件中的HETATM记录，只保留ATOM记录
"""

import argparse
import logging
from pathlib import Path
import sys


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def clean_pdb_file(input_file: Path, output_file: Path) -> dict:
    """
    清理单个PDB文件，去除HETATM记录
    
    返回: dict with 'atom_count', 'hetatm_count'
    """
    atom_lines = []
    hetatm_count = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_lines.append(line)
            elif line.startswith('HETATM'):
                hetatm_count += 1
    
    # 如果没有变化，不需要重写
    if hetatm_count == 0:
        return {'atom_count': len(atom_lines), 'hetatm_count': 0, 'changed': False}
    
    # 写入清理后的文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for line in atom_lines:
            f.write(line)
    
    return {
        'atom_count': len(atom_lines),
        'hetatm_count': hetatm_count,
        'changed': True
    }


def process_directory(input_dir: Path, output_dir: Path, in_place: bool = False):
    """批量处理目录中的PDB文件"""
    
    pdb_files = list(input_dir.glob('*.pdb'))
    
    if not pdb_files:
        logger.error(f"未找到PDB文件: {input_dir}")
        return
    
    logger.info("=" * 80)
    logger.info("清理PDB文件中的HETATM记录")
    logger.info("=" * 80)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir if not in_place else '(原地修改)'}")
    logger.info(f"文件数量: {len(pdb_files)}")
    logger.info("=" * 80)
    
    total_files = 0
    total_cleaned = 0
    total_hetatm_removed = 0
    
    for i, pdb_file in enumerate(pdb_files, 1):
        output_file = output_dir / pdb_file.name if not in_place else pdb_file
        
        stats = clean_pdb_file(pdb_file, output_file)
        
        total_files += 1
        if stats['changed']:
            total_cleaned += 1
            total_hetatm_removed += stats['hetatm_count']
            logger.info(f"[{i}/{len(pdb_files)}] ✓ {pdb_file.name}: "
                       f"移除 {stats['hetatm_count']} 个HETATM, "
                       f"保留 {stats['atom_count']} 个ATOM")
        else:
            logger.debug(f"[{i}/{len(pdb_files)}] - {pdb_file.name}: 无需修改")
    
    logger.info("\n" + "=" * 80)
    logger.info("处理完成")
    logger.info("=" * 80)
    logger.info(f"总文件数: {total_files}")
    logger.info(f"清理的文件: {total_cleaned}")
    logger.info(f"移除的HETATM记录: {total_hetatm_removed}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='清理PDB文件中的HETATM记录，只保留ATOM记录'
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='输入PDB文件目录'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出目录（如果不指定，则原地修改）'
    )
    
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='原地修改文件（覆盖原文件）'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='调试模式'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    input_dir = Path(args.input_dir)
    
    if args.in_place:
        output_dir = input_dir
    elif args.output:
        output_dir = Path(args.output)
    else:
        logger.error("必须指定 --output 或 --in-place")
        return 1
    
    process_directory(input_dir, output_dir, args.in_place)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

