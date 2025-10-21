#!/usr/bin/env python3
"""
下载指定的PDB文件

支持通过命令行参数或交互式输入PDB编号来下载文件
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import urllib.request
import urllib.error
import time


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path, max_retries: int = 3) -> bool:
    """
    下载文件，支持重试机制
    
    参数:
        url: 下载URL
        output_path: 输出文件路径
        max_retries: 最大重试次数
    
    返回:
        bool: 下载是否成功
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"下载中 (尝试 {attempt + 1}/{max_retries}): {url}")
            urllib.request.urlretrieve(url, str(output_path))
            logger.info(f"✓ 下载成功: {output_path.name}")
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.error(f"✗ 文件不存在 (404): {url}")
                return False
            logger.warning(f"HTTP错误 {e.code}: {url}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        except urllib.error.URLError as e:
            logger.error(f"网络错误: {e.reason}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        except Exception as e:
            logger.error(f"未知错误: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
    
    logger.error(f"✗ 下载失败（已重试{max_retries}次）: {url}")
    return False


def download_pdb(
    pdb_id: str,
    output_dir: Path,
    file_format: str = 'pdb',
    skip_existing: bool = True
) -> bool:
    """
    下载单个PDB结构文件
    
    参数:
        pdb_id: PDB ID（4个字符，如 '1abc' 或 '7D4F'）
        output_dir: 输出目录
        file_format: 文件格式 ('pdb' 或 'cif')
        skip_existing: 是否跳过已存在的文件
    
    返回:
        bool: 下载是否成功
    """
    # 标准化PDB ID：转为小写，去除空格
    pdb_id = pdb_id.strip().lower()
    
    # 验证PDB ID格式（通常为4个字符）
    if len(pdb_id) != 4:
        logger.warning(f"PDB ID格式可能不正确: {pdb_id} (标准格式为4个字符)")
    
    # 构建下载URL和输出路径
    if file_format == 'pdb':
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_file = output_dir / f"{pdb_id}.pdb"
    elif file_format == 'cif':
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        output_file = output_dir / f"{pdb_id}.cif"
    else:
        logger.error(f"不支持的文件格式: {file_format}，请使用 'pdb' 或 'cif'")
        return False
    
    # 检查文件是否已存在
    if skip_existing and output_file.exists():
        logger.info(f"⊘ 跳过 {pdb_id}.{file_format} (文件已存在)")
        return True
    
    # 下载文件
    return download_file(url, output_file)


def download_fasta(
    pdb_id: str,
    output_dir: Path,
    skip_existing: bool = True
) -> bool:
    """
    下载PDB的FASTA序列文件
    
    参数:
        pdb_id: PDB ID
        output_dir: 输出目录
        skip_existing: 是否跳过已存在的文件
    
    返回:
        bool: 下载是否成功
    """
    pdb_id = pdb_id.strip().lower()
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    output_file = output_dir / f"{pdb_id}.fasta"
    
    # 检查文件是否已存在
    if skip_existing and output_file.exists():
        logger.info(f"⊘ 跳过 {pdb_id}.fasta (文件已存在)")
        return True
    
    return download_file(url, output_file)


def batch_download(
    pdb_ids: List[str],
    output_dir: Path,
    file_format: str = 'pdb',
    download_seq: bool = False,
    skip_existing: bool = True
) -> dict:
    """
    批量下载多个PDB文件
    
    参数:
        pdb_ids: PDB ID列表
        output_dir: 输出目录
        file_format: 文件格式
        download_seq: 是否同时下载序列文件
        skip_existing: 是否跳过已存在的文件
    
    返回:
        dict: 统计信息 {'success': N, 'failed': M}
    """
    stats = {'success': 0, 'failed': 0}
    
    pdb_dir = output_dir / 'pdb'
    fasta_dir = output_dir / 'fasta'
    
    logger.info("=" * 80)
    logger.info(f"开始批量下载 PDB 文件")
    logger.info(f"PDB数量: {len(pdb_ids)}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"文件格式: {file_format}")
    logger.info(f"下载序列: {'是' if download_seq else '否'}")
    logger.info("=" * 80)
    
    for i, pdb_id in enumerate(pdb_ids, 1):
        logger.info(f"\n[{i}/{len(pdb_ids)}] 处理: {pdb_id.upper()}")
        
        # 下载结构文件
        pdb_success = download_pdb(pdb_id, pdb_dir, file_format, skip_existing)
        
        # 可选：下载序列文件
        fasta_success = True
        if download_seq:
            fasta_success = download_fasta(pdb_id, fasta_dir, skip_existing)
        
        # 统计
        if pdb_success and fasta_success:
            stats['success'] += 1
        else:
            stats['failed'] += 1
    
    return stats


def read_pdb_list_from_file(file_path: str) -> List[str]:
    """
    从文件读取PDB ID列表
    
    文件格式:
        - 每行一个PDB ID
        - 支持注释行（以 # 开头）
        - 支持空行
        - 支持行内注释
    
    返回:
        List[str]: PDB ID列表
    """
    pdb_ids = []
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # 跳过空行和注释行
                if not line or line.startswith('#'):
                    continue
                
                # 提取第一个单词作为PDB ID（忽略行内注释）
                pdb_id = line.split()[0]
                pdb_ids.append(pdb_id)
        
        logger.info(f"从文件 {file_path} 读取了 {len(pdb_ids)} 个PDB ID")
        return pdb_ids
    
    except FileNotFoundError:
        logger.error(f"文件不存在: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"读取文件时出错: {e}")
        sys.exit(1)


def interactive_mode(output_dir: Path, file_format: str, download_seq: bool):
    """
    交互式模式：提示用户输入PDB ID
    """
    logger.info("\n" + "=" * 80)
    logger.info("交互式模式")
    logger.info("=" * 80)
    logger.info("请输入PDB ID（多个ID用空格或逗号分隔）")
    logger.info("输入 'q' 或 'quit' 退出")
    logger.info("=" * 80 + "\n")
    
    while True:
        try:
            user_input = input("PDB ID: ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                logger.info("退出程序")
                break
            
            if not user_input:
                continue
            
            # 解析输入（支持空格或逗号分隔）
            pdb_ids = user_input.replace(',', ' ').split()
            
            if pdb_ids:
                stats = batch_download(
                    pdb_ids=pdb_ids,
                    output_dir=output_dir,
                    file_format=file_format,
                    download_seq=download_seq
                )
                
                logger.info("\n" + "=" * 80)
                logger.info(f"下载完成 - 成功: {stats['success']}, 失败: {stats['failed']}")
                logger.info("=" * 80 + "\n")
        
        except KeyboardInterrupt:
            logger.info("\n用户中断，退出程序")
            break
        except Exception as e:
            logger.error(f"错误: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='下载指定的PDB文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载单个PDB文件
  python download_pdb.py 1abc
  
  # 下载多个PDB文件
  python download_pdb.py 1abc 2def 3ghi
  
  # 从文件读取PDB ID列表
  python download_pdb.py --from-file pdb_list.txt
  
  # 下载CIF格式文件
  python download_pdb.py 1abc --format cif
  
  # 同时下载序列文件
  python download_pdb.py 1abc 2def --with-fasta
  
  # 指定输出目录
  python download_pdb.py 1abc --output ./my_pdbs
  
  # 交互式模式（不提供PDB ID时自动进入）
  python download_pdb.py
  
  # 覆盖已存在的文件
  python download_pdb.py 1abc --force

PDB列表文件格式（用于 --from-file）:
  每行一个PDB ID，支持注释：
  
  # 这是注释
  1abc
  2def  # 行内注释也支持
  3ghi
"""
    )
    
    parser.add_argument(
        'pdb_ids',
        nargs='*',
        help='PDB ID（如 1abc 2def）。如果不提供，则进入交互式模式'
    )
    
    parser.add_argument(
        '--from-file', '-f',
        dest='input_file',
        type=str,
        help='从文件读取PDB ID列表（每行一个）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./pdb_downloads',
        help='输出目录（默认: ./pdb_downloads）'
    )
    
    parser.add_argument(
        '--format',
        choices=['pdb', 'cif'],
        default='pdb',
        help='文件格式（默认: pdb）'
    )
    
    parser.add_argument(
        '--with-fasta',
        action='store_true',
        help='同时下载FASTA序列文件'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='覆盖已存在的文件（默认跳过）'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式（只显示错误）'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置日志级别
    if args.quiet:
        logger.setLevel(logging.ERROR)
    
    output_dir = Path(args.output)
    
    # 收集PDB ID
    pdb_ids = []
    
    # 从命令行参数获取
    if args.pdb_ids:
        pdb_ids.extend(args.pdb_ids)
    
    # 从文件读取
    if args.input_file:
        pdb_ids.extend(read_pdb_list_from_file(args.input_file))
    
    # 如果没有提供PDB ID，进入交互式模式
    if not pdb_ids:
        interactive_mode(
            output_dir=output_dir,
            file_format=args.format,
            download_seq=args.with_fasta
        )
        return 0
    
    # 批量下载
    stats = batch_download(
        pdb_ids=pdb_ids,
        output_dir=output_dir,
        file_format=args.format,
        download_seq=args.with_fasta,
        skip_existing=not args.force
    )
    
    # 打印总结
    logger.info("\n" + "=" * 80)
    logger.info("下载完成")
    logger.info("=" * 80)
    logger.info(f"成功: {stats['success']}")
    logger.info(f"失败: {stats['failed']}")
    logger.info(f"总计: {stats['success'] + stats['failed']}")
    logger.info("=" * 80)
    
    return 0 if stats['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

