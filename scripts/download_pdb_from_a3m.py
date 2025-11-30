#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 R 开头的 a3m 文件（CASP16 数据）中提取查询序列，通过序列搜索找到对应的 PDB ID，并下载 PDB 文件

该脚本会：
1. 扫描指定目录中所有 R 开头的 a3m 文件
2. 从 a3m 文件中提取查询序列
3. 使用 RCSB PDB 的序列搜索 API 找到匹配的 PDB ID
4. 下载对应的 PDB 文件
5. 保存查询序列为 FASTA 文件
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import re
import urllib.request
import urllib.error
import urllib.parse
import ssl
import time
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_query_sequence_from_a3m(a3m_file: Path) -> Optional[str]:
    """
    从 a3m 文件中提取查询序列
    
    参数:
        a3m_file: a3m 文件路径
    
    返回:
        查询序列（去除空白字符），如果未找到则返回 None
    """
    try:
        with open(a3m_file, 'r') as f:
            lines = f.readlines()
        
        query_sequence = []
        in_query = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('>query'):
                in_query = True
                continue
            elif line.startswith('>'):
                # 遇到下一个序列，停止读取查询序列
                break
            
            if in_query and line:
                # 移除可能的插入符号（a3m 格式中的小写字母表示插入）
                # 只保留大写字母（标准核苷酸）
                query_sequence.append(re.sub(r'[^AUCG]', '', line.upper()))
        
        if query_sequence:
            full_sequence = ''.join(query_sequence)
            return full_sequence if full_sequence else None
        
        return None
    
    except Exception as e:
        logger.error(f"读取 a3m 文件 {a3m_file} 时出错: {e}")
        return None


def search_pdb_by_sequence(sequence: str, max_results: int = 5) -> List[Dict]:
    """
    使用 RCSB PDB 的序列搜索 API 通过序列查找匹配的 PDB ID
    
    参数:
        sequence: RNA 序列（AUCG）
        max_results: 返回的最大结果数
    
    返回:
        匹配的 PDB 信息列表，每个元素包含 pdb_id 和 score
    """
    try:
        # RCSB PDB 序列搜索 API v2
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        
        # 构建查询请求 - 使用 sequence 服务
        query = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "sequence_type": "rna",
                    "value": sequence,
                    "evalue_cutoff": 1000,
                    "identity_cutoff": 0.0
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": max_results
                },
                "scoring_strategy": "sequence",
                "sort": [{
                    "sort_by": "score",
                    "direction": "desc"
                }]
            }
        }
        
        # 发送 POST 请求
        data = json.dumps(query).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(req, context=ssl_context, timeout=120) as response:
            response_data = response.read().decode('utf-8')
            result = json.loads(response_data)
        
        # 解析结果
        matches = []
        if 'result_set' in result:
            for item in result['result_set']:
                pdb_id = item.get('identifier', '')
                if pdb_id and len(pdb_id) == 4:  # 确保是有效的 PDB ID
                    matches.append({
                        'pdb_id': pdb_id.upper(),
                        'score': item.get('score', 0)
                    })
        
        return matches
    
    except urllib.error.HTTPError as e:
        logger.warning(f"序列搜索 HTTP 错误: {e.code} - {e.reason}")
        return []
    except urllib.error.URLError as e:
        logger.warning(f"序列搜索网络错误: {e.reason}")
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"序列搜索响应解析错误: {e}")
        return []
    except Exception as e:
        logger.warning(f"序列搜索失败: {e}")
        return []


def find_best_pdb_match(sequence: str, target_name: str) -> Optional[str]:
    """
    通过序列搜索找到最佳匹配的 PDB ID
    
    参数:
        sequence: 查询序列
        target_name: 目标名称（如 R1255）
    
    返回:
        最佳匹配的 PDB ID，如果未找到则返回 None
    """
    logger.info(f"  搜索序列匹配的 PDB（序列长度: {len(sequence)}）...")
    
    matches = search_pdb_by_sequence(sequence, max_results=10)
    
    if not matches:
        logger.warning(f"  未找到匹配的 PDB")
        return None
    
    # 显示所有匹配结果
    logger.info(f"  找到 {len(matches)} 个可能的匹配:")
    for i, match in enumerate(matches[:5], 1):  # 只显示前5个
        logger.info(f"    {i}. {match['pdb_id']} (score: {match.get('score', 'N/A')})")
    
    # 返回最佳匹配（第一个结果）
    best_match = matches[0]['pdb_id']
    logger.info(f"  选择最佳匹配: {best_match}")
    
    return best_match


def download_file(url: str, output_path: Path, max_retries: int = 3) -> bool:
    """
    下载文件，支持重试机制
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建 SSL 上下文
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    for attempt in range(max_retries):
        try:
            logger.info(f"下载中 (尝试 {attempt + 1}/{max_retries}): {url}")
            with urllib.request.urlopen(url, context=ssl_context, timeout=30) as response:
                content = response.read()
                # 检查内容是否有效（不是错误页面）
                if len(content) < 100 or b'Error' in content[:500] or b'404' in content[:500]:
                    logger.warning(f"下载的内容可能无效: {url}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return False
                
                with open(output_path, 'wb') as out_file:
                    out_file.write(content)
            logger.info(f"✓ 下载成功: {output_path.name}")
            return True
        
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"✗ 文件不存在 (404): {url}")
                return False
            logger.warning(f"HTTP错误 {e.code}: {url}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        except urllib.error.URLError as e:
            logger.warning(f"网络错误: {e.reason}")
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


def download_pdb(pdb_id: str, output_dir: Path, target_name: Optional[str] = None, skip_existing: bool = True) -> bool:
    """
    下载 PDB 文件
    
    参数:
        pdb_id: 要下载的 PDB ID
        output_dir: 输出目录
        target_name: 重命名后的文件名（不含扩展名），如果为 None 则使用 pdb_id
        skip_existing: 是否跳过已存在的文件
    
    返回:
        下载是否成功
    """
    pdb_id = pdb_id.strip().lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    # 确定最终的文件名
    if target_name:
        final_output_file = output_dir / f"{target_name}.pdb"
        temp_output_file = output_dir / f"{pdb_id}.pdb"  # 临时文件名
    else:
        final_output_file = output_dir / f"{pdb_id}.pdb"
        temp_output_file = final_output_file
    
    # 检查最终文件是否已存在
    if skip_existing and final_output_file.exists():
        logger.info(f"⊘ 跳过 {final_output_file.name} (文件已存在)")
        return True
    
    # 下载到临时文件
    success = download_file(url, temp_output_file)
    
    # 如果下载成功且需要重命名
    if success and target_name and temp_output_file != final_output_file:
        try:
            # 重命名文件
            temp_output_file.rename(final_output_file)
            logger.info(f"✓ 重命名为: {final_output_file.name}")
        except Exception as e:
            logger.warning(f"重命名文件失败: {e}，保留原文件名 {temp_output_file.name}")
            # 如果重命名失败，至少文件已经下载了
    
    return success


def download_fasta(pdb_id: str, output_dir: Path, skip_existing: bool = True) -> bool:
    """
    下载 PDB 的 FASTA 序列文件
    """
    pdb_id = pdb_id.strip().lower()
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    output_file = output_dir / f"{pdb_id}.fasta"
    
    if skip_existing and output_file.exists():
        logger.info(f"⊘ 跳过 {pdb_id}.fasta (文件已存在)")
        return True
    
    return download_file(url, output_file)


def save_query_fasta(pdb_id: str, sequence: str, output_dir: Path, skip_existing: bool = True) -> bool:
    """
    保存从 a3m 文件中提取的查询序列为 FASTA 文件
    """
    output_file = output_dir / f"{pdb_id}.fasta"
    
    if skip_existing and output_file.exists():
        logger.info(f"⊘ 跳过 {pdb_id}.fasta (文件已存在)")
        return True
    
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(f">{pdb_id}\n")
            # 每行 80 个字符（标准 FASTA 格式）
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + '\n')
        logger.info(f"✓ 保存查询序列: {output_file.name}")
        return True
    except Exception as e:
        logger.error(f"保存 FASTA 文件时出错: {e}")
        return False


def process_a3m_file(
    a3m_file: Path,
    pdb_output_dir: Path,
    query_fasta_output_dir: Path,
    skip_existing: bool = True,
    download_pdb_file: bool = True
) -> dict:
    """
    处理单个 R 开头的 a3m 文件（CASP16 数据）
    
    返回:
        dict: 处理结果统计
    """
    result = {
        'pdb_downloaded': False,
        'query_fasta_saved': False,
        'pdb_id': None,
        'query_sequence': None,
        'target_name': None
    }
    
    # 获取目标名称（文件名，不含扩展名）
    target_name = a3m_file.stem
    result['target_name'] = target_name
    
    logger.info(f"处理文件: {a3m_file.name} (目标: {target_name})")
    
    # 提取查询序列
    query_sequence = extract_query_sequence_from_a3m(a3m_file)
    if not query_sequence:
        logger.warning(f"  未能提取查询序列")
        return result
    
    result['query_sequence'] = query_sequence
    logger.info(f"  提取到查询序列，长度: {len(query_sequence)}")
    
    # 保存查询序列为 FASTA（使用目标名称）
    if save_query_fasta(target_name, query_sequence, query_fasta_output_dir, skip_existing):
        result['query_fasta_saved'] = True
    
    # 通过序列搜索找到对应的 PDB ID
    if download_pdb_file:
        pdb_id = find_best_pdb_match(query_sequence, target_name)
        
        if pdb_id:
            result['pdb_id'] = pdb_id
            
            # 下载 PDB 文件，并使用目标名称重命名
            if download_pdb(pdb_id, pdb_output_dir, target_name=target_name, skip_existing=skip_existing):
                result['pdb_downloaded'] = True
                logger.info(f"  ✓ 成功下载 PDB 文件: {target_name}.pdb (来源: {pdb_id})")
            else:
                logger.warning(f"  ✗ 下载 PDB 文件失败: {pdb_id}")
        else:
            logger.warning(f"  ✗ 未能找到匹配的 PDB ID")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='从 R 开头的 a3m 文件（CASP16 数据）中提取查询序列，通过序列搜索找到对应的 PDB ID 并下载',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理 rMSA 目录中所有 R 开头的 a3m 文件
  python download_pdb_from_a3m.py --input-dir benchmark_data/RNA-benchmark/single/rMSA
  
  # 指定输出目录
  python download_pdb_from_a3m.py --input-dir rMSA --pdb-output pdb_files --query-fasta-output query_sequences
  
  # 只提取序列，不下载 PDB
  python download_pdb_from_a3m.py --input-dir rMSA --no-download-pdb
"""
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='包含 a3m 文件的输入目录'
    )
    
    parser.add_argument(
        '--pdb-output', '-p',
        type=str,
        default='pdb_downloads',
        help='PDB 文件输出目录（默认: pdb_downloads）'
    )
    
    parser.add_argument(
        '--query-fasta-output',
        type=str,
        default='query_sequences',
        help='查询序列 FASTA 文件输出目录（默认: query_sequences）'
    )
    
    parser.add_argument(
        '--no-download-pdb',
        action='store_true',
        help='不下载 PDB 文件（只提取和保存查询序列）'
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
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.quiet:
        logger.setLevel(logging.ERROR)
    
    # 解析路径
    input_dir = Path(args.input_dir)
    pdb_output_dir = Path(args.pdb_output)
    query_fasta_output_dir = Path(args.query_fasta_output)
    
    # 检查输入目录
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)
    
    # 只查找 R 开头的 a3m 文件（CASP16 数据）
    a3m_files = sorted([f for f in input_dir.glob('*.a3m') if f.stem.startswith('R')])
    if not a3m_files:
        logger.warning(f"在 {input_dir} 中未找到 R 开头的 a3m 文件")
        sys.exit(0)
    
    logger.info("=" * 80)
    logger.info(f"开始处理 R 开头的 a3m 文件（CASP16 数据）")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"找到 {len(a3m_files)} 个 R 开头的 a3m 文件")
    logger.info(f"PDB 输出目录: {pdb_output_dir}")
    logger.info(f"查询序列输出目录: {query_fasta_output_dir}")
    logger.info("=" * 80)
    
    # 统计信息
    stats = {
        'total': len(a3m_files),
        'pdb_downloaded': 0,
        'pdb_failed': 0,
        'pdb_not_found': 0,
        'query_fasta_saved': 0,
        'query_fasta_failed': 0
    }
    
    # 处理每个文件
    for i, a3m_file in enumerate(a3m_files, 1):
        logger.info(f"\n[{i}/{len(a3m_files)}] 处理: {a3m_file.name}")
        
        result = process_a3m_file(
            a3m_file=a3m_file,
            pdb_output_dir=pdb_output_dir,
            query_fasta_output_dir=query_fasta_output_dir,
            skip_existing=not args.force,
            download_pdb_file=not args.no_download_pdb
        )
        
        # 更新统计
        if result['query_fasta_saved']:
            stats['query_fasta_saved'] += 1
        elif result['query_sequence']:
            stats['query_fasta_failed'] += 1
        
        if not args.no_download_pdb:
            if result['pdb_downloaded']:
                stats['pdb_downloaded'] += 1
            elif result['pdb_id']:
                stats['pdb_failed'] += 1
            elif result['query_sequence']:
                stats['pdb_not_found'] += 1
    
    # 打印总结
    logger.info("\n" + "=" * 80)
    logger.info("处理完成")
    logger.info("=" * 80)
    logger.info(f"总计处理: {stats['total']} 个文件")
    logger.info(f"\n查询序列 FASTA:")
    logger.info(f"  成功保存: {stats['query_fasta_saved']}")
    logger.info(f"  保存失败: {stats['query_fasta_failed']}")
    if not args.no_download_pdb:
        logger.info(f"\nPDB 文件:")
        logger.info(f"  成功下载: {stats['pdb_downloaded']}")
        logger.info(f"  下载失败: {stats['pdb_failed']}")
        logger.info(f"  未找到匹配: {stats['pdb_not_found']}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

