#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 MMseqs2 全局比对模式计算训练集和测试集的序列相似度
对测试集的每条序列，找到与训练集中最相似的序列及其全局相似度
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 确保可以在未安装包的情况下从项目根目录导入
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def read_sequences_from_fasta(fasta_file: str) -> List[Tuple[str, str]]:
    """
    从FASTA文件读取所有序列
    返回: [(header, sequence), ...]
    """
    sequences = []
    current_seq = []
    current_header = None
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_seq and current_header:
                    sequences.append((current_header, ''.join(current_seq).upper()))
                current_header = line[1:]  # 去掉 '>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # 添加最后一个序列
        if current_seq and current_header:
            sequences.append((current_header, ''.join(current_seq).upper()))
    
    return sequences


def read_id_list(list_file: str) -> List[str]:
    """
    从列表文件读取ID列表（每行一个ID）
    返回: [id1, id2, ...]
    """
    ids = []
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                ids.append(line)
    return ids


def load_training_sequences_from_list(
    training_list_file: str,
    training_dir: str,
    logger
) -> str:
    """
    根据列表文件从训练目录中加载指定的序列文件，并合并成一个临时FASTA文件
    
    参数:
        training_list_file: 训练集ID列表文件路径
        training_dir: 训练序列文件所在目录
        logger: 日志记录器
    
    返回:
        临时FASTA文件路径
    """
    # 读取ID列表
    logger.info(f"读取训练集ID列表: {training_list_file}")
    training_ids = read_id_list(training_list_file)
    logger.info(f"训练集包含 {len(training_ids)} 个ID")
    
    # 创建临时FASTA文件
    temp_fasta = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta')
    temp_fasta_path = temp_fasta.name
    temp_fasta.close()
    
    training_dir_path = Path(training_dir)
    if not training_dir_path.exists():
        raise FileNotFoundError(f"训练序列目录不存在: {training_dir}")
    
    # 统计信息
    found_count = 0
    missing_count = 0
    
    # 从训练目录中读取对应的序列文件
    with open(temp_fasta_path, 'w') as out_f:
        for seq_id in training_ids:
            # 尝试多种可能的文件名格式
            possible_files = [
                training_dir_path / f"{seq_id}.fasta",
                training_dir_path / f"{seq_id}.fa",
                training_dir_path / f"{seq_id}.fna",
            ]
            
            seq_file = None
            for pf in possible_files:
                if pf.exists():
                    seq_file = pf
                    break
            
            if seq_file is None:
                missing_count += 1
                if missing_count <= 10:  # 只显示前10个缺失文件的警告
                    logger.warning(f"未找到序列文件: {seq_id} (尝试了 {[str(p) for p in possible_files]})")
                continue
            
            # 读取序列文件并写入临时文件
            sequences = read_sequences_from_fasta(str(seq_file))
            for header, sequence in sequences:
                # 使用原始header，如果为空则使用seq_id
                if not header or header.strip() == '':
                    header = seq_id
                out_f.write(f">{header}\n")
                out_f.write(f"{sequence}\n")
                found_count += 1
    
    if missing_count > 10:
        logger.warning(f"... 还有 {missing_count - 10} 个文件未找到")
    
    logger.info(f"成功加载 {found_count} 条序列，缺失 {missing_count} 个文件")
    
    if found_count == 0:
        raise RuntimeError("未能加载任何训练序列，请检查训练目录和ID列表文件")
    
    return temp_fasta_path


def find_mmseqs2(mmseqs2_path: Optional[str] = None) -> str:
    """
    查找 MMseqs2 可执行文件
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    
    # 可能的路径
    possible_paths = []
    if mmseqs2_path:
        possible_paths.append(mmseqs2_path)
    
    possible_paths.extend([
        os.path.join(project_root, "tools", "mmseqs2", "mmseqs", "bin", "mmseqs"),
        os.path.join(project_root, "tools", "mmseqs2", "bin", "mmseqs"),
        "mmseqs",  # 系统PATH
    ])
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
        # 如果是相对路径，尝试直接执行
        if path == "mmseqs":
            try:
                result = subprocess.run(
                    [path, "version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except:
                pass
    
    raise FileNotFoundError(
        f"未找到 MMseqs2 可执行文件。\n"
        f"已搜索路径: {possible_paths}\n"
        f"请下载 MMseqs2 或设置 --mmseqs2_path 参数"
    )


def run_mmseqs2_global_alignment(
    query_fasta: str,
    target_fasta: str,
    mmseqs2_path: str,
    work_dir: str,
    n_cpu: int,
    logger
) -> str:
    """
    使用 MMseqs2 进行全局比对
    
    返回结果TSV文件路径
    """
    mmseqs = find_mmseqs2(mmseqs2_path)
    
    # 创建临时数据库
    query_db = os.path.join(work_dir, "query_db")
    target_db = os.path.join(work_dir, "target_db")
    result_db = os.path.join(work_dir, "result_db")
    result_tsv = os.path.join(work_dir, "result.tsv")
    
    logger.info("创建MMseqs2数据库...")
    
    # 创建查询数据库
    cmd = [mmseqs, "createdb", query_fasta, query_db]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"创建查询数据库失败: {stderr.decode('utf-8')}")
    
    # 创建目标数据库
    cmd = [mmseqs, "createdb", target_fasta, target_db]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"创建目标数据库失败: {stderr.decode('utf-8')}")
    
    logger.info("运行MMseqs2全局比对...")
    
    # 运行全局比对
    # alignment-mode 3 = 全局比对
    # search-type 3 = 核酸序列比对
    cmd = [
        mmseqs, "search",
        query_db, target_db, result_db, work_dir,
        "--threads", str(n_cpu),
        "--search-type", "3",     # 核酸序列比对
        "--alignment-mode", "3",  # 全局比对模式
        "--min-seq-id", "0.0",    # 不设置最小相似度阈值
        "--max-seqs", "1000",     # 最多返回1000个结果
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        error_msg = stderr.decode('utf-8') if stderr else "无错误信息"
        stdout_msg = stdout.decode('utf-8') if stdout else ""
        logger.error(f"MMseqs2命令: {' '.join(cmd)}")
        logger.error(f"返回码: {process.returncode}")
        logger.error(f"stderr: {error_msg}")
        logger.error(f"stdout: {stdout_msg}")
        raise RuntimeError(f"MMseqs2比对失败 (返回码: {process.returncode}): {error_msg}")
    
    logger.info("转换比对结果...")
    
    # 转换结果为TSV格式
    # 输出格式: query_id, target_id, seq_id, aln_score, seq_id, q_start, q_end, q_len, t_start, t_end, t_len, evalue, pident, nident, qcov, tcov
    cmd = [
        mmseqs, "convertalis",
        query_db, target_db, result_db, result_tsv,
        "--format-mode", "0",  # TSV格式
        "--format-output", "query,target,pident,qcov,tcov,qstart,qend,qlen,tstart,tend,tlen,evalue,bits"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"转换结果失败: {stderr.decode('utf-8')}")
    
    return result_tsv


def parse_mmseqs2_output(result_tsv: str) -> Dict[str, List[Dict]]:
    """
    解析 MMseqs2 输出结果
    返回: {query_id: [hit1, hit2, ...]}
    """
    results = {}
    
    with open(result_tsv, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            fields = line.split('\t')
            if len(fields) < 12:
                continue
            
            query_id = fields[0]
            target_id = fields[1]
            pident = float(fields[2])      # 百分比identity
            qcov = float(fields[3])       # 查询序列覆盖度
            tcov = float(fields[4])       # 目标序列覆盖度
            qstart = int(fields[5])
            qend = int(fields[6])
            qlen = int(fields[7])
            tstart = int(fields[8])
            tend = int(fields[9])
            tlen = int(fields[10])
            evalue = float(fields[11])
            bits = float(fields[12]) if len(fields) > 12 else 0.0
            
            hit = {
                'query_id': query_id,
                'target_id': target_id,
                'pident': pident,
                'qcov': qcov,
                'tcov': tcov,
                'qstart': qstart,
                'qend': qend,
                'qlen': qlen,
                'tstart': tstart,
                'tend': tend,
                'tlen': tlen,
                'evalue': evalue,
                'bits': bits,
                'aln_length': abs(qend - qstart) + 1,
            }
            
            if query_id not in results:
                results[query_id] = []
            results[query_id].append(hit)
    
    return results


def find_max_similarity(
    test_sequences: List[Tuple[str, str]],
    training_fasta: str,
    mmseqs2_path: str,
    n_cpu: int,
    work_dir: str,
    logger
) -> Dict[str, Dict]:
    """
    对测试集的每条序列，找到与训练集中最相似的序列及其全局相似度
    
    返回: {
        'test_seq_header': {
            'max_identity': float,
            'max_qcov': float,
            'max_tcov': float,
            'effective_identity': float,  # pident * qcov
            'best_match_header': str,
            'best_match_identity': float,
            'best_match_qcov': float,
            'best_match_tcov': float,
            'best_match_evalue': float,
            'best_match_bits': float,
            'num_hits': int
        },
        ...
    }
    """
    # 创建测试集FASTA文件
    test_fasta = os.path.join(work_dir, "test_sequences.fasta")
    with open(test_fasta, 'w') as f:
        for header, sequence in test_sequences:
            f.write(f">{header}\n")
            f.write(f"{sequence}\n")
    
    # 运行MMseqs2全局比对
    result_tsv = run_mmseqs2_global_alignment(
        query_fasta=test_fasta,
        target_fasta=training_fasta,
        mmseqs2_path=mmseqs2_path,
        work_dir=work_dir,
        n_cpu=n_cpu,
        logger=logger
    )
    
    # 解析结果
    all_results = parse_mmseqs2_output(result_tsv)
    
    # 为每条测试序列找到最佳匹配
    results = {}
    
    for test_header, test_sequence in test_sequences:
        if test_header in all_results:
            hits = all_results[test_header]
            
            # 计算有效相似度 (pident * qcov) 并找到最佳匹配
            for hit in hits:
                hit['effective_identity'] = hit['pident'] * hit['qcov'] / 100.0
            
            # 按有效相似度排序，取最高的
            best_hit = max(hits, key=lambda x: x['effective_identity'])
            
            results[test_header] = {
                'max_identity': best_hit['pident'],
                'max_qcov': best_hit['qcov'],
                'max_tcov': best_hit['tcov'],
                'effective_identity': best_hit['effective_identity'],
                'best_match_header': best_hit['target_id'],
                'best_match_identity': best_hit['pident'],
                'best_match_qcov': best_hit['qcov'],
                'best_match_tcov': best_hit['tcov'],
                'best_match_evalue': best_hit['evalue'],
                'best_match_bits': best_hit['bits'],
                'best_match_qlen': best_hit['qlen'],
                'best_match_tlen': best_hit['tlen'],
                'num_hits': len(hits),
                'query_length': len(test_sequence),
            }
        else:
            # 没有找到匹配
            results[test_header] = {
                'max_identity': 0.0,
                'max_qcov': 0.0,
                'max_tcov': 0.0,
                'effective_identity': 0.0,
                'best_match_header': 'N/A',
                'best_match_identity': 0.0,
                'best_match_qcov': 0.0,
                'best_match_tcov': 0.0,
                'best_match_evalue': 1.0,
                'best_match_bits': 0.0,
                'best_match_qlen': 0,
                'best_match_tlen': 0,
                'num_hits': 0,
                'query_length': len(test_sequence),
            }
    
    return results


def write_results(results: Dict[str, Dict], output_file: str, logger):
    """将结果写入TSV文件"""
    with open(output_file, 'w') as f:
        # 写入表头
        f.write("test_sequence\tmax_identity\tmax_qcov\tmax_tcov\teffective_identity\t"
                "best_match\tbest_match_identity\tbest_match_qcov\tbest_match_tcov\t"
                "best_match_evalue\tbest_match_bits\tnum_hits\tquery_length\t"
                "best_match_qlen\tbest_match_tlen\n")
        
        # 写入数据
        for test_header, data in results.items():
            f.write(f"{test_header}\t"
                   f"{data['max_identity']:.2f}\t"
                   f"{data['max_qcov']:.2f}\t"
                   f"{data['max_tcov']:.2f}\t"
                   f"{data['effective_identity']:.2f}\t"
                   f"{data['best_match_header']}\t"
                   f"{data['best_match_identity']:.2f}\t"
                   f"{data['best_match_qcov']:.2f}\t"
                   f"{data['best_match_tcov']:.2f}\t"
                   f"{data['best_match_evalue']:.2e}\t"
                   f"{data['best_match_bits']:.2f}\t"
                   f"{data['num_hits']}\t"
                   f"{data['query_length']}\t"
                   f"{data['best_match_qlen']}\t"
                   f"{data['best_match_tlen']}\n")
    
    logger.info(f"结果已保存到: {output_file}")


def ensure_parent_dir(path: str) -> None:
    """确保输出文件的父目录存在"""
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="使用 MMseqs2 全局比对模式计算训练集和测试集的序列相似度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 使用训练集列表文件:
   python scripts/compare_train_test_similarity_mmseqs2.py \\
       --test_set test_sequences.fasta \\
       --training_list processed_data/list/fold-3_train_ids \\
       --training_dir processed_data/sequences \\
       --output similarity_results.tsv

2. 直接使用FASTA文件:
   python scripts/compare_train_test_similarity_mmseqs2.py \\
       --test_set test_sequences.fasta \\
       --training_set training_sequences.fasta \\
       --output similarity_results.tsv

3. 指定MMseqs2路径:
   python scripts/compare_train_test_similarity_mmseqs2.py \\
       --test_set test_sequences.fasta \\
       --training_list processed_data/list/fold-3_train_ids \\
       --training_dir processed_data/sequences \\
       --output similarity_results.tsv \\
       --mmseqs2_path ./tools/mmseqs2/mmseqs/bin/mmseqs
        """
    )
    
    parser.add_argument(
        "--test_set",
        required=True,
        help="测试集FASTA文件路径"
    )
    parser.add_argument(
        "--training_set",
        default=None,
        help="训练集FASTA文件路径（与 --training_list 二选一）"
    )
    parser.add_argument(
        "--training_list",
        default=None,
        help="训练集ID列表文件路径（如 processed_data/list/fold-3_train_ids），与 --training_set 二选一"
    )
    parser.add_argument(
        "--training_dir",
        default=None,
        help="训练序列文件所在目录（当使用 --training_list 时必需），如 processed_data/sequences"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出TSV文件路径"
    )
    parser.add_argument(
        "--mmseqs2_path",
        default=None,
        help="MMseqs2可执行文件路径，默认自动查找"
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=4,
        help="使用的CPU核心数，默认: 4"
    )
    parser.add_argument(
        "--work_dir",
        default=None,
        help="工作目录（用于存储临时文件），默认: 系统临时目录"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = logging.getLogger(__name__)
    
    # 检查输入文件
    if not os.path.exists(args.test_set):
        logger.error(f"测试集文件不存在: {args.test_set}")
        return 1
    
    # 检查训练集参数
    if args.training_set and args.training_list:
        logger.error("不能同时指定 --training_set 和 --training_list，请只选择其中一个")
        return 1
    
    if not args.training_set and not args.training_list:
        logger.error("必须指定 --training_set 或 --training_list 之一")
        return 1
    
    # 确定训练集FASTA文件路径
    training_fasta_path = None
    temp_training_fasta = None
    
    if args.training_list:
        if not os.path.exists(args.training_list):
            logger.error(f"训练集列表文件不存在: {args.training_list}")
            return 1
        
        if not args.training_dir:
            logger.error("使用 --training_list 时必须指定 --training_dir")
            return 1
        
        if not os.path.exists(args.training_dir):
            logger.error(f"训练序列目录不存在: {args.training_dir}")
            return 1
        
        logger.info("使用训练集列表文件模式")
        training_fasta_path = load_training_sequences_from_list(
            args.training_list,
            args.training_dir,
            logger
        )
        temp_training_fasta = training_fasta_path
    else:
        if not os.path.exists(args.training_set):
            logger.error(f"训练集文件不存在: {args.training_set}")
            return 1
        training_fasta_path = args.training_set
        logger.info("使用直接FASTA文件模式")
    
    # 读取序列
    logger.info("读取测试集序列...")
    test_sequences = read_sequences_from_fasta(args.test_set)
    logger.info(f"测试集包含 {len(test_sequences)} 条序列")
    
    # 创建工作目录
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
        work_dir = args.work_dir
    else:
        work_dir = tempfile.mkdtemp(prefix="mmseqs2_similarity_")
    
    try:
        # 执行比对
        logger.info("开始MMseqs2全局比对...")
        results = find_max_similarity(
            test_sequences=test_sequences,
            training_fasta=training_fasta_path,
            mmseqs2_path=args.mmseqs2_path,
            n_cpu=args.n_cpu,
            work_dir=work_dir,
            logger=logger
        )
        
        # 写入结果
        ensure_parent_dir(args.output)
        write_results(results, args.output, logger)
        
        # 打印统计信息
        logger.info("\n" + "=" * 80)
        logger.info("统计信息")
        logger.info("=" * 80)
        
        identities = [r['max_identity'] for r in results.values()]
        effective_identities = [r['effective_identity'] for r in results.values()]
        
        if identities:
            logger.info(f"测试序列数: {len(results)}")
            logger.info(f"平均最大identity: {sum(identities) / len(identities):.2f}%")
            logger.info(f"平均有效相似度 (identity × qcov): {sum(effective_identities) / len(effective_identities):.2f}%")
            logger.info(f"最大identity范围: {min(identities):.2f}% - {max(identities):.2f}%")
            
            # 统计不同相似度区间的序列数
            high_sim = sum(1 for i in identities if i >= 80)
            medium_sim = sum(1 for i in identities if 50 <= i < 80)
            low_sim = sum(1 for i in identities if i < 50)
            
            logger.info(f"\nIdentity分布:")
            logger.info(f"  高相似度 (>=80%): {high_sim} ({high_sim/len(identities)*100:.1f}%)")
            logger.info(f"  中等相似度 (50-80%): {medium_sim} ({medium_sim/len(identities)*100:.1f}%)")
            logger.info(f"  低相似度 (<50%): {low_sim} ({low_sim/len(identities)*100:.1f}%)")
            
            # 统计有效相似度
            high_eff = sum(1 for i in effective_identities if i >= 80)
            medium_eff = sum(1 for i in effective_identities if 50 <= i < 80)
            low_eff = sum(1 for i in effective_identities if i < 50)
            
            logger.info(f"\n有效相似度分布 (identity × qcov):")
            logger.info(f"  高相似度 (>=80%): {high_eff} ({high_eff/len(effective_identities)*100:.1f}%)")
            logger.info(f"  中等相似度 (50-80%): {medium_eff} ({medium_eff/len(effective_identities)*100:.1f}%)")
            logger.info(f"  低相似度 (<50%): {low_eff} ({low_eff/len(effective_identities)*100:.1f}%)")
        
        logger.info("=" * 80)
        
    finally:
        # 清理临时训练集FASTA文件
        if temp_training_fasta and os.path.exists(temp_training_fasta):
            os.unlink(temp_training_fasta)
        
        # 清理工作目录（如果不是用户指定的）
        if not args.work_dir and os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

