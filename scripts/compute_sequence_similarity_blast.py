#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 BLAST 计算训练集和测试集的序列相似度
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


def find_blast(blast_path: Optional[str] = None, program: str = "blastn") -> str:
    """
    查找 BLAST 可执行文件
    
    参数:
        blast_path: 用户指定的BLAST路径
        program: BLAST程序名称 (blastn, blastp等)
    
    返回:
        BLAST可执行文件的完整路径
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    
    # 可能的路径
    possible_paths = []
    if blast_path:
        # 如果用户提供了路径
        if os.path.isdir(blast_path):
            # 如果是目录，添加bin子目录
            possible_paths.append(os.path.join(blast_path, "bin", program))
            possible_paths.append(os.path.join(blast_path, program))
        else:
            # 如果是文件路径
            possible_paths.append(blast_path)
    
    possible_paths.extend([
        os.path.join(project_root, "tools", "blast", "bin", program),
        os.path.join(project_root, "tools", "ncbi-blast", "bin", program),
        program,  # 系统PATH
    ])
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
        # 如果是相对路径，尝试直接执行
        if path == program:
            try:
                result = subprocess.run(
                    [path, "-version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except:
                pass
    
    raise FileNotFoundError(
        f"未找到 {program} 可执行文件。\n"
        f"已搜索路径: {possible_paths}\n"
        f"请安装 BLAST+ 或使用 --blast_path 参数指定路径"
    )


def run_blast_alignment(
    query_fasta: str,
    target_fasta: str,
    blast_path: str,
    work_dir: str,
    n_cpu: int,
    task: str,
    logger
) -> str:
    """
    使用 BLAST 进行序列比对
    
    参数:
        query_fasta: 查询序列FASTA文件
        target_fasta: 目标序列FASTA文件
        blast_path: BLAST可执行文件路径（目录或具体文件）
        work_dir: 工作目录
        n_cpu: CPU核心数
        task: BLAST任务类型 (blastn, blastn-short, megablast等)
        logger: 日志记录器
    
    返回:
        结果TSV文件路径
    """
    # 查找blastn和makeblastdb
    blastn = find_blast(blast_path, "blastn")
    makeblastdb = find_blast(blast_path, "makeblastdb")
    
    # 创建BLAST数据库
    blast_db = os.path.join(work_dir, "blast_db")
    result_tsv = os.path.join(work_dir, "blast_results.tsv")
    
    logger.info("创建BLAST数据库...")
    
    # 判断序列类型（DNA/RNA 还是 蛋白质）
    # 通过检查序列内容判断
    test_seqs = read_sequences_from_fasta(target_fasta)
    if test_seqs:
        test_seq = test_seqs[0][1][:100]  # 检查前100个字符
        is_nucleotide = all(c in 'ATCGUNRYKMSWBDHV-' for c in test_seq.upper())
        dbtype = "nucl" if is_nucleotide else "prot"
    else:
        dbtype = "nucl"  # 默认核酸
    
    # 创建数据库
    cmd = [
        makeblastdb,
        "-in", target_fasta,
        "-dbtype", dbtype,
        "-out", blast_db
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"创建BLAST数据库失败: {stderr.decode('utf-8')}")
    
    logger.info(f"运行BLAST比对 (任务类型: {task})...")
    
    # 运行BLAST比对
    # 自定义输出格式：包含query id, subject id, % identity, alignment length, 
    # mismatches, gap opens, q. start, q. end, s. start, s. end, evalue, bit score,
    # query length, subject length, query coverage per subject
    cmd = [
        blastn,
        "-query", query_fasta,
        "-db", blast_db,
        "-out", result_tsv,
        "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen qcovs",
        "-num_threads", str(n_cpu),
        "-task", task,
        "-max_target_seqs", "1000",  # 最多返回1000个比对结果
        "-word_size", "7",           # 降低word size以提高敏感度
        "-reward", "1",              # 匹配奖励
        "-penalty", "-2",            # 错配惩罚
        "-gapopen", "5",             # gap开放惩罚
        "-gapextend", "2",           # gap延伸惩罚
        "-evalue", "10",             # 放宽E-value阈值
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        error_msg = stderr.decode('utf-8') if stderr else "无错误信息"
        stdout_msg = stdout.decode('utf-8') if stdout else ""
        logger.error(f"BLAST命令: {' '.join(cmd)}")
        logger.error(f"返回码: {process.returncode}")
        logger.error(f"stderr: {error_msg}")
        logger.error(f"stdout: {stdout_msg}")
        raise RuntimeError(f"BLAST比对失败 (返回码: {process.returncode}): {error_msg}")
    
    logger.info("BLAST比对完成")
    
    return result_tsv


def parse_blast_output(result_tsv: str) -> Dict[str, List[Dict]]:
    """
    解析 BLAST 输出结果
    
    返回: {query_id: [hit1, hit2, ...]}
    """
    results = {}
    
    if not os.path.exists(result_tsv):
        return results
    
    with open(result_tsv, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            fields = line.split('\t')
            if len(fields) < 15:
                continue
            
            query_id = fields[0]
            subject_id = fields[1]
            pident = float(fields[2])      # 百分比identity
            aln_length = int(fields[3])    # 比对长度
            mismatch = int(fields[4])      # 错配数
            gapopen = int(fields[5])       # gap数
            qstart = int(fields[6])
            qend = int(fields[7])
            sstart = int(fields[8])
            send = int(fields[9])
            evalue = float(fields[10])
            bitscore = float(fields[11])
            qlen = int(fields[12])         # 查询序列长度
            slen = int(fields[13])         # 目标序列长度
            qcovs = float(fields[14])      # 查询覆盖度（百分比）
            
            # 计算目标覆盖度
            tcovs = (abs(send - sstart) + 1) / slen * 100.0
            
            hit = {
                'query_id': query_id,
                'subject_id': subject_id,
                'pident': pident,
                'aln_length': aln_length,
                'mismatch': mismatch,
                'gapopen': gapopen,
                'qstart': qstart,
                'qend': qend,
                'sstart': sstart,
                'send': send,
                'evalue': evalue,
                'bitscore': bitscore,
                'qlen': qlen,
                'slen': slen,
                'qcovs': qcovs,
                'tcovs': tcovs,
            }
            
            if query_id not in results:
                results[query_id] = []
            results[query_id].append(hit)
    
    return results


def find_max_similarity(
    test_sequences: List[Tuple[str, str]],
    training_fasta: str,
    blast_path: str,
    n_cpu: int,
    work_dir: str,
    task: str,
    logger
) -> Dict[str, Dict]:
    """
    对测试集的每条序列，找到与训练集中最相似的序列及其相似度
    
    返回: {
        'test_seq_header': {
            'max_identity': float,
            'max_qcov': float,
            'max_tcov': float,
            'effective_identity': float,  # pident * qcov / 100
            'best_match_header': str,
            'best_match_identity': float,
            'best_match_qcov': float,
            'best_match_tcov': float,
            'best_match_evalue': float,
            'best_match_bitscore': float,
            'best_match_length': int,
            'num_hits': int,
            'query_length': int
        },
        ...
    }
    """
    # 创建测试集FASTA文件
    # BLAST会在遇到空格时截断header，所以我们使用简化的header
    test_fasta = os.path.join(work_dir, "test_sequences.fasta")
    with open(test_fasta, 'w') as f:
        for header, sequence in test_sequences:
            # 使用简化的header（只保留第一个词/部分）
            simple_header = header.split()[0] if ' ' in header else header
            f.write(f">{simple_header}\n")
            f.write(f"{sequence}\n")
    
    # 运行BLAST比对
    result_tsv = run_blast_alignment(
        query_fasta=test_fasta,
        target_fasta=training_fasta,
        blast_path=blast_path,
        work_dir=work_dir,
        n_cpu=n_cpu,
        task=task,
        logger=logger
    )
    
    # 解析结果
    all_results = parse_blast_output(result_tsv)
    
    # 为每条测试序列找到最佳匹配
    results = {}
    
    for test_header, test_sequence in test_sequences:
        # 使用简化的header进行查找（BLAST会截断包含空格的header）
        lookup_header = test_header.split()[0] if ' ' in test_header else test_header
        
        if lookup_header in all_results:
            hits = all_results[lookup_header]
            
            # 计算有效相似度 (pident * qcov / 100) 并找到最佳匹配
            for hit in hits:
                hit['effective_identity'] = hit['pident'] * hit['qcovs'] / 100.0
            
            # 按有效相似度排序，取最高的
            best_hit = max(hits, key=lambda x: x['effective_identity'])
            
            results[test_header] = {
                'max_identity': best_hit['pident'],
                'max_qcov': best_hit['qcovs'],
                'max_tcov': best_hit['tcovs'],
                'effective_identity': best_hit['effective_identity'],
                'best_match_header': best_hit['subject_id'],
                'best_match_identity': best_hit['pident'],
                'best_match_qcov': best_hit['qcovs'],
                'best_match_tcov': best_hit['tcovs'],
                'best_match_evalue': best_hit['evalue'],
                'best_match_bitscore': best_hit['bitscore'],
                'best_match_length': best_hit['aln_length'],
                'best_match_qlen': best_hit['qlen'],
                'best_match_slen': best_hit['slen'],
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
                'best_match_bitscore': 0.0,
                'best_match_length': 0,
                'best_match_qlen': 0,
                'best_match_slen': 0,
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
                "best_match_evalue\tbest_match_bitscore\tbest_match_length\tnum_hits\t"
                "query_length\tbest_match_qlen\tbest_match_slen\n")
        
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
                   f"{data['best_match_bitscore']:.2f}\t"
                   f"{data['best_match_length']}\t"
                   f"{data['num_hits']}\t"
                   f"{data['query_length']}\t"
                   f"{data['best_match_qlen']}\t"
                   f"{data['best_match_slen']}\n")
    
    logger.info(f"结果已保存到: {output_file}")


def ensure_parent_dir(path: str) -> None:
    """确保输出文件的父目录存在"""
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="使用 BLAST 计算训练集和测试集的序列相似度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 使用训练集列表文件:
   python scripts/compute_sequence_similarity_blast.py \\
       --test_set test_sequences.fasta \\
       --training_list processed_data/list/fold-3_train_ids \\
       --training_dir processed_data/sequences \\
       --output similarity_results.tsv

2. 直接使用FASTA文件:
   python scripts/compute_sequence_similarity_blast.py \\
       --test_set test_sequences.fasta \\
       --training_set training_sequences.fasta \\
       --output similarity_results.tsv

3. 指定BLAST路径:
   python scripts/compute_sequence_similarity_blast.py \\
       --test_set test_sequences.fasta \\
       --training_list processed_data/list/fold-3_train_ids \\
       --training_dir processed_data/sequences \\
       --output similarity_results.tsv \\
       --blast_path ./tools/blast/bin

4. 使用不同的BLAST任务类型:
   python scripts/compute_sequence_similarity_blast.py \\
       --test_set test_sequences.fasta \\
       --training_set training_sequences.fasta \\
       --output similarity_results.tsv \\
       --task blastn-short  # 适用于短序列
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
        "--blast_path",
        default=None,
        help="BLAST可执行文件路径或bin目录，默认自动查找"
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=4,
        help="使用的CPU核心数，默认: 4"
    )
    parser.add_argument(
        "--task",
        default="blastn",
        choices=["blastn", "blastn-short", "megablast", "dc-megablast"],
        help="BLAST任务类型，默认: blastn。blastn-short适用于短序列(<30bp)"
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
        work_dir = tempfile.mkdtemp(prefix="blast_similarity_")
    
    try:
        # 执行比对
        logger.info("开始BLAST比对...")
        results = find_max_similarity(
            test_sequences=test_sequences,
            training_fasta=training_fasta_path,
            blast_path=args.blast_path,
            n_cpu=args.n_cpu,
            work_dir=work_dir,
            task=args.task,
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
            logger.info(f"平均有效相似度 (identity × qcov / 100): {sum(effective_identities) / len(effective_identities):.2f}%")
            logger.info(f"最大identity范围: {min(identities):.2f}% - {max(identities):.2f}%")
            
            # 统计不同相似度区间的序列数
            high_sim = sum(1 for i in identities if i >= 80)
            medium_sim = sum(1 for i in identities if 50 <= i < 80)
            low_sim = sum(1 for i in identities if i < 50)
            zero_sim = sum(1 for i in identities if i == 0)
            
            logger.info(f"\nIdentity分布:")
            logger.info(f"  高相似度 (>=80%): {high_sim} ({high_sim/len(identities)*100:.1f}%)")
            logger.info(f"  中等相似度 (50-80%): {medium_sim} ({medium_sim/len(identities)*100:.1f}%)")
            logger.info(f"  低相似度 (<50%): {low_sim} ({low_sim/len(identities)*100:.1f}%)")
            logger.info(f"  无匹配 (0%): {zero_sim} ({zero_sim/len(identities)*100:.1f}%)")
            
            # 统计有效相似度
            high_eff = sum(1 for i in effective_identities if i >= 80)
            medium_eff = sum(1 for i in effective_identities if 50 <= i < 80)
            low_eff = sum(1 for i in effective_identities if i < 50)
            zero_eff = sum(1 for i in effective_identities if i == 0)
            
            logger.info(f"\n有效相似度分布 (identity × qcov / 100):")
            logger.info(f"  高相似度 (>=80%): {high_eff} ({high_eff/len(effective_identities)*100:.1f}%)")
            logger.info(f"  中等相似度 (50-80%): {medium_eff} ({medium_eff/len(effective_identities)*100:.1f}%)")
            logger.info(f"  低相似度 (<50%): {low_eff} ({low_eff/len(effective_identities)*100:.1f}%)")
            logger.info(f"  无匹配 (0%): {zero_eff} ({zero_eff/len(effective_identities)*100:.1f}%)")
        
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

