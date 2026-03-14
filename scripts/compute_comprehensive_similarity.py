#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合相似度分析脚本
同时计算序列相似度（BLAST）和结构相似度（TM-score），并合并模型性能数据
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
import logging
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# ==================== BLAST 相关函数 ====================

def read_sequences_from_fasta(fasta_file: str) -> List[Tuple[str, str]]:
    """从FASTA文件读取所有序列"""
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
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_seq and current_header:
            sequences.append((current_header, ''.join(current_seq).upper()))
    
    return sequences


def read_id_list(list_file: str) -> List[str]:
    """从列表文件读取ID列表"""
    ids = []
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ids.append(line)
    return ids


def load_training_sequences_from_list(training_list_file: str, training_dir: str, logger) -> str:
    """根据列表文件从训练目录中加载序列"""
    logger.info(f"读取训练集序列ID列表: {training_list_file}")
    training_ids = read_id_list(training_list_file)
    logger.info(f"训练集包含 {len(training_ids)} 个ID")
    
    temp_fasta = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta')
    temp_fasta_path = temp_fasta.name
    temp_fasta.close()
    
    training_dir_path = Path(training_dir)
    found_count = 0
    missing_count = 0
    
    with open(temp_fasta_path, 'w') as out_f:
        for seq_id in training_ids:
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
                continue
            
            sequences = read_sequences_from_fasta(str(seq_file))
            for header, sequence in sequences:
                if not header or header.strip() == '':
                    header = seq_id
                out_f.write(f">{header}\n{sequence}\n")
                found_count += 1
    
    logger.info(f"成功加载 {found_count} 条序列，缺失 {missing_count} 个文件")
    return temp_fasta_path


def find_blast(blast_path: Optional[str], program: str = "blastn") -> str:
    """查找BLAST可执行文件"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    possible_paths = []
    
    if blast_path:
        if os.path.isdir(blast_path):
            possible_paths.extend([
                os.path.join(blast_path, "bin", program),
                os.path.join(blast_path, program)
            ])
        else:
            possible_paths.append(blast_path)
    
    possible_paths.extend([
        os.path.join(project_root, "tools", "blast", "bin", program),
        os.path.join(project_root, "tools", "ncbi-blast-2.15.0+", "bin", program),
        program,
    ])
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
        if path == program:
            try:
                subprocess.run([path, "-version"], capture_output=True, timeout=5)
                return path
            except:
                pass
    
    raise FileNotFoundError(f"未找到 {program} 可执行文件")


def run_blast_similarity(test_fasta: str, training_fasta: str, blast_path: str, 
                        work_dir: str, n_cpu: int, logger) -> Dict[str, Dict]:
    """运行BLAST序列相似度分析"""
    logger.info("=" * 80)
    logger.info("开始BLAST序列相似度分析")
    logger.info("=" * 80)
    
    blastn = find_blast(blast_path, "blastn")
    makeblastdb = find_blast(blast_path, "makeblastdb")
    
    blast_db = os.path.join(work_dir, "blast_db")
    result_tsv = os.path.join(work_dir, "blast_results.tsv")
    
    logger.info("创建BLAST数据库...")
    cmd = [makeblastdb, "-in", training_fasta, "-dbtype", "nucl", "-out", blast_db]
    subprocess.run(cmd, check=True, capture_output=True)
    
    logger.info("运行BLAST比对...")
    cmd = [
        blastn, "-query", test_fasta, "-db", blast_db, "-out", result_tsv,
        "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen qcovs",
        "-num_threads", str(n_cpu), "-task", "blastn",
        "-word_size", "7", "-reward", "1", "-penalty", "-2",
        "-gapopen", "5", "-gapextend", "2", "-evalue", "10",
        "-max_target_seqs", "1000",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    # 解析结果
    results = {}
    test_sequences = read_sequences_from_fasta(test_fasta)
    
    with open(result_tsv, 'r') as f:
        blast_hits = {}
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 15:
                continue
            qid = fields[0]
            if qid not in blast_hits:
                blast_hits[qid] = []
            
            hit = {
                'subject_id': fields[1],
                'pident': float(fields[2]),
                'qcovs': float(fields[14]),
                'evalue': float(fields[10]),
                'num_hits': 1
            }
            hit['effective_identity'] = hit['pident'] * hit['qcovs'] / 100.0
            blast_hits[qid].append(hit)
    
    for header, sequence in test_sequences:
        simple_header = header.split()[0] if ' ' in header else header
        base_id = simple_header.split('_')[0]
        
        if simple_header in blast_hits and blast_hits[simple_header]:
            best_hit = max(blast_hits[simple_header], key=lambda x: x['effective_identity'])
            results[base_id] = {
                'max_identity': best_hit['pident'],
                'max_qcov': best_hit['qcovs'],
                'effective_identity': best_hit['effective_identity'],
                'best_match': best_hit['subject_id'],
                'num_hits': len(blast_hits[simple_header]),
                'query_length': len(sequence)
            }
        else:
            results[base_id] = {
                'max_identity': 0.0,
                'max_qcov': 0.0,
                'effective_identity': 0.0,
                'best_match': 'N/A',
                'num_hits': 0,
                'query_length': len(sequence)
            }
    
    logger.info(f"完成BLAST分析，处理了 {len(results)} 个测试序列")
    return results


# ==================== TM-score 相关函数 ====================

def run_usalign(pdb1: str, pdb2: str, usalign_path: str) -> Optional[float]:
    """运行USalign并返回TM-score"""
    try:
        cmd = [usalign_path, pdb1, pdb2]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return None
        
        for line in result.stdout.split('\n'):
            if 'TM-score=' in line and 'normalized by length of Structure_1' in line:
                match = re.search(r'TM-score=\s*([\d.]+)', line)
                if match:
                    return float(match.group(1))
        return None
    except:
        return None


def get_pdb_files(directory: Path) -> List[Path]:
    """获取目录中所有PDB文件"""
    return sorted(directory.glob("*.pdb"))


def run_tmscore_similarity(test_dir: Path, train_dir: Path, usalign_path: str, 
                          train_id_list: Optional[List[str]], logger) -> Dict[str, Dict]:
    """运行TM-score结构相似度分析"""
    logger.info("=" * 80)
    logger.info("开始TM-score结构相似度分析")
    logger.info("=" * 80)
    
    test_pdbs = get_pdb_files(test_dir)
    train_pdbs = get_pdb_files(train_dir)
    
    if train_id_list:
        train_id_set = set(train_id_list)
        train_pdbs = [p for p in train_pdbs if p.stem in train_id_set]
    
    logger.info(f"测试集样本数: {len(test_pdbs)}")
    logger.info(f"训练集样本数: {len(train_pdbs)}")
    
    results = {}
    
    for i, test_pdb in enumerate(test_pdbs, 1):
        test_name = test_pdb.stem
        logger.info(f"[{i}/{len(test_pdbs)}] 处理: {test_name}")
        
        max_tm_score = 0.0
        max_tm_train = None
        high_similarity_count = 0
        
        for train_pdb in train_pdbs:
            if test_pdb.stem == train_pdb.stem:
                continue
            
            tm_score = run_usalign(str(test_pdb), str(train_pdb), usalign_path)
            
            if tm_score is not None:
                if tm_score > max_tm_score:
                    max_tm_score = tm_score
                    max_tm_train = train_pdb.stem
                if tm_score >= 0.5:
                    high_similarity_count += 1
        
        results[test_name] = {
            '最大TM-score': max_tm_score,
            '最相似训练样本': max_tm_train or 'N/A',
            '高相似度计数': high_similarity_count
        }
        
        logger.info(f"  最大TM-score: {max_tm_score:.4f} (与 {max_tm_train})")
    
    logger.info(f"完成TM-score分析，处理了 {len(results)} 个测试样本")
    return results


# ==================== 数据合并和输出 ====================

def load_performance_data(tm_file: str, logger) -> Dict[str, Dict]:
    """加载模型性能数据"""
    logger.info(f"加载模型性能数据: {tm_file}")
    df = pd.read_csv(tm_file, sep='\t')
    
    results = {}
    for _, row in df.iterrows():
        base_id = str(row['测试样本'])
        results[base_id] = {
            'Diffold_TM-score': row.get('Diffold_TM-score'),
            'Diffold_RMSD': row.get('Diffold_RMSD'),
            'RhoFold+_TM-score': row.get('RhoFold+_TM-score'),
            'RhoFold+_RMSD': row.get('RhoFold+_RMSD'),
        }
    
    logger.info(f"加载了 {len(results)} 个样本的性能数据")
    return results


def classify_leakage_risk(effective_identity: float) -> str:
    """分类数据泄露风险"""
    if pd.isna(effective_identity):
        return 'Unknown'
    elif effective_identity >= 80:
        return 'High'
    elif effective_identity >= 50:
        return 'Medium'
    elif effective_identity > 0:
        return 'Low'
    else:
        return 'None'


def merge_and_output(blast_results: Dict, tm_results: Dict, perf_results: Dict, 
                    output_file: str, logger):
    """合并所有结果并输出"""
    logger.info("=" * 80)
    logger.info("合并结果并生成输出表格")
    logger.info("=" * 80)
    
    # 收集所有样本ID
    all_ids = set(blast_results.keys()) | set(tm_results.keys()) | set(perf_results.keys())
    
    rows = []
    for base_id in sorted(all_ids):
        blast = blast_results.get(base_id, {})
        tm = tm_results.get(base_id, {})
        perf = perf_results.get(base_id, {})
        
        # 只保留两个模型都有结果的样本
        if pd.isna(perf.get('Diffold_TM-score')) or pd.isna(perf.get('RhoFold+_TM-score')):
            continue
        
        row = {
            'base_id': base_id,
            'query_length': blast.get('query_length'),
            'max_identity': blast.get('max_identity'),
            'max_qcov': blast.get('max_qcov'),
            'effective_identity': blast.get('effective_identity'),
            'best_match': blast.get('best_match'),
            'num_hits': blast.get('num_hits'),
            '最大TM-score': tm.get('最大TM-score'),
            '最相似训练样本': tm.get('最相似训练样本'),
            'Diffold_TM-score': perf.get('Diffold_TM-score'),
            'Diffold_RMSD': perf.get('Diffold_RMSD'),
            'RhoFold+_TM-score': perf.get('RhoFold+_TM-score'),
            'RhoFold+_RMSD': perf.get('RhoFold+_RMSD'),
        }
        
        # 添加风险等级
        row['leakage_risk'] = classify_leakage_risk(row['effective_identity'])
        
        rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    
    # 按effective_identity降序排序
    df = df.sort_values('effective_identity', ascending=False, na_position='last')
    
    # 保存结果
    df.to_csv(output_file, sep='\t', index=False, float_format='%.4f')
    logger.info(f"结果已保存到: {output_file}")
    logger.info(f"总样本数: {len(df)}")
    
    # 打印统计信息
    print_statistics(df, logger)


def print_statistics(df: pd.DataFrame, logger):
    """打印统计信息"""
    logger.info("\n" + "=" * 80)
    logger.info("统计摘要")
    logger.info("=" * 80)
    
    # 风险分布
    logger.info("\n【数据泄露风险分布】")
    risk_counts = df['leakage_risk'].value_counts()
    for risk, count in risk_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  {risk:10s}: {count:3d} ({pct:5.1f}%)")
    
    # 按风险等级的性能
    logger.info(f"\n【各风险等级的平均Diffold性能】")
    for risk in ['High', 'Medium', 'Low', 'None']:
        risk_df = df[df['leakage_risk'] == risk]
        if len(risk_df) > 0:
            avg_tm = risk_df['Diffold_TM-score'].mean()
            avg_rmsd = risk_df['Diffold_RMSD'].mean()
            logger.info(f"  {risk:10s} (n={len(risk_df):2d}): TM-score={avg_tm:.4f}, RMSD={avg_rmsd:.4f}")
    
    # 相关性分析
    logger.info("\n【相关性分析】")
    corr_data = df[['effective_identity', 'Diffold_TM-score', '最大TM-score']].dropna()
    if len(corr_data) > 2:
        corr_eff_tm = corr_data['effective_identity'].corr(corr_data['Diffold_TM-score'])
        corr_maxtm_tm = corr_data['最大TM-score'].corr(corr_data['Diffold_TM-score'])
        logger.info(f"  序列相似度 vs 模型性能: {corr_eff_tm:7.4f}")
        logger.info(f"  结构相似度 vs 模型性能: {corr_maxtm_tm:7.4f}")
    
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="综合计算序列和结构相似度，并合并模型性能数据",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 测试集参数
    parser.add_argument('--test_fasta', required=True, help="测试集FASTA文件")
    parser.add_argument('--test_pdb_dir', required=True, help="测试集PDB目录")
    
    # 训练集参数
    parser.add_argument('--train_list', required=True, help="训练集ID列表文件")
    parser.add_argument('--train_seq_dir', required=True, help="训练集序列目录")
    parser.add_argument('--train_pdb_dir', required=True, help="训练集PDB目录")
    
    # 性能数据
    parser.add_argument('--performance_file', required=True, help="模型性能数据TSV文件")
    
    # 工具路径
    parser.add_argument('--blast_path', default=None, help="BLAST路径")
    parser.add_argument('--usalign_path', default='tools/USalign', help="USalign路径")
    
    # 输出
    parser.add_argument('--output', required=True, help="输出TSV文件")
    parser.add_argument('--n_cpu', type=int, default=4, help="CPU核心数")
    parser.add_argument('--work_dir', default=None, help="工作目录")
    
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    
    # 创建工作目录
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
        work_dir = args.work_dir
    else:
        work_dir = tempfile.mkdtemp(prefix="comprehensive_similarity_")
    
    try:
        # 加载训练集序列
        training_fasta = load_training_sequences_from_list(
            args.train_list, args.train_seq_dir, logger
        )
        
        # 运行BLAST分析
        blast_results = run_blast_similarity(
            args.test_fasta, training_fasta, args.blast_path,
            work_dir, args.n_cpu, logger
        )
        
        # 加载训练集ID列表
        train_ids = read_id_list(args.train_list)
        
        # 运行TM-score分析
        tm_results = run_tmscore_similarity(
            Path(args.test_pdb_dir), Path(args.train_pdb_dir),
            args.usalign_path, train_ids, logger
        )
        
        # 加载性能数据
        perf_results = load_performance_data(args.performance_file, logger)
        
        # 合并并输出
        merge_and_output(blast_results, tm_results, perf_results, args.output, logger)
        
        # 清理
        os.unlink(training_fasta)
        
    finally:
        if not args.work_dir and os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

