#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 EMBOSS needleall 进行全局比对，计算训练集和测试集之间的序列相似度。

对测试集的每条序列，找到与训练集中最相似的序列及其全局 identity。
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

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def read_sequences_from_fasta(fasta_file: str) -> List[Tuple[str, str]]:
    """
    从FASTA文件读取所有序列
    返回: [(header, sequence), ...]
    """
    sequences = []
    current_seq: List[str] = []
    current_header: Optional[str] = None

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq and current_header:
                    sequences.append((current_header, "".join(current_seq).upper()))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        # 添加最后一个序列
        if current_seq and current_header:
            sequences.append((current_header, "".join(current_seq).upper()))

    return sequences


def read_id_list(list_file: str) -> List[str]:
    """
    从列表文件读取ID列表（每行一个ID）
    返回: [id1, id2, ...]
    """
    ids: List[str] = []
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def load_training_sequences_from_list(
    training_list_file: str,
    training_dir: str,
    logger: logging.Logger,
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
    logger.info(f"读取训练集ID列表: {training_list_file}")
    training_ids = read_id_list(training_list_file)
    logger.info(f"训练集包含 {len(training_ids)} 个ID")

    # 创建临时FASTA文件
    temp_fasta = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta")
    temp_fasta_path = temp_fasta.name
    temp_fasta.close()

    training_dir_path = Path(training_dir)
    if not training_dir_path.exists():
        raise FileNotFoundError(f"训练序列目录不存在: {training_dir}")

    found_count = 0
    missing_count = 0

    with open(temp_fasta_path, "w") as out_f:
        for seq_id in training_ids:
            # 支持多种扩展名
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
                if missing_count <= 10:
                    logger.warning(
                        f"未找到序列文件: {seq_id} (尝试了 {[str(p) for p in possible_files]})"
                    )
                continue

            sequences = read_sequences_from_fasta(str(seq_file))
            for header, sequence in sequences:
                if not header or header.strip() == "":
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


def ensure_parent_dir(path: str) -> None:
    """确保输出文件的父目录存在"""
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def run_needleall(
    query_fasta: str,
    target_fasta: str,
    gapopen: float,
    gapextend: float,
    work_dir: str,
    logger: logging.Logger,
) -> str:
    """
    使用 needleall 对 query_fasta 与 target_fasta 做全局比对

    返回 needleall 输出文件路径
    """
    needleall = "needleall"  # 假定已安装在 PATH 中

    out_file = os.path.join(work_dir, "needleall_out.txt")

    cmd = [
        needleall,
        "-asequence",
        query_fasta,
        "-bsequence",
        target_fasta,
        "-gapopen",
        str(gapopen),
        "-gapextend",
        str(gapextend),
        "-aformat3",
        "simple",
        "-auto",
        "-outfile",
        out_file,
    ]

    logger.info("运行 needleall 全局比对...")
    logger.info("命令: %s", " ".join(cmd))

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        logger.error("needleall 运行失败")
        logger.error("stdout: %s", stdout.decode("utf-8") if stdout else "")
        logger.error("stderr: %s", stderr.decode("utf-8") if stderr else "")
        raise RuntimeError(f"needleall 运行失败，返回码 {process.returncode}")

    return out_file


def parse_needleall_output(out_file: str) -> Dict[str, List[Dict]]:
    """
    解析 needleall simple 格式输出

    返回: {query_id: [hit1, hit2, ...]}
    每个 hit: {'query_id', 'target_id', 'identity', 'length', 'similarity', 'gaps', 'score'}
    """
    results: Dict[str, List[Dict]] = {}

    with open(out_file, "r") as f:
        current_query: Optional[str] = None
        current_target: Optional[str] = None
        current_identity: Optional[float] = None
        current_length: Optional[int] = None
        current_similarity: Optional[float] = None
        current_gaps: Optional[float] = None
        current_score: Optional[float] = None

        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# 1:"):
                # query id
                # 形如: "# 1: q1"
                current_query = line.split(":", 1)[1].strip().split()[0]
            elif line.startswith("# 2:"):
                # target id
                current_target = line.split(":", 1)[1].strip().split()[0]
            elif line.startswith("# Length:"):
                # 形如: "# Length: 8"
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        current_length = int(parts[2])
                    except ValueError:
                        current_length = None
            elif line.startswith("# Identity:"):
                # 形如: "# Identity:       7/8 (87.5%)"
                # 我们取括号中的百分比
                if "(" in line and "%" in line:
                    try:
                        percent_str = line.split("(", 1)[1].split("%", 1)[0]
                        current_identity = float(percent_str)
                    except ValueError:
                        current_identity = None
            elif line.startswith("# Similarity:"):
                # 可选，不一定要用
                if "(" in line and "%" in line:
                    try:
                        percent_str = line.split("(", 1)[1].split("%", 1)[0]
                        current_similarity = float(percent_str)
                    except ValueError:
                        current_similarity = None
            elif line.startswith("# Gaps:"):
                # 形如: "# Gaps:           0/8 ( 0.0%)"
                if "(" in line and "%" in line:
                    try:
                        percent_str = line.split("(", 1)[1].split("%", 1)[0]
                        current_gaps = float(percent_str)
                    except ValueError:
                        current_gaps = None
            elif line.startswith("# Score:"):
                # 形如: "# Score: 31.0"
                parts = line.split()
                try:
                    current_score = float(parts[-1])
                except (ValueError, IndexError):
                    current_score = None
            elif line.startswith("#======================================="):
                # 一个新的比对块开始；如果上一个块的信息完整，则存入
                if (
                    current_query is not None
                    and current_target is not None
                    and current_identity is not None
                ):
                    hit = {
                        "query_id": current_query,
                        "target_id": current_target,
                        "identity": current_identity,
                        "length": current_length,
                        "similarity": current_similarity,
                        "gaps": current_gaps,
                        "score": current_score,
                    }
                    results.setdefault(current_query, []).append(hit)

                # 重置当前块信息（但保留 query/target 会在后面重新赋值）
                current_query = None
                current_target = None
                current_identity = None
                current_length = None
                current_similarity = None
                current_gaps = None
                current_score = None

        # 文件结束时再写入一次最后的块
        if (
            current_query is not None
            and current_target is not None
            and current_identity is not None
        ):
            hit = {
                "query_id": current_query,
                "target_id": current_target,
                "identity": current_identity,
                "length": current_length,
                "similarity": current_similarity,
                "gaps": current_gaps,
                "score": current_score,
            }
            results.setdefault(current_query, []).append(hit)

    return results


def compute_max_identity_per_query(
    test_sequences: List[Tuple[str, str]],
    all_hits: Dict[str, List[Dict]],
) -> Dict[str, Dict]:
    """
    对每个测试序列，从 needleall 的 all_hits 中找到 identity 最高的匹配
    """
    results: Dict[str, Dict] = {}

    # 建立 header -> 简短ID 的映射（needleall 使用的是 FASTA ID，
    # 即 '>' 后遇到的第一个空格前的字段）
    header_to_id: Dict[str, str] = {}
    for header, _seq in test_sequences:
        short_id = header.split()[0]
        header_to_id[header] = short_id

    # 反向映射：id -> 原始header（用于写结果）
    id_to_header: Dict[str, str] = {v: k for k, v in header_to_id.items()}

    for header, sequence in test_sequences:
        qid = header_to_id[header]
        hits = all_hits.get(qid, [])

        if hits:
            # 取 identity 最高的 hit
            best_hit = max(hits, key=lambda h: h.get("identity", 0.0))
            results[header] = {
                "max_identity": best_hit.get("identity", 0.0),
                "best_match": best_hit.get("target_id", "N/A"),
                "aln_length": best_hit.get("length", 0) or 0,
                "similarity": best_hit.get("similarity", 0.0)
                if best_hit.get("similarity") is not None
                else 0.0,
                "gaps": best_hit.get("gaps", 0.0)
                if best_hit.get("gaps") is not None
                else 0.0,
                "score": best_hit.get("score", 0.0)
                if best_hit.get("score") is not None
                else 0.0,
                "query_length": len(sequence),
            }
        else:
            results[header] = {
                "max_identity": 0.0,
                "best_match": "N/A",
                "aln_length": 0,
                "similarity": 0.0,
                "gaps": 0.0,
                "score": 0.0,
                "query_length": len(sequence),
            }

    return results


def write_results(results: Dict[str, Dict], output_file: str, logger: logging.Logger):
    """将结果写入TSV文件"""
    with open(output_file, "w") as f:
        f.write(
            "test_sequence\tmax_identity\tbest_match\taln_length\t"
            "similarity\tgaps\tscore\tquery_length\n"
        )
        for header, data in results.items():
            f.write(
                f"{header}\t"
                f"{data['max_identity']:.2f}\t"
                f"{data['best_match']}\t"
                f"{data['aln_length']}\t"
                f"{data['similarity']:.2f}\t"
                f"{data['gaps']:.2f}\t"
                f"{data['score']:.2f}\t"
                f"{data['query_length']}\n"
            )

    logger.info(f"结果已保存到: {output_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="使用 EMBOSS needleall 进行全局比对，计算训练集与测试集的序列相似度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 使用训练集列表文件:
   uv run python scripts/compute_sequence_similarity_needle.py \\
       --test_set benchmark_data/casp16/test_sequences_combined.fasta \\
       --training_list processed_data/list/fold-3_train_ids \\
       --training_dir processed_data/sequences \\
       --output benchmark_data/casp16/train_test_similarity_needle.tsv

2. 直接使用训练集FASTA文件:
   uv run python scripts/compute_sequence_similarity_needle.py \\
       --test_set test_sequences.fasta \\
       --training_set training_sequences.fasta \\
       --output similarity_results_needle.tsv
""",
    )

    parser.add_argument(
        "--test_set",
        required=True,
        help="测试集FASTA文件路径",
    )
    parser.add_argument(
        "--training_set",
        default=None,
        help="训练集FASTA文件路径（与 --training_list 二选一）",
    )
    parser.add_argument(
        "--training_list",
        default=None,
        help="训练集ID列表文件路径（如 processed_data/list/fold-3_train_ids），与 --training_set 二选一",
    )
    parser.add_argument(
        "--training_dir",
        default=None,
        help="训练序列文件所在目录（当使用 --training_list 时必需），如 processed_data/sequences",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出TSV文件路径",
    )
    parser.add_argument(
        "--gapopen",
        type=float,
        default=10.0,
        help="needleall gap open 罚分，默认 10.0",
    )
    parser.add_argument(
        "--gapextend",
        type=float,
        default=0.5,
        help="needleall gap extend 罚分，默认 0.5",
    )
    parser.add_argument(
        "--work_dir",
        default=None,
        help="工作目录（用于存储临时文件），默认: 系统临时目录",
    )

    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    # 检查测试集文件
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

    # 确定训练集FASTA路径
    training_fasta_path: Optional[str] = None
    temp_training_fasta: Optional[str] = None

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
            logger,
        )
        temp_training_fasta = training_fasta_path
    else:
        if not os.path.exists(args.training_set):
            logger.error(f"训练集文件不存在: {args.training_set}")
            return 1
        training_fasta_path = args.training_set
        logger.info("使用直接训练集FASTA文件模式")

    # 读取测试集序列（主要用于统计长度）
    logger.info("读取测试集序列...")
    test_sequences = read_sequences_from_fasta(args.test_set)
    logger.info("测试集包含 %d 条序列", len(test_sequences))

    # 工作目录
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
        work_dir = args.work_dir
    else:
        work_dir = tempfile.mkdtemp(prefix="needle_similarity_")

    try:
        # 运行 needleall
        out_file = run_needleall(
            query_fasta=args.test_set,
            target_fasta=training_fasta_path,
            gapopen=args.gapopen,
            gapextend=args.gapextend,
            work_dir=work_dir,
            logger=logger,
        )

        # 解析输出
        all_hits = parse_needleall_output(out_file)

        # 计算每条测试序列的最大 identity
        results = compute_max_identity_per_query(test_sequences, all_hits)

        # 写入结果
        ensure_parent_dir(args.output)
        write_results(results, args.output, logger)

    finally:
        # 清理临时训练集FASTA
        if temp_training_fasta and os.path.exists(temp_training_fasta):
            os.unlink(temp_training_fasta)
        # 清理工作目录（如果是自动创建的）
        if not args.work_dir and os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())


