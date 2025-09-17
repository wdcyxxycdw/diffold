import argparse
import os
import sys
import tempfile
from pathlib import Path

# 确保可以在未安装包的情况下从项目根目录导入 rhofold
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rhofold.data.balstn import BLASTN


def write_temp_fasta(sequence: str, work_dir: str) -> str:
    """
    Write a single-sequence FASTA file for BLAST query.
    Header is fixed to ">query" to align with existing examples.
    """
    sequence = sequence.strip().replace(" ", "").replace("\n", "")
    if not sequence:
        raise ValueError("输入的序列为空")
    fasta_path = os.path.join(work_dir, "query.fasta")
    with open(fasta_path, "w") as f:
        f.write(">query\n")
        f.write(sequence + "\n")
    return fasta_path


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def read_sequence_from_fasta(fasta_path: str) -> str:
    """
    Read the first sequence from a FASTA file (simple parser to avoid hard deps).
    """
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"找不到输入FASTA文件: {fasta_path}")
    seq_lines = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_lines:
                    break
                else:
                    continue
            seq_lines.append(line)
    if not seq_lines:
        raise ValueError(f"FASTA文件中未找到序列: {fasta_path}")
    return "".join(seq_lines)


def build_msa(
    *,
    input_fas: str,
    output_a3m: str,
    database_dpath: str,
    binary_dpath: str,
    n_cpu: int,
):
    from rhofold.utils import get_logger

    ensure_parent_dir(output_a3m)

    # Minimal console logger
    logger = get_logger(stream=True, log_path=None)

    # Validate resources early with helpful messages
    databases = [
        os.path.join(database_dpath, "rnacentral.fasta"),
        os.path.join(database_dpath, "nt"),
    ]
    missing = [p for p in databases if not os.path.exists(p)]
    if missing:
        logger.error(f"缺少数据库文件: {missing}")
        logger.error("请先运行 ./database/bin/builddb.sh 构建数据库，或指定 --database_dpath")
        raise FileNotFoundError(f"缺少数据库文件: {missing}")

    blastn_bin = os.path.join(binary_dpath, "blastn")
    if not os.path.exists(blastn_bin):
        logger.error(f"未找到 BLASTN 可执行文件: {blastn_bin}")
        logger.error("请设置 --binary_dpath 为包含 blastn/parse_blastn_local.pl/reformat.pl 的目录")
        raise FileNotFoundError(f"未找到 BLASTN 可执行文件: {blastn_bin}")

    blast = BLASTN(binary_dpath=binary_dpath, databases=databases, n_cpu=n_cpu)
    blast.query(input_fasta_path=input_fas, output_msa_path=output_a3m, logger=logger)


def build_msa_online(
    *,
    sequence: str,
    output_a3m: str,
    email: str,
    hit_limit: int = 250,
    program: str = "blastn",
    database: str = "nt",
    expect: float = 10.0,
    megablast: bool = False,
    word_size: int = 7,
    filter_low_complexity: bool = False,
) -> None:
    """
    Query NCBI QBlast online and convert top hits to a3m-like alignment.
    This avoids local database/build steps.
    """
    ensure_parent_dir(output_a3m)

    # Lazy import to keep offline mode lightweight if Bio is absent
    try:
        from Bio import Entrez
        from Bio.Blast import NCBIWWW, NCBIXML
    except Exception as e:
        raise RuntimeError("需要 Biopython 才能进行在线查询，请先安装 biopython: pip install biopython") from e

    if not email:
        raise ValueError("在线查询需要提供 --email 以遵守NCBI使用规范")

    Entrez.email = email

    # NCBIWWW.qblast parameters (use descriptions/alignments to control outputs)
    # Keep defaults compatible with current NCBI settings
    handle = NCBIWWW.qblast(
        program=program,
        database=database,
        sequence=sequence,
        expect=expect,
        descriptions=hit_limit,
        alignments=hit_limit,
        format_type="XML",
        megablast=megablast,
        word_size=word_size,
        filter=filter_low_complexity,
    )

    # Parse all records (qblast may return multiple)
    records = list(NCBIXML.parse(handle))

    # Write A3M
    with open(output_a3m, "w") as out_f:
        # Write master query
        out_f.write(">query\n")
        out_f.write(sequence.strip().replace(" ", "").replace("\n", "") + "\n")
        num_written = 0
        for rec in records:
            for alignment in rec.alignments:
                if num_written >= hit_limit:
                    break
                if not alignment.hsps:
                    continue
                hsp = alignment.hsps[0]

                accession = getattr(alignment, "accession", None) or alignment.title.split()[0]
                header = f"{accession}_{hsp.query_start}_{hsp.query_end}"

                # Build subject aligned to full query length (reference-based MSA)
                aligned_subject_chars = ["-"] * len(sequence)
                qpos = int(hsp.query_start) - 1  # 0-based index into query
                q_aln = hsp.query
                s_aln = hsp.sbjct

                valid = {"A", "C", "G", "T", "U"}
                for qch, sch in zip(q_aln, s_aln):
                    if qch != "-":
                        # Map this alignment column to the corresponding query position
                        sch_up = sch.upper()
                        aligned_subject_chars[qpos] = sch_up if sch_up in valid else "-"
                        qpos += 1

                aligned_subject = "".join(aligned_subject_chars)

                out_f.write(f">{header}\n")
                out_f.write(aligned_subject + "\n")
                num_written += 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "查询RNA序列的MSA并输出为a3m格式。可直接输入原始序列或提供FASTA文件。"
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--sequence",
        type=str,
        help="原始核酸序列（仅限A/U/G/C/T和'-'），将自动写入临时FASTA查询",
    )
    src.add_argument(
        "--input_fasta",
        type=str,
        help="输入FASTA文件路径（包含单条查询序列）",
    )
    parser.add_argument(
        "--output_a3m",
        type=str,
        required=True,
        help="输出a3m文件路径，例如 ./processed_data/rMSA/sample.a3m",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="使用NCBI在线BLASTN进行查询（无需本地数据库）",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="用于NCBI在线服务的Email（开启 --online 时必填）",
    )
    # Online tuning params (good defaults for short RNA fragments)
    parser.add_argument(
        "--online_expect",
        type=float,
        default=10.0,
        help="在线BLAST的E值阈值 (default: 10.0)",
    )
    parser.add_argument(
        "--online_word_size",
        type=int,
        default=7,
        help="在线BLAST的词长 (default: 7, 更适合短序列)",
    )
    parser.add_argument(
        "--online_megablast",
        action="store_true",
        help="使用megablast服务 (默认关闭，更适合短序列)",
    )
    parser.add_argument(
        "--online_filter_low_complexity",
        action="store_true",
        help="开启低复杂度屏蔽 (默认关闭)",
    )
    parser.add_argument(
        "--database_dpath",
        type=str,
        default="./database",
        help="数据库根目录（包含 rnacentral.fasta 与 nt）",
    )
    parser.add_argument(
        "--binary_dpath",
        type=str,
        default="./rhofold/data/bin",
        help="包含 blastn/parse_blastn_local.pl/reformat.pl 的目录",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=4,
        help="BLAST 搜索使用的CPU核心数",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Determine sequence and/or local FASTA
    if args.input_fasta:
        input_fas = args.input_fasta
        seq_text = read_sequence_from_fasta(input_fas)
    else:
        seq_text = args.sequence
        input_fas = None

    if args.online:
        build_msa_online(
            sequence=seq_text,
            output_a3m=args.output_a3m,
            email=args.email,
            hit_limit=250,
            expect=args.online_expect,
            megablast=args.online_megablast,
            word_size=args.online_word_size,
            filter_low_complexity=args.online_filter_low_complexity,
        )
    else:
        # Local BLAST path
        if input_fas is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                input_fas = write_temp_fasta(seq_text, tmpdir)
                build_msa(
                    input_fas=input_fas,
                    output_a3m=args.output_a3m,
                    database_dpath=args.database_dpath,
                    binary_dpath=args.binary_dpath,
                    n_cpu=args.n_cpu,
                )
                return 0

        build_msa(
            input_fas=input_fas,
            output_a3m=args.output_a3m,
            database_dpath=args.database_dpath,
            binary_dpath=args.binary_dpath,
            n_cpu=args.n_cpu,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())


