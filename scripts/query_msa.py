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

    # Disable SSL certificate verification for systems with self-signed certs
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

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

    # Write A3M with deduplication
    with open(output_a3m, "w") as out_f:
        # Write master query
        query_seq = sequence.strip().replace(" ", "").replace("\n", "")
        out_f.write(">query\n")
        out_f.write(query_seq + "\n")
        
        # Track seen sequences for deduplication
        seen_sequences = {query_seq}  # 初始化时包含query序列
        num_written = 0
        num_duplicates = 0
        
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
                
                # 去重：跳过已经见过的序列
                if aligned_subject in seen_sequences:
                    num_duplicates += 1
                    continue
                
                seen_sequences.add(aligned_subject)
                out_f.write(f">{header}\n")
                out_f.write(aligned_subject + "\n")
                num_written += 1
        
        # 打印去重统计信息
        if num_duplicates > 0:
            print(f"  (去重: 移除了 {num_duplicates} 条重复序列)", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "查询RNA序列的MSA并输出为a3m格式。可直接输入原始序列或提供FASTA文件，也支持批量处理。"
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
    src.add_argument(
        "--input_dir",
        type=str,
        help="输入文件夹路径（批量处理模式，将处理文件夹中所有.fasta/.fa文件）",
    )
    parser.add_argument(
        "--output_a3m",
        type=str,
        required=True,
        help="输出a3m文件路径（单文件模式）或输出目录（批量模式，使用--input_dir时）",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=".a3m",
        help="批量模式下输出文件的后缀 (default: .a3m)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="批量模式下跳过已存在的输出文件",
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


def process_single_file(
    input_fas: str,
    output_a3m: str,
    args,
    seq_text: str = None,
) -> bool:
    """
    处理单个文件，返回是否成功
    """
    try:
        if args.online:
            if seq_text is None:
                seq_text = read_sequence_from_fasta(input_fas)
            build_msa_online(
                sequence=seq_text,
                output_a3m=output_a3m,
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
                        output_a3m=output_a3m,
                        database_dpath=args.database_dpath,
                        binary_dpath=args.binary_dpath,
                        n_cpu=args.n_cpu,
                    )
                    return True

            build_msa(
                input_fas=input_fas,
                output_a3m=output_a3m,
                database_dpath=args.database_dpath,
                binary_dpath=args.binary_dpath,
                n_cpu=args.n_cpu,
            )
        return True
    except Exception as e:
        print(f"✗ 处理失败 {os.path.basename(input_fas)}: {e}", file=sys.stderr)
        return False


def batch_process(args):
    """
    批量处理文件夹中的所有FASTA文件
    """
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_a3m)
    
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}", file=sys.stderr)
        return 1
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有FASTA文件
    fasta_extensions = ["*.fasta", "*.fa", "*.fna"]
    fasta_files = []
    for ext in fasta_extensions:
        fasta_files.extend(input_dir.glob(ext))
    
    if not fasta_files:
        print(f"警告: 在 {input_dir} 中未找到FASTA文件 (.fasta/.fa/.fna)", file=sys.stderr)
        return 1
    
    print("=" * 80)
    print(f"批量MSA构建")
    print("=" * 80)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(fasta_files)} 个FASTA文件")
    print(f"模式: {'在线BLAST' if args.online else '本地BLAST'}")
    if args.skip_existing:
        print("跳过已存在的文件: 是")
    print("=" * 80)
    print()
    
    # 处理每个文件
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for i, fasta_file in enumerate(fasta_files, 1):
        # 生成输出文件名
        base_name = fasta_file.stem  # 不带扩展名的文件名
        output_file = output_dir / f"{base_name}{args.output_suffix}"
        
        # 检查是否跳过
        if args.skip_existing and output_file.exists():
            print(f"[{i}/{len(fasta_files)}] ⊘ 跳过 {fasta_file.name} (已存在)")
            skip_count += 1
            continue
        
        print(f"[{i}/{len(fasta_files)}] 处理中: {fasta_file.name} ...", end=" ", flush=True)
        
        # 处理文件
        success = process_single_file(
            input_fas=str(fasta_file),
            output_a3m=str(output_file),
            args=args,
        )
        
        if success:
            print(f"✓ 完成 -> {output_file.name}")
            success_count += 1
        else:
            fail_count += 1
    
    # 汇总统计
    print()
    print("=" * 80)
    print(f"批量处理完成")
    print("=" * 80)
    print(f"总计: {len(fasta_files)} 个文件")
    print(f"成功: {success_count} 个")
    if skip_count > 0:
        print(f"跳过: {skip_count} 个")
    if fail_count > 0:
        print(f"失败: {fail_count} 个")
    print("=" * 80)
    
    return 0 if fail_count == 0 else 1


def main(argv=None):
    args = parse_args(argv)

    # 批量处理模式
    if args.input_dir:
        return batch_process(args)
    
    # 单文件处理模式
    # Determine sequence and/or local FASTA
    if args.input_fasta:
        input_fas = args.input_fasta
        seq_text = read_sequence_from_fasta(input_fas)
    else:
        seq_text = args.sequence
        input_fas = None

    success = process_single_file(
        input_fas=input_fas,
        output_a3m=args.output_a3m,
        args=args,
        seq_text=seq_text,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())


