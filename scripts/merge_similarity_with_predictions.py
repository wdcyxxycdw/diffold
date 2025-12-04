#!/usr/bin/env python3
"""
将测试集-训练集相似性表（TM-score 独立性分析）
与一个或多个模型评估结果（evaluation_results.csv）进行汇总，
在原表格上增加每个模型的「预测TM-score」列，方便对比。

典型用法（RNA-benchmark, d0=5, Diffold + RhoFold 一起对比）:

uv run scripts/merge_similarity_with_predictions.py \
  --similarity_tsv results/dataset_comparision/RNAbenchmark_tm_scores_analysis_d0=5.tsv \
  --evaluation_csvs \
    results/single_diffold_output/evaluation_results_d0=5/evaluation_results.csv \
    results/single_rhofold_output/evaluation_results_d0=5/evaluation_results.csv \
  --model_names Diffold RhoFold \
  --output_tsv results/dataset_comparision/RNAbenchmark_tm_scores_analysis_d0=5_with_pred.tsv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd


def setup_logging():
    """简单日志配置（只输出到控制台）"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 TM-score 独立性表与一个或多个模型 evaluation_results.csv 合并，增加预测 TM-score 列"
    )
    parser.add_argument(
        "--similarity_tsv",
        required=True,
        help="相似性表 TSV 文件路径，例如 results/dataset_comparision/RNAbenchmark_tm_scores_analysis_d0=5.tsv",
    )
    parser.add_argument(
        "--evaluation_csvs",
        nargs="+",
        required=True,
        help="一个或多个模型评估结果 CSV 文件路径，例如 results/single_diffold_output/evaluation_results_d0=5/evaluation_results.csv",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        required=True,
        help="与 evaluation_csvs 一一对应的模型名称，例如 Diffold RhoFold",
    )
    parser.add_argument(
        "--output_tsv",
        required=True,
        help="输出的合并 TSV 文件路径",
    )
    return parser.parse_args()


def load_similarity_df(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """读取独立性分析 TSV，兼容中英文列名"""
    logger.info(f"读取相似性表: {path}")
    df = pd.read_csv(path, sep="\t")

    cols = list(df.columns)

    # 兼容两种表头：
    # 1) 测试样本  最大TM-score  最相似训练样本  高相似度计数
    # 2) sample_name  max_tm_score  ...
    name_col = None
    if "测试样本" in cols:
        name_col = "测试样本"
    elif "sample_name" in cols:
        name_col = "sample_name"
    else:
        raise ValueError(
            f"相似性表必须包含 '测试样本' 或 'sample_name' 列，当前列为: {cols}"
        )

    # 统一为 sample_name 便于 merge
    df = df.rename(columns={name_col: "sample_name"})
    return df, name_col


def load_eval_df(path: Path, model_name: str, logger: logging.Logger) -> pd.DataFrame:
    """
    读取某个模型的 evaluation_results.csv，
    提取 sample_name, tm_score, rmsd, seq_len 方便对比。
    最终报告中会汇总为一个统一的序列长度列 seq_len。
    """
    logger.info(f"读取评估结果 ({model_name}): {path}")
    df = pd.read_csv(path)
    required = {"sample_name", "tm_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"评估文件 {path} 缺少必要列: {missing}，当前列为: {list(df.columns)}"
        )

    # 有些早期文件可能没有 rmsd / seq_len，这里做容错
    has_rmsd = "rmsd" in df.columns
    has_seq_len = "seq_len" in df.columns

    cols = ["sample_name", "tm_score"]
    if has_rmsd:
        cols.append("rmsd")
    if has_seq_len:
        cols.append("seq_len")

    df = df[cols].copy()

    rename_map = {"tm_score": f"{model_name}_TM-score"}
    if has_rmsd:
        rename_map["rmsd"] = f"{model_name}_RMSD"
    # 对于长度，先保留为模型专属列，稍后再汇总为统一的 seq_len
    if has_seq_len:
        rename_map["seq_len"] = f"{model_name}_seq_len"

    df = df.rename(columns=rename_map)
    return df


def merge_tables(
    sim_df: pd.DataFrame,
    eval_dfs: list[pd.DataFrame],
    model_names: list[str],
    original_name_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """按 sample_name 左连接，将多个模型的预测指标加到独立性表里，并生成统一的 seq_len 列"""
    logger.info("开始合并表格（左连接，多模型）")
    merged = sim_df.copy()

    for model_name, eval_df in zip(model_names, eval_dfs):
        tm_col = f"{model_name}_TM-score"
        logger.info(f"  合并模型 {model_name} 的列: {list(eval_df.columns)}")
        merged = pd.merge(merged, eval_df, on="sample_name", how="left")

        missing = merged[tm_col].isna().sum()
        if missing > 0:
            logger.warning(f"模型 {model_name} 有 {missing} 个样本在评估结果中未找到对应的 sample_name")

    # 统一构造一个 seq_len 列（如果存在）
    seq_len_cols = [f"{m}_seq_len" for m in model_names if f"{m}_seq_len" in merged.columns]
    if seq_len_cols:
        logger.info(f"发现模型序列长度列: {seq_len_cols}，将汇总为统一的 'seq_len'")
        # 按行从多个模型长度列中取第一个非空值
        merged["seq_len"] = merged[seq_len_cols].bfill(axis=1).iloc[:, 0]
        # 删除各模型专属长度列，只保留统一的 seq_len
        merged = merged.drop(columns=seq_len_cols)

    # 恢复原来的样本名列名（例如「测试样本」），方便和原表保持一致
    merged = merged.rename(columns={"sample_name": original_name_col})
    return merged


def main() -> int:
    logger = setup_logging()
    args = parse_args()

    similarity_path = Path(args.similarity_tsv)
    output_path = Path(args.output_tsv)

    if not similarity_path.exists():
        logger.error(f"相似性表不存在: {similarity_path}")
        return 1

    if len(args.evaluation_csvs) != len(args.model_names):
        logger.error("evaluation_csvs 与 model_names 数量必须一致")
        return 1

    eval_paths = [Path(p) for p in args.evaluation_csvs]
    for p in eval_paths:
        if not p.exists():
            logger.error(f"评估结果文件不存在: {p}")
            return 1

    sim_df, original_name_col = load_similarity_df(similarity_path, logger)
    eval_dfs = [
        load_eval_df(path, model_name, logger)
        for path, model_name in zip(eval_paths, args.model_names)
    ]

    merged_df = merge_tables(sim_df, eval_dfs, args.model_names, original_name_col, logger)

    # 输出 TSV，保持制表符分隔
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"合并结果已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


