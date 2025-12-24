#!/usr/bin/env python3
"""
绘制模型性能（如 TM-score）与序列长度之间关系的散点图（带线性拟合）。

特性：
- 支持读取多个 evaluation_results*.csv 文件（不同模型 / 不同数据集）
- 自动从 CSV 中提取 seq_len 和指定性能指标列（默认 tm_score）
- 将多文件结果合并到一张图上，用不同颜色区分数据集，用不同标记区分模型
- 对每个数据集-模型组合分别做线性拟合，并画出拟合直线、在图例中标注斜率和 Pearson r

典型用法示例（单数据集）：
uv run scripts/plot_length_performance.py \\
  --csv-files \\
    results/single_diffold_output/evaluation_results_d0=5/evaluation_results.csv \\
    results/single_rhofold_output/evaluation_results_d0=5/evaluation_results.csv \\
  --labels Diffold RhoFold \\
  --metric tm_score \\
  --output results/plots/length_vs_tmscore_rnabenchmark_d0=5.png

典型用法示例（多数据集合并）：
uv run scripts/plot_length_performance.py \\
  --csv-files \\
    results/single_diffold_output/evaluation_results_d0=5/evaluation_results.csv \\
    results/single_rhofold_output/evaluation_results_d0=5/evaluation_results.csv \\
    results/diffold_casp16/evaluation_results_d0=5/evaluation_results.csv \\
    results/rhofold_casp16_relaxed/evaluation_results_d0=5/evaluation_results.csv \\
  --labels Diffold RhoFold Diffold RhoFold \\
  --datasets RNA-benchmark RNA-benchmark CASP16 CASP16 \\
  --metric tm_score \\
  --output results/plots/length_vs_tmscore_combined_d0=5.png
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 设置字体和样式 - 学术论文标准
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

# 专业学术配色方案（与 compare_models.py 一致）
MODEL_COLORS = {
    'Diffold': '#4472C4',      # 专业蓝色
    'RhoFold': '#ED7D31',      # 专业橙色
    'RhoFold+': '#ED7D31',     # 兼容变体
}

# 支持的指标及其正式名称（统一格式）
METRICS = ["tm_score", "lddt", "gdt_ts", "rmsd"]
METRIC_NAME_MAP = {
    "tm_score": "TM-score",
    "lddt": "lDDT",
    "gdt_ts": "GDT-TS",
    "rmsd": "RMSD (Å)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制模型性能与序列长度之间关系的散点图（带线性拟合）"
    )

    parser.add_argument(
        "--csv-files",
        nargs="+",
        required=True,
        help="一个或多个 evaluation_results*.csv 文件路径",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="与 csv-files 一一对应的标签（通常为模型名）",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="与 csv-files 一一对应的数据集名称（可选，如果不提供则只用模型区分）",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="tm_score",
        help="性能指标列名（默认: tm_score；也可为 all，表示 TM-score/lDDT/GDT-TS/RMSD 的 2x2 汇总图）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/plots/length_vs_performance.png",
        help="输出图片路径（默认: results/plots/length_vs_performance.png）",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[7, 5],
        help="图片大小（宽度 高度），默认 7 5",
    )

    return parser.parse_args()


def load_eval_csv(path: Path, metric: str, label: str, dataset: str = None) -> pd.DataFrame:
    """读取单个 evaluation_results.csv，并提取 seq_len + metric."""
    if not path.exists():
        raise FileNotFoundError(f"评估文件不存在: {path}")

    df = pd.read_csv(path)

    if "seq_len" not in df.columns:
        raise ValueError(f"评估文件 {path} 缺少 'seq_len' 列，请确认已使用最新的 evaluate_structures.py 生成。")
    if metric not in df.columns:
        raise ValueError(f"评估文件 {path} 缺少性能列 '{metric}'，当前列有: {list(df.columns)}")

    out = df[["sample_name", "seq_len", metric]].copy()
    out["model"] = label
    if dataset is not None:
        out["dataset"] = dataset
    return out


def plot_length_performance(
    df: pd.DataFrame,
    metric: str,
    output: Path,
    figsize: List[float],
) -> None:
    """绘制长度-性能散点图，并对每个模型做线性拟合。

    注意：无论是否来自多个数据集，这里都会将所有样本合并在一起，
    仅用颜色区分模型本身，而不区分数据集。
    """
    models = sorted(df["model"].unique())

    fig, ax = plt.subplots(figsize=tuple(figsize))
    legend_entries = []

    for idx, model in enumerate(models):
        sub = df[df["model"] == model].dropna(subset=["seq_len", metric])
        if sub.empty:
            continue

        x = sub["seq_len"].values.astype(float)
        y = sub[metric].values.astype(float)

        # 使用统一的配色方案
        color = MODEL_COLORS.get(model, f'C{idx}')

        # 散点
        ax.scatter(
            x,
            y,
            s=40,
            alpha=0.6,
            color=color,
            edgecolor='black',
            linewidth=0.5,
            label=model,
        )

        # 线性拟合
        if len(sub) >= 2:
            slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                color=color,
                linewidth=2.5,
                alpha=0.8,
            )
            legend_entries.append(
                f"{model}: slope={slope:.4f}, r={r_value:.3f}, p={p_value:.1e}"
            )
        else:
            legend_entries.append(f"{model}: 样本数不足，无法拟合")

    # 指标名美化
    pretty_metric = METRIC_NAME_MAP.get(metric, metric)

    ax.set_xlabel("Sequence length", fontsize=13)
    ax.set_ylabel(pretty_metric, fontsize=13)
    ax.set_title(f"Sequence length vs {pretty_metric}", fontsize=14, pad=15)
    ax.grid(alpha=0.25, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='both', labelsize=11)

    # 图例信息放在图外底部
    text = "\n".join(legend_entries)
    fig.text(
        0.5,
        0.02,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor='gray', linewidth=1),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图片已保存到: {output}")


def load_eval_csv_all(path: Path, label: str, dataset: str = None) -> pd.DataFrame:
    """读取 evaluation_results.csv，提取 seq_len + 所有关心的指标。"""
    if not path.exists():
        raise FileNotFoundError(f"评估文件不存在: {path}")

    df = pd.read_csv(path)

    if "seq_len" not in df.columns:
        raise ValueError(
            f"评估文件 {path} 缺少 'seq_len' 列，请确认已使用最新的 evaluate_structures.py 生成。"
        )

    missing_metrics = [m for m in METRICS if m not in df.columns]
    if missing_metrics:
        raise ValueError(
            f"评估文件 {path} 缺少以下指标列: {missing_metrics}，当前列有: {list(df.columns)}"
        )

    cols = ["sample_name", "seq_len"] + METRICS
    out = df[cols].copy()
    out["model"] = label
    if dataset is not None:
        out["dataset"] = dataset
    return out


def plot_length_performance_grid(
    df: pd.DataFrame,
    output: Path,
    figsize: List[float],
) -> None:
    """绘制 2x2 子图的汇总图：长度 vs TM-score / lDDT / GDT-TS / RMSD。"""
    models = sorted(df["model"].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=tuple(figsize))
    axes = axes.reshape(-1)
    
    # 子图标号
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for panel_idx, (ax, metric) in enumerate(zip(axes, METRICS)):
        pretty_metric = METRIC_NAME_MAP.get(metric, metric)
        legend_entries: List[str] = []

        for idx, model in enumerate(models):
            sub = df[df["model"] == model].dropna(subset=["seq_len", metric])
            if sub.empty:
                continue

            x = sub["seq_len"].values.astype(float)
            y = sub[metric].values.astype(float)
            
            # 使用统一的配色方案
            color = MODEL_COLORS.get(model, f'C{idx}')

            # 只在第一个子图上加图例项，后面用全局 legend
            ax.scatter(
                x,
                y,
                s=25,
                alpha=0.6,
                color=color,
                edgecolor='black',
                linewidth=0.3,
                label=model if metric == METRICS[0] else None,
            )

            if len(sub) >= 2:
                slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(
                    x_line,
                    y_line,
                    color=color,
                    linewidth=2.0,
                    alpha=0.8,
                )
                legend_entries.append(
                    f"{model}: r={r_value:.3f}, p={p_value:.1e}"
                )

        ax.set_xlabel("Sequence length", fontsize=12)
        ax.set_ylabel(pretty_metric, fontsize=12)
        ax.set_title(pretty_metric, fontsize=13, pad=12)
        ax.grid(alpha=0.25, linestyle='--', axis='y')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=10)
        
        # 添加子图标号
        ax.text(-0.12, 1.05, panel_labels[panel_idx], transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='right')

        if legend_entries:
            ax.text(
                0.98,
                0.03,
                "\n".join(legend_entries),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, 
                         edgecolor='gray', linewidth=0.8),
            )

    # 全局图例：基于第一个子图
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(models),
            fontsize=11,
            frameon=True,
            fancybox=False,
            edgecolor='black',
            framealpha=0.9,
        )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"汇总图片已保存到: {output}")


def main() -> int:
    args = parse_args()

    if len(args.csv_files) != len(args.labels):
        raise ValueError("csv-files 与 labels 的数量必须一致")
    
    if args.datasets is not None:
        if len(args.csv_files) != len(args.datasets):
            raise ValueError("csv-files 与 datasets 的数量必须一致")

    csv_paths = [Path(p) for p in args.csv_files]

    all_dfs: List[pd.DataFrame] = []
    for idx, (path, label) in enumerate(zip(csv_paths, args.labels)):
        dataset = args.datasets[idx] if args.datasets is not None else None
        print(
            "读取评估文件: "
            f"{path} (label={label}" + (f", dataset={dataset})" if dataset else ")")
        )
        if args.metric == "all":
            df = load_eval_csv_all(path, label, dataset)
        else:
            df = load_eval_csv(path, args.metric, label, dataset)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"总样本数: {len(combined)}")

    output_path = Path(args.output)

    if args.metric == "all":
        # 绘制 2x2 汇总图
        plot_length_performance_grid(
            combined,
            output=output_path,
            figsize=args.figsize,
        )
    else:
        # 单一指标图
        plot_length_performance(
            combined,
            metric=args.metric,
            output=output_path,
            figsize=args.figsize,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


