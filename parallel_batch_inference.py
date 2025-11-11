#!/usr/bin/env python3
"""
并行批量推理脚本
支持多GPU并行处理，每个GPU处理不同的样本子集
"""

import argparse
import subprocess
import sys
from pathlib import Path
import torch


def split_sample_list(sample_list_file: str, num_gpus: int) -> list:
    """将样本列表分割成多个子列表"""
    with open(sample_list_file, 'r') as f:
        samples = [line.strip() for line in f if line.strip()]
    
    # 计算每个GPU处理的样本数
    samples_per_gpu = len(samples) // num_gpus
    remainder = len(samples) % num_gpus
    
    # 分割样本
    splits = []
    start_idx = 0
    
    for i in range(num_gpus):
        # 前几个GPU多处理一个样本（如果有余数）
        end_idx = start_idx + samples_per_gpu + (1 if i < remainder else 0)
        splits.append(samples[start_idx:end_idx])
        start_idx = end_idx
    
    return splits


def create_split_files(splits: list, base_output_dir: str) -> list:
    """创建分割后的样本列表文件"""
    output_dir = Path(base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_files = []
    for i, samples in enumerate(splits):
        split_file = output_dir / f"samples_gpu{i}.txt"
        with open(split_file, 'w') as f:
            for sample in samples:
                f.write(f"{sample}\n")
        split_files.append(str(split_file))
        print(f"GPU {i}: {len(samples)} 个样本 -> {split_file}")
    
    return split_files


def run_parallel_inference(args):
    """运行并行推理"""
    
    # 检测可用GPU
    if args.num_gpus == 'auto':
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("❌ 未检测到可用的GPU")
            return 1
    else:
        num_gpus = int(args.num_gpus)
        available_gpus = torch.cuda.device_count()
        if num_gpus > available_gpus:
            print(f"⚠️  警告: 请求 {num_gpus} 个GPU，但只有 {available_gpus} 个可用")
            num_gpus = available_gpus
    
    print(f"🚀 使用 {num_gpus} 个GPU进行并行推理")
    
    # 分割样本列表
    print(f"\n📋 分割样本列表: {args.sample_list_file}")
    splits = split_sample_list(args.sample_list_file, num_gpus)
    
    # 创建临时目录和分割文件
    temp_dir = Path(args.output_dir) / "_parallel_temp"
    split_files = create_split_files(splits, str(temp_dir))
    
    # 构建并启动多个推理进程
    processes = []
    
    print(f"\n🔄 启动 {num_gpus} 个并行推理进程...")
    print("=" * 60)
    
    for gpu_id, split_file in enumerate(split_files):
        # 为每个GPU创建独立的输出目录
        gpu_output_dir = Path(args.output_dir) / f"gpu_{gpu_id}"
        
        # 构建命令
        cmd = [
            sys.executable,  # python
            "batch_inference_metrics.py",
            "--data_dir", args.data_dir,
            "--checkpoint_path", args.checkpoint_path,
            "--rhofold_checkpoint", args.rhofold_checkpoint,
            "--output_dir", str(gpu_output_dir),
            "--sample_list_file", split_file,
            "--device", f"cuda:{gpu_id}",
            "--max_sequence_length", str(args.max_sequence_length),
            "--num_workers", str(args.num_workers),
            "--num_sampling", str(args.num_sampling),
            "--selection_strategy", args.selection_strategy,
            "--log_level", args.log_level,
        ]
        
        # 添加可选参数
        if args.lora_path:
            cmd.extend(["--lora_path", args.lora_path])
        
        if args.save_all_samples:
            cmd.append("--save_all_samples")
        
        if not args.use_msa:
            cmd.append("--no-use_msa")
        
        print(f"GPU {gpu_id}: 启动推理进程...")
        print(f"  样本数: {len(splits[gpu_id])}")
        print(f"  输出目录: {gpu_output_dir}")
        
        # 启动进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        processes.append((gpu_id, process, gpu_output_dir))
    
    print("=" * 60)
    print("\n⏳ 等待所有GPU完成推理...")
    
    # 等待所有进程完成
    results = []
    for gpu_id, process, output_dir in processes:
        stdout, stderr = process.communicate()
        returncode = process.returncode
        
        if returncode == 0:
            print(f"✅ GPU {gpu_id} 完成")
            results.append((gpu_id, True, output_dir))
        else:
            print(f"❌ GPU {gpu_id} 失败 (返回码: {returncode})")
            print(f"错误信息:\n{stderr}")
            results.append((gpu_id, False, output_dir))
    
    # 统计结果
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print("\n" + "=" * 60)
    print("📊 并行推理完成!")
    print("=" * 60)
    print(f"成功: {successful}/{len(results)} 个GPU")
    print(f"失败: {failed}/{len(results)} 个GPU")
    
    # 合并结果
    if args.merge_results and successful > 0:
        print(f"\n🔄 合并结果...")
        merge_results(results, args.output_dir)
    else:
        print(f"\n📁 各GPU的结果保存在:")
        for gpu_id, success, output_dir in results:
            if success:
                print(f"  GPU {gpu_id}: {output_dir}")
    
    return 0 if failed == 0 else 1


def merge_results(results: list, output_dir: str):
    """合并多个GPU的推理结果"""
    import json
    import pandas as pd
    
    output_path = Path(output_dir)
    
    all_results = []
    all_detailed_metrics = {}
    
    # 收集所有结果
    for gpu_id, success, gpu_output_dir in results:
        if not success:
            continue
        
        # 读取JSON结果
        json_file = Path(gpu_output_dir) / "batch_inference_results.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                gpu_results = json.load(f)
                all_results.extend(gpu_results)
        
        # 读取详细指标
        detailed_file = Path(gpu_output_dir) / "detailed_metrics.json"
        if detailed_file.exists():
            with open(detailed_file, 'r') as f:
                gpu_detailed = json.load(f)
                all_detailed_metrics.update(gpu_detailed)
    
    if not all_results:
        print("⚠️  警告: 没有可合并的结果")
        return
    
    # 保存合并后的结果
    merged_json = output_path / "merged_results.json"
    with open(merged_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  JSON: {merged_json}")
    
    # 保存CSV
    merged_csv = output_path / "merged_results.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(merged_csv, index=False)
    print(f"  CSV: {merged_csv}")
    
    # 保存详细指标
    merged_detailed = output_path / "merged_detailed_metrics.json"
    with open(merged_detailed, 'w') as f:
        json.dump(all_detailed_metrics, f, indent=2, default=str)
    print(f"  详细指标: {merged_detailed}")
    
    # 生成合并报告
    generate_merged_report(all_results, output_path / "merged_report.txt")
    print(f"  报告: {output_path / 'merged_report.txt'}")
    
    print(f"✅ 结果合并完成! 共 {len(all_results)} 个样本")


def generate_merged_report(results: list, report_file: Path):
    """生成合并报告"""
    import numpy as np
    
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']
    
    with open(report_file, 'w') as f:
        f.write("并行批量推理合并报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总样本数: {len(results)}\n")
        f.write(f"成功样本数: {len(successful_results)}\n")
        f.write(f"失败样本数: {len(failed_results)}\n")
        f.write(f"成功率: {len(successful_results)/len(results)*100:.2f}%\n\n")
        
        if successful_results:
            # RMSD统计
            rmsd_values = [r.get('best_rmsd', r.get('avg_rmsd', float('inf'))) 
                          for r in successful_results]
            if rmsd_values:
                f.write("RMSD统计:\n")
                f.write(f"  平均值: {np.mean(rmsd_values):.4f}\n")
                f.write(f"  中位数: {np.median(rmsd_values):.4f}\n")
                f.write(f"  标准差: {np.std(rmsd_values):.4f}\n")
                f.write(f"  最小值: {np.min(rmsd_values):.4f}\n")
                f.write(f"  最大值: {np.max(rmsd_values):.4f}\n\n")
            
            # 其他指标
            metrics_keys = ['avg_tm_score', 'avg_lddt', 'avg_clash_score']
            for metric in metrics_keys:
                values = [r[metric] for r in successful_results if metric in r]
                if values:
                    f.write(f"{metric}:\n")
                    f.write(f"  平均值: {np.mean(values):.4f}\n")
                    f.write(f"  中位数: {np.median(values):.4f}\n")
                    f.write(f"  标准差: {np.std(values):.4f}\n\n")
        
        if failed_results:
            f.write("失败样本:\n")
            f.write("-" * 30 + "\n")
            for result in failed_results:
                f.write(f"{result['sample_name']}: {result.get('error', 'unknown')}\n")


def main():
    parser = argparse.ArgumentParser(
        description="多GPU并行批量推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 自动检测所有GPU并行推理:
   python parallel_batch_inference.py \\
       --sample_list_file samples.txt \\
       --checkpoint_path model.pt \\
       --lora_path lora/ \\
       --output_dir results/

2. 指定使用4个GPU:
   python parallel_batch_inference.py \\
       --sample_list_file samples.txt \\
       --num_gpus 4 \\
       --checkpoint_path model.pt \\
       --output_dir results/
        """
    )
    
    # 必需参数
    parser.add_argument("--sample_list_file", required=True,
                       help="样本列表文件")
    parser.add_argument("--checkpoint_path", required=True,
                       help="模型检查点路径")
    parser.add_argument("--data_dir", required=True,
                       help="数据目录")
    parser.add_argument("--output_dir", required=True,
                       help="输出目录")
    
    # GPU设置
    parser.add_argument("--num_gpus", default="auto",
                       help="使用的GPU数量（auto=自动检测，或指定数字）")
    
    # 模型参数
    parser.add_argument("--rhofold_checkpoint", 
                       default="./pretrained/model_20221010_params.pt",
                       help="RhoFold检查点路径")
    parser.add_argument("--lora_path", default=None,
                       help="LoRA适配器路径（可选）")
    
    # 数据参数
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_msa", action="store_true", default=True)
    
    # 采样参数
    parser.add_argument("--num_sampling", type=int, default=1)
    parser.add_argument("--selection_strategy", default="rmsd",
                       choices=['rmsd', 'tm_score', 'lddt', 'clash_score', 'composite'])
    parser.add_argument("--save_all_samples", action="store_true", default=False)
    
    # 其他参数
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--merge_results", action="store_true", default=True,
                       help="是否合并所有GPU的结果")
    
    args = parser.parse_args()
    
    return run_parallel_inference(args)


if __name__ == "__main__":
    exit(main())

