#!/usr/bin/env python3
"""
RHOfold 并行批量推理脚本
支持多GPU并行处理，每个GPU处理不同的样本子集
"""

import argparse
import subprocess
import sys
from pathlib import Path
import torch
import shutil


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


def create_split_files_and_data_dirs(splits: list, base_output_dir: str, data_dir: str) -> list:
    """为每个GPU创建独立的数据目录和样本列表文件"""
    output_dir = Path(base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src_data_dir = Path(data_dir)
    
    gpu_configs = []
    
    for i, samples in enumerate(splits):
        # 为每个GPU创建独立的数据目录
        gpu_data_dir = output_dir / f"_gpu_data_{i}"
        gpu_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制或链接关键子目录（避免复制大文件）
        for subdir in ['pdb', 'sequences', 'rMSA', 'pdb_raw', 'alignments', 'msa']:
            src_subdir = src_data_dir / subdir
            dst_subdir = gpu_data_dir / subdir
            if src_subdir.exists() and not dst_subdir.exists():
                # 使用符号链接而不是复制，节省空间
                try:
                    dst_subdir.symlink_to(src_subdir.resolve(), target_is_directory=True)
                except:
                    # 如果符号链接失败，尝试复制
                    if src_subdir.is_dir():
                        shutil.copytree(src_subdir, dst_subdir, dirs_exist_ok=True)
        
        # 创建 list 目录并写入样本列表
        # 使用 fold-997 避免与 parallel_batch_inference.py 的 fold-999 冲突
        list_dir = gpu_data_dir / "list"
        list_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建验证集文件（使用固定的 fold-997）
        valid_file = list_dir / "valid_fold-997"
        with open(valid_file, 'w') as f:
            for sample in samples:
                f.write(f"{sample}\n")
        
        # 创建训练集文件（放入相同样本，避免空列表报错）
        train_file = list_dir / "fold-997_train_ids"
        with open(train_file, 'w') as f:
            for sample in samples:
                f.write(f"{sample}\n")
        
        print(f"GPU {i}: {len(samples)} 个样本")
        print(f"  数据目录: {gpu_data_dir}")
        print(f"  样本列表: {valid_file}")
        
        gpu_configs.append({
            'gpu_id': i,
            'data_dir': str(gpu_data_dir),
            'num_samples': len(samples),
            'samples': samples
        })
    
    return gpu_configs


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
    
    # 读取样本列表
    if args.sample_list_file:
        # 直接指定样本列表文件
        sample_list_file = Path(args.sample_list_file)
        if not sample_list_file.exists():
            print(f"❌ 样本列表文件不存在: {sample_list_file}")
            return 1
    else:
        # 自动从 data_dir/casp*_samples.txt 或第一个可用的样本列表文件
        data_dir_path = Path(args.data_dir)
        
        # 尝试找到样本列表文件
        possible_files = [
            data_dir_path / "casp16_samples.txt",
            data_dir_path / "casp15_samples.txt",
            data_dir_path / "samples.txt",
        ]
        
        sample_list_file = None
        for f in possible_files:
            if f.exists():
                sample_list_file = f
                break
        
        # 如果还是没找到，尝试从 list 目录找任意一个 valid_fold-* 文件
        if sample_list_file is None:
            list_dir = data_dir_path / "list"
            if list_dir.exists():
                valid_files = sorted(list_dir.glob("valid_fold-*"))
                if valid_files:
                    sample_list_file = valid_files[0]
        
        if sample_list_file is None:
            print(f"❌ 未找到样本列表文件，请使用 --sample_list_file 参数指定")
            return 1
        
        print(f"自动检测到样本列表: {sample_list_file}")
    
    # 分割样本列表并创建GPU专用数据目录
    print(f"\n📋 分割样本列表: {sample_list_file}")
    splits = split_sample_list(str(sample_list_file), num_gpus)
    gpu_configs = create_split_files_and_data_dirs(
        splits, 
        args.output_dir, 
        args.data_dir
    )
    
    # 构建并启动多个推理进程
    processes = []
    
    print(f"\n🔄 启动 {num_gpus} 个并行推理进程...")
    print("=" * 60)
    
    for config in gpu_configs:
        gpu_id = config['gpu_id']
        gpu_data_dir = config['data_dir']
        
        # 为每个GPU创建独立的输出目录
        gpu_output_dir = Path(args.output_dir) / f"gpu_{gpu_id}"
        
        # 构建命令（使用固定的 fold-997 避免与 Diffold 的 fold-999 冲突）
        cmd = [
            sys.executable,  # python
            "batch_inference_rhofold.py",
            "--rhofold_checkpoint", args.rhofold_checkpoint,
            "--data_dir", gpu_data_dir,
            "--output_dir", str(gpu_output_dir),
            "--fold", "997",
            "--device", f"cuda:{gpu_id}",
            "--max_sequence_length", str(args.max_sequence_length),
            "--num_workers", str(args.num_workers),
            "--log_level", args.log_level,
            "--usalign_path", args.usalign_path,  # US-align路径
        ]
        
        # 添加可选参数
        if args.use_msa:
            cmd.append("--use_msa")
        
        if args.single_seq_pred:
            cmd.append("--single_seq_pred")
        
        if args.msa_dir:
            cmd.extend(["--msa_dir", args.msa_dir])
        
        if args.relax_steps is not None:
            cmd.extend(["--relax_steps", str(args.relax_steps)])
        
        if args.max_samples:
            cmd.extend(["--max_samples", str(args.max_samples)])
        
        print(f"GPU {gpu_id}: 启动推理进程...")
        print(f"  样本数: {config['num_samples']}")
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
    
    # 清理临时GPU数据目录
    if args.cleanup_temp:
        print(f"\n🧹 清理临时文件...")
        for gpu_id in range(num_gpus):
            gpu_data_dir = Path(args.output_dir) / f"_gpu_data_{gpu_id}"
            if gpu_data_dir.exists():
                try:
                    shutil.rmtree(gpu_data_dir)
                    print(f"  已删除: {gpu_data_dir}")
                except Exception as e:
                    print(f"  ⚠️  无法删除 {gpu_data_dir}: {e}")
    
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
        json_file = Path(gpu_output_dir) / "rhofold_test_results.json"
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
    merged_json = output_path / "rhofold_merged_results.json"
    with open(merged_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  JSON: {merged_json}")
    
    # 保存CSV
    merged_csv = output_path / "rhofold_merged_results.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(merged_csv, index=False)
    print(f"  CSV: {merged_csv}")
    
    # 保存详细指标
    merged_detailed = output_path / "rhofold_merged_detailed_metrics.json"
    with open(merged_detailed, 'w') as f:
        json.dump(all_detailed_metrics, f, indent=2, default=str)
    print(f"  详细指标: {merged_detailed}")
    
    # 生成合并报告
    generate_merged_report(all_results, output_path / "rhofold_merged_report.txt")
    print(f"  报告: {output_path / 'rhofold_merged_report.txt'}")
    
    print(f"✅ 结果合并完成! 共 {len(all_results)} 个样本")


def generate_merged_report(results: list, report_file: Path):
    """生成合并报告"""
    import numpy as np
    
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']
    
    with open(report_file, 'w') as f:
        f.write("RHOfold 并行批量推理合并报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总样本数: {len(results)}\n")
        f.write(f"成功样本数: {len(successful_results)}\n")
        f.write(f"失败样本数: {len(failed_results)}\n")
        f.write(f"成功率: {len(successful_results)/len(results)*100:.2f}%\n\n")
        
        if successful_results:
            # RMSD统计
            rmsd_values = [r['rmsd'] for r in successful_results if 'rmsd' in r]
            if rmsd_values:
                f.write("RMSD统计:\n")
                f.write(f"  平均值: {np.mean(rmsd_values):.4f}\n")
                f.write(f"  中位数: {np.median(rmsd_values):.4f}\n")
                f.write(f"  标准差: {np.std(rmsd_values):.4f}\n")
                f.write(f"  最小值: {np.min(rmsd_values):.4f}\n")
                f.write(f"  最大值: {np.max(rmsd_values):.4f}\n\n")
            
            # 其他指标
            metrics_keys = ['tm_score', 'lddt', 'clash_score']
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
        description="RHOfold 多GPU并行批量推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 自动检测所有GPU并行推理（自动查找样本列表）:
   python parallel_batch_inference_rhofold.py \\
       --data_dir ./benchmark_data/casp16 \\
       --rhofold_checkpoint ./pretrained/model_20221010_params.pt \\
       --output_dir ./rhofold_parallel_output \\
       --use_msa \\
       --relax_steps 1000

2. 指定样本列表文件和GPU数量:
   python parallel_batch_inference_rhofold.py \\
       --data_dir ./benchmark_data/casp16 \\
       --sample_list_file ./benchmark_data/casp16/casp16_samples.txt \\
       --rhofold_checkpoint ./pretrained/model_20221010_params.pt \\
       --num_gpus 4 \\
       --output_dir ./rhofold_parallel_output
        """
    )
    
    # 必需参数
    parser.add_argument("--data_dir", required=True,
                       help="数据目录路径")
    parser.add_argument("--rhofold_checkpoint", required=True,
                       help="RhoFold模型检查点路径")
    parser.add_argument("--output_dir", required=True,
                       help="输出目录")
    
    # 样本列表（可选，如果不指定则自动检测）
    parser.add_argument("--sample_list_file", default=None,
                       help="样本列表文件路径（可选，如不指定则自动从data_dir查找）")
    
    # GPU设置
    parser.add_argument("--num_gpus", default="auto",
                       help="使用的GPU数量（auto=自动检测，或指定数字）")
    
    # 数据参数
    parser.add_argument("--max_sequence_length", type=int, default=256,
                       help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="数据加载器工作进程数")
    parser.add_argument("--use_msa", action="store_true", default=False,
                       help="是否使用MSA")
    
    # MSA相关参数
    parser.add_argument("--single_seq_pred", action="store_true", default=False,
                       help="使用单序列预测（不使用MSA）")
    parser.add_argument("--msa_dir", default=None,
                       help="MSA文件目录路径")
    
    # Amber relaxation参数
    parser.add_argument("--relax_steps", type=int, default=None,
                       help="Amber relaxation步数（默认: None，不进行relaxation）")
    
    # 指标计算参数
    parser.add_argument("--usalign_path", default="./USalign/USalign",
                       help="US-align可执行文件路径 (用于计算权威指标)")
    
    # 其他参数
    parser.add_argument("--log_level", default="INFO",
                       help="日志级别")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大处理样本数（用于测试）")
    parser.add_argument("--merge_results", action="store_true", default=True,
                       help="是否合并所有GPU的结果")
    parser.add_argument("--cleanup_temp", action="store_true", default=True,
                       help="是否在完成后清理临时GPU数据目录")
    
    args = parser.parse_args()
    
    return run_parallel_inference(args)


if __name__ == "__main__":
    exit(main())

