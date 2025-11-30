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
        
        # 为每个GPU创建独立的数据目录副本（避免临时文件冲突）
        gpu_data_dir = Path(args.output_dir) / f"_gpu_data_{gpu_id}"
        gpu_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制必要的数据目录结构
        src_data_dir = Path(args.data_dir)
        
        # 复制或链接关键子目录（避免复制大文件）
        for subdir in ['pdb', 'sequences', 'rMSA', 'pdb_raw']:
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
        
        # 构建命令
        cmd = [
            sys.executable,  # python
            "batch_inference.py",  # 使用新的简化推理脚本
            "--data_dir", str(gpu_data_dir),  # 使用GPU专用的数据目录
            "--checkpoint_path", args.checkpoint_path,
            "--rhofold_checkpoint", args.rhofold_checkpoint,
            "--output_dir", str(gpu_output_dir),
            "--sample_list_file", split_file,
            "--device", f"cuda:{gpu_id}",
            "--max_sequence_length", str(args.max_sequence_length),
            "--num_workers", str(args.num_workers),
            "--num_sampling", str(args.num_sampling),
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
    
    # 创建合并的PDB文件目录
    merged_pdb_dir = output_path / "merged_pdb_files"
    merged_pdb_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有结果
    for gpu_id, success, gpu_output_dir in results:
        if not success:
            continue
        
        # 读取JSON结果
        json_file = Path(gpu_output_dir) / "inference_results.json"
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
        
        # 复制PDB文件到合并目录
        gpu_pdb_dir = Path(gpu_output_dir) / "pdb_files"
        if gpu_pdb_dir.exists():
            for pdb_file in gpu_pdb_dir.glob("*.pdb"):
                # 复制PDB文件
                dest_file = merged_pdb_dir / pdb_file.name
                if not dest_file.exists():  # 避免重复
                    shutil.copy2(pdb_file, dest_file)
    
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
    
    # 生成推理状态报告
    generate_merged_report(all_results, output_path / "inference_status_report.txt")
    print(f"  推理状态报告: {output_path / 'inference_status_report.txt'}")
    
    # 统计PDB文件
    pdb_count = len(list(merged_pdb_dir.glob("*.pdb")))
    print(f"  PDB文件: {merged_pdb_dir} ({pdb_count} 个文件)")
    
    print(f"✅ 结果合并完成! 共 {len(all_results)} 个样本")


def generate_merged_report(results: list, report_file: Path):
    """生成推理状态报告（仅统计推理成功/失败，不包含评估指标）"""
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']
    
    with open(report_file, 'w') as f:
        f.write("并行批量推理状态报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总样本数: {len(results)}\n")
        f.write(f"成功推理: {len(successful_results)}\n")
        f.write(f"失败推理: {len(failed_results)}\n")
        f.write(f"成功率: {len(successful_results)/len(results)*100:.2f}%\n\n")
        
        if successful_results:
            f.write(f"成功样本列表 ({len(successful_results)} 个):\n")
            f.write("-" * 30 + "\n")
            for result in successful_results:
                f.write(f"  ✓ {result['sample_name']}\n")
            f.write("\n")
        
        if failed_results:
            f.write(f"失败样本列表 ({len(failed_results)} 个):\n")
            f.write("-" * 30 + "\n")
            for result in failed_results:
                error_msg = result.get('error', 'unknown')
                f.write(f"  ✗ {result['sample_name']}: {error_msg}\n")
            f.write("\n")
        
        f.write("=" * 50 + "\n")
        f.write("注意: 此报告仅包含推理状态统计\n")
        f.write("如需结构评估指标，请使用 evaluate_structures.py 进行评估\n")
        f.write("=" * 50 + "\n")


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
    parser.add_argument("--max_sequence_length", type=int, default=1024,
                       help="最大序列长度（用于分配内存，不再限制输入）")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_msa", action="store_true", default=True)
    
    # 采样参数
    parser.add_argument("--num_sampling", type=int, default=1)
    parser.add_argument("--save_all_samples", action="store_true", default=False)
    
    # 其他参数
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--merge_results", action="store_true", default=True,
                       help="是否合并所有GPU的结果")
    parser.add_argument("--cleanup_temp", action="store_true", default=True,
                       help="是否在完成后清理临时GPU数据目录")
    
    args = parser.parse_args()
    
    return run_parallel_inference(args)


if __name__ == "__main__":
    exit(main())

