#!/usr/bin/env python3
"""
Diffold批量推理脚本
对验证集样本进行批量推理，输出PDB结构文件
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# 导入Diffold相关模块
from diffold.diffold import Diffold
from diffold.dataloader import create_data_loaders
from rhofold.utils import get_device, timing

# 导入PDB转换功能
from diffold.output import diffold_coords_to_pdb, validate_diffold_output


def setup_logging(output_dir: str, log_level: str = "INFO"):
    """设置日志"""
    log_file = Path(output_dir) / "batch_inference.log"
    
    # 创建日志目录
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(getattr(logging, log_level.upper()))
    stream_handler.setFormatter(formatter)
    
    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []  # 清除现有处理器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger


def load_model(config: argparse.Namespace, logger: logging.Logger):
    """加载Diffold模型"""
    logger.info("构建Diffold模型")
    model = Diffold(config, rhofold_checkpoint_path=config.rhofold_checkpoint)
    model = model.to(config.device)
    model.eval()
    
    # 加载检查点
    if config.checkpoint_path:
        logger.info(f"加载检查点: {config.checkpoint_path}")
        try:
            checkpoint = torch.load(config.checkpoint_path, map_location=config.device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("检查点加载完成")
        except Exception as e:
            logger.warning(f"weights_only=True 加载失败，尝试使用 weights_only=False: {e}")
            checkpoint = torch.load(config.checkpoint_path, map_location=config.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("检查点加载完成")
    
    # 加载 LoRA 适配器（如果指定）
    if hasattr(config, 'lora_path') and config.lora_path:
        logger.info("=" * 60)
        logger.info(f"🎯 加载 LoRA 适配器: {config.lora_path}")
        logger.info("=" * 60)
        try:
            from diffold.lora_utils import LoRAManager
            model = LoRAManager.load_lora_weights(model, config.lora_path)
            logger.info("✅ LoRA 适配器加载成功!")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"❌ LoRA 适配器加载失败: {e}")
            raise
    
    return model


def load_validation_data(config: argparse.Namespace, logger: logging.Logger):
    """加载验证数据"""
    logger.info("加载验证数据")
    
    # 支持两种模式：指定样本列表文件 或 使用 fold
    if hasattr(config, 'sample_list_file') and config.sample_list_file:
        # 模式1: 从指定的样本列表文件读取
        sample_list_file = Path(config.sample_list_file)
        if not sample_list_file.exists():
            raise FileNotFoundError(f"样本列表文件不存在: {sample_list_file}")
        
        logger.info(f"从指定文件加载样本列表: {sample_list_file}")
        with open(sample_list_file, 'r') as f:
            sample_names = [line.strip() for line in f if line.strip()]
        
        logger.info(f"样本数量: {len(sample_names)}")
        
        # 创建临时 fold 文件
        temp_list_dir = Path(config.data_dir) / "list"
        temp_list_dir.mkdir(parents=True, exist_ok=True)
        
        temp_valid_file = temp_list_dir / "valid_fold-999"
        with open(temp_valid_file, 'w') as f:
            for name in sample_names:
                f.write(f"{name}\n")
        
        temp_train_file = temp_list_dir / "fold-999_train_ids"
        with open(temp_train_file, 'w') as f:
            for name in sample_names:
                f.write(f"{name}\n")
        
        train_loader, valid_loader = create_data_loaders(
            data_dir=config.data_dir,
            batch_size=1,
            max_length=config.max_sequence_length,
            num_workers=config.num_workers,
            fold=999,
            use_msa=config.use_msa,
            use_all_folds=False,
            world_size=1,
            local_rank=0
        )
    else:
        # 模式2: 使用 fold 方式
        valid_list_file = Path(config.data_dir) / "list" / f"valid_fold-{config.fold}"
        if not valid_list_file.exists():
            raise FileNotFoundError(f"验证集列表文件不存在: {valid_list_file}")
        
        logger.info(f"使用交叉验证 fold-{config.fold}")
        with open(valid_list_file, 'r') as f:
            sample_names = [line.strip() for line in f if line.strip()]
        
        logger.info(f"验证集样本数量: {len(sample_names)}")
        
        train_loader, valid_loader = create_data_loaders(
            data_dir=config.data_dir,
            batch_size=1,
            max_length=config.max_sequence_length,
            num_workers=config.num_workers,
            fold=config.fold,
            use_msa=config.use_msa,
            use_all_folds=False,
            world_size=1,
            local_rank=0
        )
    
    return valid_loader, sample_names


def inference_sample(model: Diffold, 
                     batch: Dict[str, torch.Tensor], 
                     sample_name: str,
                     data_dir: str,
                     output_dir: str,
                     num_sampling: int,
                     save_all_samples: bool,
                     logger: logging.Logger) -> Dict[str, Any]:
    """对单个样本进行推理并保存PDB文件（支持多次采样）"""
    try:
        # 准备输入数据
        device = next(model.parameters()).device
        tokens = batch['tokens'].to(device)
        sequences = batch['sequences']
        coordinates = batch.get('coordinates', None)
        missing_atom_masks = batch.get('missing_atom_masks', None)
        rna_fm_tokens = batch.get('rna_fm_tokens', None)
        
        if coordinates is not None:
            coordinates = coordinates.to(device)
        if missing_atom_masks is not None:
            missing_atom_masks = missing_atom_masks.to(device)
        if rna_fm_tokens is not None:
            rna_fm_tokens = rna_fm_tokens.to(device)
        
        # 获取序列
        batch_sample_name = batch.get('names', [sample_name])[0] if batch.get('names') else sample_name
        
        if batch_sample_name != sample_name:
            logger.warning(f"样本名称不匹配: batch中为 {batch_sample_name}, 期望为 {sample_name}")
            sample_name = batch_sample_name
        
        sequence = sequences[0] if sequences else ""
        
        # 多次采样
        all_samples = []
        
        logger.debug(f"样本 {sample_name}: 开始 {num_sampling} 次采样")
        
        for sample_idx in range(num_sampling):
            # 模型推理
            with torch.no_grad():
                result = model(
                    tokens=tokens,
                    rna_fm_tokens=rna_fm_tokens,
                    seq=sequences,
                    target_coords=coordinates,
                    missing_atom_mask=missing_atom_masks
                )
            
            if result is None:
                logger.warning(f"样本 {sample_name} 采样 {sample_idx+1}: 模型推理返回None")
                continue
            
            # 提取预测坐标
            predicted_coords = result.get('predicted_coords')
            atom_mask = result.get('atom_mask', None)
            
            if predicted_coords is None:
                logger.warning(f"样本 {sample_name} 采样 {sample_idx+1}: 未获取到预测坐标")
                continue
            
            # 保存采样结果
            sample_result = {
                'sample_idx': sample_idx,
                'predicted_coords': predicted_coords,
                'atom_mask': atom_mask
            }
            all_samples.append(sample_result)
        
        if not all_samples:
            logger.warning(f"样本 {sample_name}: 所有采样都失败")
            return {
                'sample_name': sample_name,
                'status': 'failed',
                'error': 'all_sampling_failed'
            }
        
        # 创建PDB输出目录
        pdb_output_dir = Path(output_dir) / "pdb_files"
        pdb_output_dir.mkdir(parents=True, exist_ok=True)
        
        pdb_file_paths = []
        
        # 保存最佳采样（默认第一个）
        best_sample = all_samples[0]
        best_pdb_path = pdb_output_dir / f"{sample_name}.pdb"
        try:
            diffold_coords_to_pdb(
                predicted_coords=best_sample['predicted_coords'],
                sequence=sequence,
                output_path=str(best_pdb_path),
                atom_mask=best_sample['atom_mask'],
                logger_instance=logger
            )
            pdb_file_paths.append(str(best_pdb_path))
            logger.debug(f"样本 {sample_name}: PDB文件已保存到 {best_pdb_path}")
        except Exception as e:
            logger.warning(f"样本 {sample_name}: PDB文件保存失败: {e}")
        
        # 如果需要保存所有采样结果
        if save_all_samples and num_sampling > 1:
            for i, sample_result in enumerate(all_samples):
                sample_pdb_path = pdb_output_dir / f"{sample_name}_sample_{i+1}.pdb"
                try:
                    diffold_coords_to_pdb(
                        predicted_coords=sample_result['predicted_coords'],
                        sequence=sequence,
                        output_path=str(sample_pdb_path),
                        atom_mask=sample_result['atom_mask'],
                        logger_instance=logger
                    )
                    pdb_file_paths.append(str(sample_pdb_path))
                except Exception as e:
                    logger.warning(f"样本 {sample_name} 采样 {i+1}: PDB文件保存失败: {e}")
        
        # 准备返回结果
        result_dict = {
            'sample_name': sample_name,
            'status': 'success',
            'sequence_length': len(sequence) if sequence else 0,
            'num_sampling': num_sampling,
            'successful_samples': len(all_samples),
            'pdb_file_paths': pdb_file_paths,
            'main_pdb_path': str(best_pdb_path) if pdb_file_paths else None
        }
        
        logger.debug(f"样本 {sample_name}: 完成 {len(all_samples)}/{num_sampling} 次成功采样")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"样本 {sample_name} 处理失败: {e}")
        return {
            'sample_name': sample_name,
            'status': 'failed',
            'error': str(e)
        }


def save_results(results: List[Dict[str, Any]], output_dir: str, logger: logging.Logger):
    """保存结果到文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON格式
    json_file = output_path / "inference_results.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"结果已保存到: {json_file}")
    
    # 保存为CSV格式
    csv_file = output_path / "inference_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    logger.info(f"结果已保存到: {csv_file}")
    
    # 生成统计报告
    report_file = output_path / "inference_report.txt"
    generate_report(results, report_file, logger)
    
    return json_file, csv_file, report_file


def generate_report(results: List[Dict[str, Any]], report_file: Path, logger: logging.Logger):
    """生成统计报告"""
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']
    
    with open(report_file, 'w') as f:
        f.write("Diffold批量推理报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总样本数: {len(results)}\n")
        f.write(f"成功样本数: {len(successful_results)}\n")
        f.write(f"失败样本数: {len(failed_results)}\n")
        f.write(f"成功率: {len(successful_results)/len(results)*100:.2f}%\n\n")
        
        if successful_results:
            # 采样统计信息
            num_sampling_values = [r.get('num_sampling', 1) for r in successful_results]
            successful_samples_values = [r.get('successful_samples', 1) for r in successful_results]
            
            if num_sampling_values:
                f.write("采样统计:\n")
                f.write("-" * 30 + "\n")
                f.write(f"每样本采样次数: {num_sampling_values[0]}\n")
                f.write(f"平均成功采样数: {np.mean(successful_samples_values):.2f}\n")
                f.write(f"采样成功率: {np.mean(successful_samples_values)/num_sampling_values[0]*100:.2f}%\n\n")
        
        if failed_results:
            f.write("失败样本:\n")
            f.write("-" * 30 + "\n")
            for result in failed_results:
                f.write(f"{result['sample_name']}: {result.get('error', 'unknown_error')}\n")
    
    logger.info(f"报告已保存到: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Diffold批量推理")
    
    # 基本参数
    parser.add_argument("--data_dir", default="./processed_data", 
                       help="数据目录路径")
    parser.add_argument("--output_dir", default="./inference_output", 
                       help="输出目录路径")
    
    # 数据选择：支持两种模式
    data_mode = parser.add_mutually_exclusive_group()
    data_mode.add_argument("--fold", type=int, default=None, 
                          help="使用交叉验证折数（例如: 0, 1, 2, 3, 4）")
    data_mode.add_argument("--sample_list_file", type=str, default=None,
                          help="指定样本列表文件路径（每行一个样本名称）")
    
    # 模型参数
    parser.add_argument("--checkpoint_path", required=True,
                       help="模型检查点路径")
    parser.add_argument("--rhofold_checkpoint", default="./pretrained/model_20221010_params.pt",
                       help="RhoFold检查点路径")
    parser.add_argument("--lora_path", default=None,
                       help="LoRA适配器路径（可选）")
    
    # 数据参数
    parser.add_argument("--max_sequence_length", type=int, default=256,
                       help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="数据加载器工作进程数")
    parser.add_argument("--use_msa", action="store_true", default=True,
                       help="是否使用MSA")
    
    # 设备参数
    parser.add_argument("--device", default="auto",
                       help="计算设备 (auto, cpu, cuda)")
    parser.add_argument("--log_level", default="INFO",
                       help="日志级别")
    
    # 可选参数
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大处理样本数（用于测试）")
    parser.add_argument("--num_sampling", type=int, default=1,
                       help="每个样本的采样次数")
    parser.add_argument("--save_all_samples", action="store_true", default=False,
                       help="是否保存所有采样结果")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.fold is None and args.sample_list_file is None:
        parser.error("必须指定 --fold 或 --sample_list_file 之一")
    
    # 设置设备
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        args.device = get_device(args.device)
    
    # 设置日志
    logger = setup_logging(args.output_dir, args.log_level)
    logger.info("=" * 60)
    logger.info("开始Diffold批量推理")
    logger.info("=" * 60)
    logger.info(f"设备: {args.device}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    
    if args.sample_list_file:
        logger.info(f"数据模式: 指定样本列表")
        logger.info(f"样本列表文件: {args.sample_list_file}")
    else:
        logger.info(f"数据模式: 交叉验证")
        logger.info(f"Fold: {args.fold}")
    
    logger.info(f"模型检查点: {args.checkpoint_path}")
    if args.lora_path:
        logger.info(f"LoRA适配器: {args.lora_path}")
    logger.info(f"每样本采样次数: {args.num_sampling}")
    logger.info(f"保存所有采样结果: {args.save_all_samples}")
    logger.info("=" * 60)
    
    try:
        # 加载模型
        model = load_model(args, logger)
        
        # 加载验证数据
        valid_loader, sample_names = load_validation_data(args, logger)
        
        # 批量处理
        results = []
        start_time = time.time()
        
        # 限制样本数量（用于测试）
        if args.max_samples:
            sample_names = sample_names[:args.max_samples]
            logger.info(f"限制处理样本数为: {len(sample_names)}")
        
        logger.info("开始批量推理...")
        for i, batch in enumerate(tqdm(valid_loader, 
                                     total=len(valid_loader),
                                     desc="处理样本")):
            
            # 从batch中获取实际的样本名称
            batch_names = batch.get('names', [])
            if not batch_names:
                logger.warning(f"Batch {i}: 未找到样本名称，跳过")
                continue
            
            sample_name = batch_names[0] if batch_names else f"unknown_{i}"
            
            logger.debug(f"处理样本 {i+1}/{len(valid_loader)}: {sample_name}")
            
            # 推理
            result = inference_sample(
                model=model,
                batch=batch,
                sample_name=sample_name,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                num_sampling=args.num_sampling,
                save_all_samples=args.save_all_samples,
                logger=logger
            )
            
            results.append(result)
            
            # 每100个样本输出一次进度
            if (i + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_sample = elapsed_time / (i + 1)
                remaining_samples = len(sample_names) - (i + 1)
                estimated_remaining_time = remaining_samples * avg_time_per_sample
                
                logger.info(f"已处理 {i+1}/{len(sample_names)} 样本, "
                          f"平均时间: {avg_time_per_sample:.2f}秒/样本, "
                          f"预计剩余时间: {estimated_remaining_time/60:.1f}分钟")
        
        # 保存结果
        logger.info("保存结果...")
        json_file, csv_file, report_file = save_results(results, args.output_dir, logger)
        
        # 输出总结
        total_time = time.time() - start_time
        successful_count = len([r for r in results if r['status'] == 'success'])
        failed_count = len([r for r in results if r['status'] == 'failed'])
        
        logger.info("批量推理完成!")
        logger.info(f"总处理时间: {total_time/60:.1f}分钟")
        logger.info(f"成功样本: {successful_count}")
        logger.info(f"失败样本: {failed_count}")
        logger.info(f"成功率: {successful_count/len(results)*100:.2f}%")
        
        logger.info(f"结果文件:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  报告: {report_file}")
        logger.info(f"  PDB文件目录: {Path(args.output_dir) / 'pdb_files'}")
        
        if args.save_all_samples:
            logger.info(f"  注意: 所有采样的PDB文件也已保存")
        
    except Exception as e:
        logger.error(f"批量推理失败: {e}")
        raise


if __name__ == "__main__":
    main()

