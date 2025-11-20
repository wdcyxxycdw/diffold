#!/usr/bin/env python3
"""
RNA 结构 Amber 能量最小化脚本
使用 rhofold 的 AmberRelaxation 类进行结构优化
支持单个文件或批量处理目录
"""
import sys
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 注意：不在这里导入 AmberRelaxation，在子进程中延迟导入以避免 CUDA 初始化问题

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def relax_single_file_worker(task_info):
    """
    Worker function for multiprocessing
    
    Args:
        task_info: tuple of (in_pdb, out_pdb, max_iterations, use_gpu, worker_id)
    
    Returns:
        tuple: (file_name, success)
    """
    in_pdb, out_pdb, max_iterations, use_gpu, worker_id = task_info
    
    # 为每个进程创建独立的logger
    worker_logger = logging.getLogger(f'worker_{worker_id}')
    
    try:
        # 如果使用GPU，设置环境变量（但现在主要使用CPU）
        if use_gpu and worker_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_id)
            worker_logger.info(f"[Worker {worker_id}] 处理文件: {os.path.basename(in_pdb)} (尝试GPU)")
        else:
            worker_logger.info(f"[Worker {worker_id}] 处理文件: {os.path.basename(in_pdb)} (CPU)")
        
        worker_logger.info(f"  输出: {os.path.basename(out_pdb)}")
        worker_logger.info(f"  最大迭代: {max_iterations}")
        
        # 延迟导入 AmberRelaxation
        from rhofold.relax.relax import AmberRelaxation
        
        # 创建 AmberRelaxation 实例
        amber_relax = AmberRelaxation(
            max_iterations=max_iterations,
            use_gpu=use_gpu,
            logger=worker_logger
        )
        
        # 执行能量最小化
        amber_relax.process(in_pdb, out_pdb)
        
        worker_logger.info(f"[Worker {worker_id}] ✓ 完成: {os.path.basename(out_pdb)}")
        return (os.path.basename(in_pdb), True)
        
    except Exception as e:
        worker_logger.error(f"[Worker {worker_id}] ✗ 失败: {os.path.basename(in_pdb)} - {str(e)}")
        return (os.path.basename(in_pdb), False)

def relax_single_file(in_pdb, out_pdb, max_iterations=4000, use_gpu=False, gpu_id=None):
    """
    对单个 PDB 文件进行能量最小化（单进程版本）
    
    Args:
        in_pdb: 输入 PDB 文件路径
        out_pdb: 输出 PDB 文件路径
        max_iterations: 最大迭代次数
        use_gpu: 是否使用 GPU
        gpu_id: GPU ID (用于多GPU并行)
    
    Returns:
        bool: 是否成功
    """
    try:
        # 延迟导入 AmberRelaxation
        from rhofold.relax.relax import AmberRelaxation
        
        logger.info(f"处理文件: {in_pdb}")
        logger.info(f"  输出: {out_pdb}")
        logger.info(f"  最大迭代: {max_iterations}")
        
        # 创建 AmberRelaxation 实例
        amber_relax = AmberRelaxation(
            max_iterations=max_iterations,
            use_gpu=use_gpu,
            logger=logger
        )
        
        # 执行能量最小化
        amber_relax.process(in_pdb, out_pdb)
        
        logger.info(f"✓ 完成: {os.path.basename(out_pdb)}")
        return True
        
    except Exception as e:
        logger.error(f"✗ 失败: {in_pdb} - {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='RNA 结构 Amber 能量最小化 - 支持单文件或批量处理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件
  python minimize_rna_openmm.py -i input.pdb -o output.pdb
  
  # 批量处理目录（单进程CPU）
  python minimize_rna_openmm.py -d /path/to/pdb_files -o /path/to/output
  
  # 批量处理目录（多进程CPU并行，推荐）
  python minimize_rna_openmm.py -d ./pdb_files -o ./relaxed --num-workers 10
  
  # 指定最大迭代次数
  python minimize_rna_openmm.py -d ./pdb_files -o ./relaxed --num-workers 10 --max-iter 5000
        """
    )
    
    # 输入选项 (单文件或目录)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-i', '--input',
        type=str,
        help='输入单个 PDB 文件'
    )
    input_group.add_argument(
        '-d', '--input-dir',
        type=str,
        help='输入 PDB 文件目录（批量处理）'
    )
    
    # 输出选项
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='输出文件或目录'
    )
    
    # 其他选项
    parser.add_argument(
        '--max-iter',
        type=int,
        default=4000,
        help='最大迭代次数 (默认: 4000)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='使用 GPU 加速'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='并行处理的进程数 (默认: 1，用于多进程并行处理)'
    )
    
    parser.add_argument(
        '--suffix',
        type=str,
        default='_relaxed',
        help='输出文件后缀 (默认: _relaxed)'
    )
    
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = datetime.now()
    logger.info("="*70)
    logger.info("RNA 结构 Amber 能量最小化")
    logger.info("="*70)
    
    # 单文件模式
    if args.input:
        if not os.path.exists(args.input):
            logger.error(f"输入文件不存在: {args.input}")
            sys.exit(1)
        
        success = relax_single_file(
            args.input,
            args.output,
            args.max_iter,
            args.gpu
        )
        
        if success:
            logger.info("✓ 处理完成！")
        else:
            logger.error("✗ 处理失败")
            sys.exit(1)
    
    # 批量目录模式
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output)
        
        if not input_dir.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            sys.exit(1)
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有 PDB 文件
        pdb_files = sorted(input_dir.glob("*.pdb"))
        
        if not pdb_files:
            logger.error(f"在目录中未找到 PDB 文件: {input_dir}")
            sys.exit(1)
        
        logger.info(f"找到 {len(pdb_files)} 个 PDB 文件")
        logger.info(f"输出目录: {output_dir}")
        
        # 多进程并行处理
        if args.num_workers > 1:
            if args.gpu:
                logger.info(f"使用 {args.num_workers} 个进程并行处理 (尝试GPU)")
            else:
                logger.info(f"使用 {args.num_workers} 个进程并行处理 (CPU)")
            logger.info("")
            
            # 准备任务列表
            tasks = []
            for i, pdb_file in enumerate(pdb_files):
                base_name = pdb_file.stem
                out_name = f"{base_name}{args.suffix}.pdb"
                out_path = output_dir / out_name
                
                # 分配worker ID (循环分配，CPU模式下只是标识符)
                worker_id = i % args.num_workers
                tasks.append((str(pdb_file), str(out_path), args.max_iter, args.gpu, worker_id))
            
            # 使用进程池并行处理
            with Pool(processes=args.num_workers) as pool:
                results = pool.map(relax_single_file_worker, tasks)
            
            # 统计结果
            success_count = sum(1 for _, success in results if success)
            failed_count = len(results) - success_count
            
        else:
            # 单进程串行处理
            if args.gpu:
                logger.info("使用单进程串行处理 (尝试GPU)")
            else:
                logger.info("使用单进程串行处理 (CPU)")
            logger.info("")
            
            success_count = 0
            failed_count = 0
            
            for i, pdb_file in enumerate(pdb_files, 1):
                logger.info(f"[{i}/{len(pdb_files)}] 处理中...")
                
                # 生成输出文件名
                base_name = pdb_file.stem
                out_name = f"{base_name}{args.suffix}.pdb"
                out_path = output_dir / out_name
                
                # 处理文件
                if relax_single_file(str(pdb_file), str(out_path), args.max_iter, args.gpu):
                    success_count += 1
                else:
                    failed_count += 1
                
                logger.info("")
        
        # 统计结果
        elapsed = datetime.now() - start_time
        logger.info("="*70)
        logger.info("批量处理完成！")
        logger.info(f"  成功: {success_count}/{len(pdb_files)}")
        logger.info(f"  失败: {failed_count}/{len(pdb_files)}")
        logger.info(f"  耗时: {elapsed}")
        logger.info("="*70)
        
        if failed_count > 0:
            sys.exit(1)
    
    elapsed = datetime.now() - start_time
    logger.info(f"总耗时: {elapsed}")

if __name__ == '__main__':
    main()
