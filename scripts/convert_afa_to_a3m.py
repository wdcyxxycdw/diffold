#!/usr/bin/env python3
"""
将AFA格式的MSA文件批量转换为A3M格式

使用HHsuite的reformat.pl工具进行转换
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Optional
import sys


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_reformat_script() -> Optional[Path]:
    """
    查找reformat.pl脚本
    
    返回:
        Path: reformat.pl脚本的路径，如果未找到则返回None
    """
    # 在rhofold/data/bin目录中查找
    possible_paths = [
        Path(__file__).parent.parent / "rhofold" / "data" / "bin" / "reformat.pl",
        Path("rhofold/data/bin/reformat.pl"),
        Path("/usr/local/bin/reformat.pl"),
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"找到reformat.pl脚本: {path}")
            return path
    
    return None


def convert_afa_to_a3m(
    input_file: Path,
    output_file: Path,
    reformat_script: Path
) -> bool:
    """
    转换单个AFA文件为A3M格式
    
    参数:
        input_file: 输入AFA文件
        output_file: 输出A3M文件
        reformat_script: reformat.pl脚本路径
    
    返回:
        bool: 是否成功转换
    """
    try:
        # 创建输出目录
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 调用reformat.pl进行转换
        # 格式：reformat.pl fas a3m input.afa output.a3m -M first
        # -M first: 使所有包含第一个序列残基的列成为匹配列
        cmd = [
            "perl",
            str(reformat_script),
            "fas",  # 输入格式（afa实际上就是fas）
            "a3m",  # 输出格式
            str(input_file),
            str(output_file),
            "-M", "first"  # 根据第一个序列（query）确定匹配列
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.debug(f"  ✓ {input_file.name} -> {output_file.name}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"  ✗ 转换失败: {input_file.name}")
        logger.error(f"    错误信息: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"  ✗ 转换失败: {input_file.name}")
        logger.error(f"    错误: {e}")
        return False


def batch_convert(
    input_dir: Path,
    output_dir: Path,
    reformat_script: Path
) -> dict:
    """
    批量转换目录中的所有AFA文件
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        reformat_script: reformat.pl脚本路径
    
    返回:
        dict: 统计信息
    """
    # 收集所有AFA文件
    afa_files = list(input_dir.glob("*.afa"))
    
    if not afa_files:
        logger.error(f"未找到AFA文件: {input_dir}")
        return {'total': 0, 'success': 0, 'failed': 0}
    
    logger.info("=" * 80)
    logger.info("批量转换AFA到A3M格式")
    logger.info("=" * 80)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"文件数量: {len(afa_files)}")
    logger.info(f"reformat脚本: {reformat_script}")
    logger.info("=" * 80)
    
    # 转换每个文件
    success_count = 0
    failed_count = 0
    
    for i, afa_file in enumerate(afa_files, 1):
        logger.info(f"[{i}/{len(afa_files)}] 转换: {afa_file.name}")
        
        # 生成输出文件名（保持相同的basename，只改扩展名）
        output_file = output_dir / afa_file.name.replace('.afa', '.a3m')
        
        if convert_afa_to_a3m(afa_file, output_file, reformat_script):
            success_count += 1
        else:
            failed_count += 1
    
    return {
        'total': len(afa_files),
        'success': success_count,
        'failed': failed_count
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='批量转换AFA格式的MSA文件为A3M格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 批量转换目录中的所有AFA文件
  python convert_afa_to_a3m.py benchmark_data/RNA-benchmark/single/rMSA \\
         --output benchmark_data/RNA-benchmark/single/rMSA_a3m
  
  # 指定reformat.pl脚本路径
  python convert_afa_to_a3m.py input_dir --output output_dir \\
         --reformat-script /path/to/reformat.pl

格式说明:
  AFA (Aligned FASTA): 标准对齐FASTA格式，所有gaps用'-'表示
  A3M: HHsuite格式，大写=匹配，小写=插入，省略与插入对齐的gaps
"""
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='包含AFA文件的输入目录'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录'
    )
    
    parser.add_argument(
        '--reformat-script',
        type=str,
        help='reformat.pl脚本的路径（如果不指定，会自动搜索）'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式（只显示警告和错误）'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='调试模式（显示详细信息）'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    
    # 查找reformat.pl脚本
    if args.reformat_script:
        reformat_script = Path(args.reformat_script)
        if not reformat_script.exists():
            logger.error(f"指定的reformat.pl脚本不存在: {reformat_script}")
            return 1
    else:
        reformat_script = find_reformat_script()
        if not reformat_script:
            logger.error("未找到reformat.pl脚本！")
            logger.error("请使用 --reformat-script 参数指定脚本路径")
            return 1
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return 1
    
    # 批量转换
    stats = batch_convert(input_dir, output_dir, reformat_script)
    
    # 打印总结
    logger.info("\n" + "=" * 80)
    logger.info("转换完成")
    logger.info("=" * 80)
    logger.info(f"总文件数: {stats['total']}")
    logger.info(f"成功转换: {stats['success']}")
    logger.info(f"转换失败: {stats['failed']}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 80)
    
    return 0 if stats['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

