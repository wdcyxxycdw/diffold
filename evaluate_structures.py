#!/usr/bin/env python3
"""
结构评估脚本
比较预测结构和真实结构，计算RMSD、TM-score、GDT-TS等指标
使用US-align进行权威指标计算
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
import subprocess
import re

# BioPython导入（用于lDDT和clash score计算）
from Bio.PDB import PDBParser, NeighborSearch


class LDDTCalculator:
    """
    lDDT (Local Distance Difference Test) 计算器
    参考: Mariani et al. (2013) Bioinformatics
    https://academic.oup.com/bioinformatics/article/29/21/2722/195896
    """
    
    def __init__(self, 
                 inclusion_radius: float = 15.0,
                 thresholds: List[float] = None):
        """
        Args:
            inclusion_radius: 考虑的邻居距离范围 (Å)
            thresholds: 距离差异阈值，默认 [0.5, 1.0, 2.0, 4.0] Å
        """
        self.inclusion_radius = inclusion_radius
        self.thresholds = thresholds or [0.5, 1.0, 2.0, 4.0]
        self.parser = PDBParser(QUIET=True)
    
    def calculate(self, pred_pdb: str, native_pdb: str) -> Dict:
        """计算 lDDT"""
        try:
            pred_struct = self.parser.get_structure("pred", pred_pdb)
            native_struct = self.parser.get_structure("native", native_pdb)
            
            # 提取 C4' 原子（RNA骨架）
            pred_atoms = [atom for atom in pred_struct.get_atoms() 
                         if atom.get_name() == "C4'"]
            native_atoms = [atom for atom in native_struct.get_atoms() 
                           if atom.get_name() == "C4'"]
            
            if len(pred_atoms) != len(native_atoms):
                min_len = min(len(pred_atoms), len(native_atoms))
                pred_atoms = pred_atoms[:min_len]
                native_atoms = native_atoms[:min_len]
            
            if len(native_atoms) < 2:
                return {'error': 'Too few atoms', 'lddt': 0.0}
            
            # 构建邻居搜索树
            native_ns = NeighborSearch(native_atoms)
            
            lddt_scores = []
            
            for i, (pred_atom, native_atom) in enumerate(zip(pred_atoms, native_atoms)):
                # 在 native 结构中找到 inclusion_radius 范围内的邻居
                neighbors = native_ns.search(
                    native_atom.coord, 
                    self.inclusion_radius, 
                    level='A'
                )
                
                if len(neighbors) < 2:  # 至少需要1个邻居（除了自己）
                    continue
                
                preserved = 0
                total = 0
                
                for neighbor_atom in neighbors:
                    if neighbor_atom == native_atom:
                        continue
                    
                    # 找到对应的预测原子
                    try:
                        neighbor_idx = native_atoms.index(neighbor_atom)
                    except ValueError:
                        continue
                    
                    if neighbor_idx >= len(pred_atoms):
                        continue
                    
                    pred_neighbor = pred_atoms[neighbor_idx]
                    
                    # 计算距离
                    native_dist = native_atom - neighbor_atom
                    pred_dist = pred_atom - pred_neighbor
                    
                    diff = abs(pred_dist - native_dist)
                    
                    # 对每个阈值，检查是否保持
                    for threshold in self.thresholds:
                        if diff < threshold:
                            preserved += 1
                            break
                    
                    total += 1
                
                if total > 0:
                    # lDDT = 保持的距离对数 / 总的距离对数
                    lddt_scores.append(preserved / total)
            
            if not lddt_scores:
                return {'error': 'No valid scores', 'lddt': 0.0}
            
            return {
                'lddt': np.mean(lddt_scores),
                'lddt_std': np.std(lddt_scores),
                'num_residues': len(lddt_scores)
            }
            
        except Exception as e:
            return {'error': f'lDDT calculation failed: {str(e)}', 'lddt': 0.0}


class ClashScoreCalculator:
    """
    原子冲突分数计算器
    检测非成键原子间的空间冲突
    """
    
    def __init__(self, clash_threshold: float = 2.0):
        """
        Args:
            clash_threshold: 非成键原子的最小允许距离 (Å)
        """
        self.clash_threshold = clash_threshold
        self.parser = PDBParser(QUIET=True)
    
    def _are_bonded(self, atom1, atom2) -> bool:
        """判断两个原子是否可能成键（简化判断）"""
        res1 = atom1.get_parent().get_id()[1]
        res2 = atom2.get_parent().get_id()[1]
        # 同一残基或相邻残基认为可能成键
        return abs(res1 - res2) <= 1
    
    def calculate(self, pred_pdb: str) -> Dict:
        """计算 Clash score"""
        try:
            structure = self.parser.get_structure("pred", pred_pdb)
            atoms = list(structure.get_atoms())
            
            if len(atoms) < 2:
                return {'error': 'Too few atoms', 'clash_score': 0.0}
            
            # 构建邻居搜索
            ns = NeighborSearch(atoms)
            
            clashes = 0
            total_pairs = 0
            clash_details = []
            
            checked_pairs = set()
            
            for atom in atoms:
                # 找到距离小于 clash_threshold 的原子
                neighbors = ns.search(atom.coord, self.clash_threshold, level='A')
                
                for neighbor in neighbors:
                    if neighbor == atom:
                        continue
                    
                    # 避免重复检查
                    pair = tuple(sorted([id(atom), id(neighbor)]))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)
                    
                    # 检查是否是成键原子
                    if self._are_bonded(atom, neighbor):
                        continue
                    
                    total_pairs += 1
                    distance = atom - neighbor
                    
                    if distance < self.clash_threshold:
                        clashes += 1
                        if len(clash_details) < 10:  # 只保存前10个
                            clash_details.append({
                                'atom1': f"{atom.get_parent().get_id()[1]}:{atom.get_name()}",
                                'atom2': f"{neighbor.get_parent().get_id()[1]}:{neighbor.get_name()}",
                                'distance': float(distance)
                            })
            
            clash_score = clashes / max(total_pairs, 1) if total_pairs > 0 else 0.0
            
            return {
                'clash_score': clash_score,
                'num_clashes': clashes,
                'total_pairs': total_pairs,
                'clash_details': clash_details[:10]  # 只保存前10个
            }
            
        except Exception as e:
            return {'error': f'Clash calculation failed: {str(e)}', 'clash_score': 0.0}


class USalignWrapper:
    """US-align包装器 - 使用权威工具计算 RMSD, TM-score, GDT-TS"""
    
    def __init__(self, usalign_path: str = "./USalign/USalign"):
        self.usalign_path = usalign_path
        
        # 检查 US-align 是否存在
        if not Path(usalign_path).exists():
            raise FileNotFoundError(
                f"US-align 未找到: {usalign_path}\n"
                f"请先编译 US-align:\n"
                f"  cd USalign\n"
                f"  g++ -static -O3 -ffast-math -o USalign USalign.cpp"
            )
    
    def calculate_metrics(self, pred_pdb: str, native_pdb: str) -> Dict:
        """
        使用 US-align 计算 RMSD, TM-score, GDT-TS
        
        返回格式：
        {
            'rmsd': float,
            'tm_score': float,
            'gdt_ts': float,
            'aligned_length': int,
            'seq_identity': float,
            'raw_output': str
        }
        """
        try:
            # 运行 US-align
            cmd = [
                self.usalign_path,
                pred_pdb,
                native_pdb,
                "-mol", "RNA",  # RNA 模式
                "-ter", "0"     # 不按 TER 记录分割链
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {
                    'error': f"US-align failed: {result.stderr}",
                    'returncode': result.returncode
                }
            
            output = result.stdout
            
            # 解析输出
            metrics = {}
            
            # RMSD
            rmsd_match = re.search(r'RMSD=\s*([\d.]+)', output)
            if rmsd_match:
                metrics['rmsd'] = float(rmsd_match.group(1))
            
            # TM-score (使用第一个，通常是按第一个结构归一化的)
            tm_matches = re.findall(r'TM-score=\s*([\d.]+)', output)
            if tm_matches:
                metrics['tm_score'] = float(tm_matches[0])
            
            # GDT-TS
            gdt_match = re.search(r'GDT-TS-score=\s*([\d.]+)', output)
            if gdt_match:
                metrics['gdt_ts'] = float(gdt_match.group(1))
            
            # Aligned length
            len_match = re.search(r'Aligned length=\s*(\d+)', output)
            if len_match:
                metrics['aligned_length'] = int(len_match.group(1))
            
            # Sequence identity
            seqid_match = re.search(r'Seq_ID.*?=\s*([\d.]+)', output)
            if seqid_match:
                metrics['seq_identity'] = float(seqid_match.group(1))
            
            metrics['raw_output'] = output
            
            return metrics
            
        except subprocess.TimeoutExpired:
            return {'error': 'US-align timeout (>30s)'}
        except Exception as e:
            return {'error': f'Exception: {str(e)}'}


def setup_logging(output_dir: str, log_level: str = "INFO"):
    """设置日志"""
    log_file = Path(output_dir) / "evaluation.log"
    
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


def find_pdb_pairs(pred_dir: Path, native_dir: Path, logger: logging.Logger) -> List[tuple]:
    """
    查找预测和真实PDB文件对
    
    返回: [(sample_name, pred_pdb_path, native_pdb_path), ...]
    """
    pairs = []
    
    # 遍历预测目录中的所有PDB文件
    pred_pdbs = list(pred_dir.glob("*.pdb"))
    
    logger.info(f"在预测目录中找到 {len(pred_pdbs)} 个PDB文件")
    
    for pred_pdb in pred_pdbs:
        # 获取样本名称（去除扩展名和可能的后缀）
        sample_name = pred_pdb.stem
        
        # 处理可能的后缀（如 _best, _relaxed_1000 等）
        # 尝试多种可能的匹配模式
        possible_names = [
            sample_name,  # 原始名称
            sample_name.replace('_best', ''),  # 去除 _best
            sample_name.replace('_unrelaxed', ''),  # 去除 _unrelaxed
        ]
        
        # 处理 relaxed 后缀 (如 _relaxed_1000)
        if '_relaxed_' in sample_name:
            base_name = sample_name.split('_relaxed_')[0]
            possible_names.append(base_name)
        
        # 在真实目录中查找匹配的PDB
        native_pdb = None
        matched_name = None
        
        for name in possible_names:
            candidate = native_dir / f"{name}.pdb"
            if candidate.exists():
                native_pdb = candidate
                matched_name = name
                break
        
        if native_pdb:
            pairs.append((matched_name, pred_pdb, native_pdb))
        else:
            logger.warning(f"未找到真实PDB文件: {sample_name} (尝试了: {possible_names})")
    
    logger.info(f"找到 {len(pairs)} 对匹配的PDB文件")
    
    return pairs


def evaluate_structure_pair(sample_name: str, 
                           pred_pdb: Path, 
                           native_pdb: Path,
                           usalign_wrapper: USalignWrapper,
                           lddt_calculator: LDDTCalculator,
                           clash_calculator: ClashScoreCalculator,
                           logger: logging.Logger) -> Dict[str, Any]:
    """评估单个结构对"""
    try:
        # 使用US-align计算指标
        metrics = usalign_wrapper.calculate_metrics(str(pred_pdb), str(native_pdb))
        
        if 'error' in metrics:
            logger.warning(f"样本 {sample_name}: US-align失败: {metrics['error']}")
            return {
                'sample_name': sample_name,
                'status': 'failed',
                'error': metrics['error'],
                'pred_pdb': str(pred_pdb),
                'native_pdb': str(native_pdb)
            }
        
        # 构建结果
        result = {
            'sample_name': sample_name,
            'status': 'success',
            'pred_pdb': str(pred_pdb),
            'native_pdb': str(native_pdb),
            'rmsd': metrics.get('rmsd'),
            'tm_score': metrics.get('tm_score'),
            'gdt_ts': metrics.get('gdt_ts'),
            'aligned_length': metrics.get('aligned_length'),
            'seq_identity': metrics.get('seq_identity')
        }
        
        # 计算lDDT
        lddt_metrics = lddt_calculator.calculate(str(pred_pdb), str(native_pdb))
        if 'error' not in lddt_metrics:
            result['lddt'] = lddt_metrics.get('lddt')
            result['lddt_std'] = lddt_metrics.get('lddt_std')
            result['lddt_num_residues'] = lddt_metrics.get('num_residues')
        else:
            logger.debug(f"样本 {sample_name}: lDDT计算失败: {lddt_metrics['error']}")
        
        # 计算clash score
        clash_metrics = clash_calculator.calculate(str(pred_pdb))
        if 'error' not in clash_metrics:
            result['clash_score'] = clash_metrics.get('clash_score')
            result['num_clashes'] = clash_metrics.get('num_clashes')
            result['total_pairs'] = clash_metrics.get('total_pairs')
        else:
            logger.debug(f"样本 {sample_name}: Clash计算失败: {clash_metrics['error']}")
        
        # 构建日志信息
        log_parts = [f"RMSD={result['rmsd']:.4f}", f"TM={result['tm_score']:.4f}"]
        if 'lddt' in result:
            log_parts.append(f"lDDT={result['lddt']:.4f}")
        if 'clash_score' in result:
            log_parts.append(f"Clash={result['clash_score']:.4f}")
        
        logger.debug(f"样本 {sample_name}: {', '.join(log_parts)}")
        
        return result
        
    except Exception as e:
        logger.error(f"样本 {sample_name} 评估失败: {e}")
        return {
            'sample_name': sample_name,
            'status': 'failed',
            'error': str(e),
            'pred_pdb': str(pred_pdb),
            'native_pdb': str(native_pdb)
        }


def save_results(results: List[Dict[str, Any]], output_dir: Path, logger: logging.Logger):
    """保存评估结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON格式
    json_file = output_dir / "evaluation_results.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"结果已保存到: {json_file}")
    
    # 保存为CSV格式
    csv_file = output_dir / "evaluation_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    logger.info(f"结果已保存到: {csv_file}")
    
    # 生成统计报告
    report_file = output_dir / "evaluation_report.txt"
    generate_report(results, report_file, logger)
    
    return json_file, csv_file, report_file


def generate_report(results: List[Dict[str, Any]], report_file: Path, logger: logging.Logger):
    """生成统计报告"""
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']
    
    with open(report_file, 'w') as f:
        f.write("结构评估报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"总样本数: {len(results)}\n")
        f.write(f"成功评估: {len(successful_results)}\n")
        f.write(f"失败评估: {len(failed_results)}\n")
        f.write(f"成功率: {len(successful_results)/len(results)*100:.2f}%\n\n")
        
        if successful_results:
            # RMSD统计
            rmsd_values = [r['rmsd'] for r in successful_results if r.get('rmsd') is not None]
            if rmsd_values:
                f.write("RMSD统计:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  平均值: {np.mean(rmsd_values):.4f} Å\n")
                f.write(f"  中位数: {np.median(rmsd_values):.4f} Å\n")
                f.write(f"  标准差: {np.std(rmsd_values):.4f} Å\n")
                f.write(f"  最小值: {np.min(rmsd_values):.4f} Å\n")
                f.write(f"  最大值: {np.max(rmsd_values):.4f} Å\n\n")
            
            # TM-score统计
            tm_values = [r['tm_score'] for r in successful_results if r.get('tm_score') is not None]
            if tm_values:
                f.write("TM-score统计:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  平均值: {np.mean(tm_values):.4f}\n")
                f.write(f"  中位数: {np.median(tm_values):.4f}\n")
                f.write(f"  标准差: {np.std(tm_values):.4f}\n")
                f.write(f"  最小值: {np.min(tm_values):.4f}\n")
                f.write(f"  最大值: {np.max(tm_values):.4f}\n\n")
            
            # GDT-TS统计
            gdt_values = [r['gdt_ts'] for r in successful_results if r.get('gdt_ts') is not None]
            if gdt_values:
                f.write("GDT-TS统计:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  平均值: {np.mean(gdt_values):.4f}\n")
                f.write(f"  中位数: {np.median(gdt_values):.4f}\n")
                f.write(f"  标准差: {np.std(gdt_values):.4f}\n")
                f.write(f"  最小值: {np.min(gdt_values):.4f}\n")
                f.write(f"  最大值: {np.max(gdt_values):.4f}\n\n")
            
            # lDDT统计
            lddt_values = [r['lddt'] for r in successful_results if r.get('lddt') is not None]
            if lddt_values:
                f.write("lDDT统计:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  平均值: {np.mean(lddt_values):.4f}\n")
                f.write(f"  中位数: {np.median(lddt_values):.4f}\n")
                f.write(f"  标准差: {np.std(lddt_values):.4f}\n")
                f.write(f"  最小值: {np.min(lddt_values):.4f}\n")
                f.write(f"  最大值: {np.max(lddt_values):.4f}\n")
                f.write(f"  高质量样本(≥0.7): {np.mean(np.array(lddt_values) >= 0.7)*100:.1f}%\n\n")
            
            # Clash Score统计
            clash_values = [r['clash_score'] for r in successful_results if r.get('clash_score') is not None]
            if clash_values:
                f.write("Clash Score统计:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  平均值: {np.mean(clash_values):.4f}\n")
                f.write(f"  中位数: {np.median(clash_values):.4f}\n")
                f.write(f"  标准差: {np.std(clash_values):.4f}\n")
                f.write(f"  最小值: {np.min(clash_values):.4f}\n")
                f.write(f"  最大值: {np.max(clash_values):.4f}\n")
                f.write(f"  低冲突样本(≤0.05): {np.mean(np.array(clash_values) <= 0.05)*100:.1f}%\n\n")
        
        if failed_results:
            f.write("失败样本:\n")
            f.write("-" * 40 + "\n")
            for result in failed_results:
                f.write(f"  {result['sample_name']}: {result.get('error', 'unknown_error')}\n")
    
    logger.info(f"报告已保存到: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="结构评估脚本 - 比较预测PDB和真实PDB，计算多种评估指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 基本用法:
   python evaluate_structures.py \\
       --pred_dir ./output/pdb_files \\
       --native_dir ./data/pdb \\
       --output_dir ./evaluation_results

2. 指定US-align路径:
   python evaluate_structures.py \\
       --pred_dir ./predictions \\
       --native_dir ./ground_truth \\
       --output_dir ./results \\
       --usalign_path ./USalign/USalign

评估指标:
  - RMSD: 均方根偏差（使用US-align）
  - TM-score: 模板建模分数（使用US-align）
  - GDT-TS: 全局距离测试分数（使用US-align）
  - lDDT: 局部距离差异测试（使用BioPython）
  - Clash Score: 原子冲突分数（使用BioPython）

依赖:
  - US-align: 用于RMSD/TM-score/GDT-TS计算
  - BioPython: 用于lDDT和Clash Score计算
    安装: pip install biopython
        """
    )
    
    # 必需参数
    parser.add_argument("--pred_dir", required=True,
                       help="预测PDB文件目录")
    parser.add_argument("--native_dir", required=True,
                       help="真实PDB文件目录")
    parser.add_argument("--output_dir", required=True,
                       help="评估结果输出目录")
    
    # 可选参数
    parser.add_argument("--usalign_path", default="./USalign/USalign",
                       help="US-align可执行文件路径")
    parser.add_argument("--log_level", default="INFO",
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 验证目录存在
    pred_dir = Path(args.pred_dir)
    native_dir = Path(args.native_dir)
    output_dir = Path(args.output_dir)
    
    if not pred_dir.exists():
        print(f"❌ 预测目录不存在: {pred_dir}")
        return 1
    
    if not native_dir.exists():
        print(f"❌ 真实目录不存在: {native_dir}")
        return 1
    
    # 设置日志
    logger = setup_logging(args.output_dir, args.log_level)
    logger.info("=" * 60)
    logger.info("开始结构评估")
    logger.info("=" * 60)
    logger.info(f"预测目录: {pred_dir}")
    logger.info(f"真实目录: {native_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)
    
    try:
        # 创建US-align包装器
        usalign_wrapper = USalignWrapper(usalign_path=args.usalign_path)
        logger.info(f"✅ US-align已就绪: {args.usalign_path}")
        
        # 创建lDDT和Clash计算器
        lddt_calculator = LDDTCalculator()
        clash_calculator = ClashScoreCalculator()
        logger.info("✅ lDDT和Clash Score计算器已就绪")
        
        # 查找PDB文件对
        logger.info("\n查找PDB文件对...")
        pdb_pairs = find_pdb_pairs(pred_dir, native_dir, logger)
        
        if not pdb_pairs:
            logger.error("❌ 未找到任何匹配的PDB文件对")
            return 1
        
        # 评估所有结构对
        logger.info(f"\n开始评估 {len(pdb_pairs)} 个结构...")
        results = []
        start_time = time.time()
        
        for sample_name, pred_pdb, native_pdb in tqdm(pdb_pairs, desc="评估进度"):
            result = evaluate_structure_pair(
                sample_name=sample_name,
                pred_pdb=pred_pdb,
                native_pdb=native_pdb,
                usalign_wrapper=usalign_wrapper,
                lddt_calculator=lddt_calculator,
                clash_calculator=clash_calculator,
                logger=logger
            )
            results.append(result)
        
        # 保存结果
        logger.info("\n保存评估结果...")
        json_file, csv_file, report_file = save_results(results, output_dir, logger)
        
        # 输出总结
        total_time = time.time() - start_time
        successful_count = len([r for r in results if r['status'] == 'success'])
        failed_count = len([r for r in results if r['status'] == 'failed'])
        
        logger.info("\n" + "=" * 60)
        logger.info("评估完成!")
        logger.info("=" * 60)
        logger.info(f"总耗时: {total_time:.1f}秒")
        logger.info(f"成功评估: {successful_count}/{len(results)}")
        logger.info(f"失败评估: {failed_count}/{len(results)}")
        logger.info(f"成功率: {successful_count/len(results)*100:.2f}%")
        
        # 输出关键统计
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            rmsd_values = [r['rmsd'] for r in successful_results if r.get('rmsd') is not None]
            tm_values = [r['tm_score'] for r in successful_results if r.get('tm_score') is not None]
            lddt_values = [r['lddt'] for r in successful_results if r.get('lddt') is not None]
            clash_values = [r['clash_score'] for r in successful_results if r.get('clash_score') is not None]
            
            logger.info("\n关键指标统计:")
            if rmsd_values:
                logger.info(f"  RMSD: 平均={np.mean(rmsd_values):.4f} Å, "
                          f"中位数={np.median(rmsd_values):.4f} Å")
            if tm_values:
                logger.info(f"  TM-score: 平均={np.mean(tm_values):.4f}, "
                          f"中位数={np.median(tm_values):.4f}")
            if lddt_values:
                logger.info(f"  lDDT: 平均={np.mean(lddt_values):.4f}, "
                          f"中位数={np.median(lddt_values):.4f}")
            if clash_values:
                logger.info(f"  Clash Score: 平均={np.mean(clash_values):.4f}, "
                          f"中位数={np.median(clash_values):.4f}")
        
        logger.info(f"\n结果文件:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  报告: {report_file}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"评估失败: {e}")
        raise


if __name__ == "__main__":
    exit(main())

