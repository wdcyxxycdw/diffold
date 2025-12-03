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
from typing import Dict, List, Any, Optional
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


class MolProbityClashCalculator:
    """
    使用 MolProbity probe 工具计算 Clash Score
    更准确，使用正确的范德华半径和overlap标准
    """
    
    def __init__(self, probe_path: str = "./tools/probe", overlap_cutoff: float = 0.4):
        """
        Args:
            probe_path: probe 工具路径
            overlap_cutoff: overlap 阈值 (Å)，通常 0.4 表示严重 clash
        """
        self.probe_path = probe_path
        self.overlap_cutoff = overlap_cutoff
        
        if not Path(probe_path).exists():
            raise FileNotFoundError(f"Probe tool not found: {probe_path}")
    
    def calculate(self, pred_pdb: str) -> Dict:
        """使用 MolProbity probe 计算 Clash score"""
        try:
            # 运行 probe 工具
            cmd = [
                self.probe_path,
                "-q",           # quiet mode
                "-u",           # unformatted output
                "-mc",          # main chain
                "-self",        # self contacts
                "all",          # all atoms
                pred_pdb
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # probe 可能返回多种 returncode，我们只检查是否有输出
            # returncode 0 = 正常, 141 = SIGPIPE (正常), 其他 = 可能也正常
            output = result.stdout
            
            # 如果没有输出，才认为失败
            if not output or len(output) < 10:
                return {
                    'error': f"Probe returned no output (returncode: {result.returncode})",
                    'clash_score': 0.0
                }
            
            # 解析 probe 输出
            # 格式: :1->1:wc: A   1   U  C2' : A   1   U  C2  :overlap:gap:...
            # parts[0]="", parts[1]="1->1", parts[2]="wc", parts[3]=atom1, parts[4]=atom2
            # parts[5]=overlap, parts[6]=gap
            total_contacts = 0
            bad_clashes = 0  # overlap > cutoff
            
            for line in output.split('\n'):
                if not line.strip() or not line.startswith(':'):
                    continue
                
                parts = line.split(':')
                if len(parts) < 7:  # 至少需要7个字段才能获取 overlap
                    continue
                
                try:
                    # overlap 值在第6个位置（0-indexed，parts[5]实际是第6个字段）
                    overlap = float(parts[5])
                    total_contacts += 1
                    
                    if overlap > self.overlap_cutoff:
                        bad_clashes += 1
                except (ValueError, IndexError):
                    continue
            
            # Clash score = 严重冲突数 / 每1000个原子
            # 这是 MolProbity 的标准定义
            # 但我们这里简化为：bad_clashes / total_contacts
            if total_contacts > 0:
                clash_score = bad_clashes / total_contacts
            else:
                clash_score = 0.0
            
            return {
                'clash_score': clash_score,
                'num_clashes': bad_clashes,
                'total_contacts': total_contacts
            }
            
        except subprocess.TimeoutExpired:
            return {'error': 'Probe timeout (>60s)', 'clash_score': 0.0}
        except Exception as e:
            return {'error': f'Probe calculation failed: {str(e)}', 'clash_score': 0.0}


class USalignWrapper:
    """US-align包装器 - 计算 RMSD, TM-score"""
    
    def __init__(self, usalign_path: str = "./tools/USalign"):
        self.usalign_path = usalign_path
        
        # 检查 US-align 是否存在
        if not Path(usalign_path).exists():
            raise FileNotFoundError(
                f"US-align 未找到: {usalign_path}\n"
                f"请确保工具已正确安装在 tools/ 目录"
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
                "-ter", "0",    # 不按 TER 记录分割链
                "-d", "5.0"     # 设置 d0=5.0 用于 TM-score 归一化
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
            
            # TM-score (优先使用 user-specified d0 的值，如果没有则使用第一个)
            # 查找 user-specified d0 的 TM-score
            tm_user_match = re.search(r'TM-score=\s*([\d.]+)\s*\(scaled by user-specified d0', output)
            if tm_user_match:
                metrics['tm_score'] = float(tm_user_match.group(1))
            else:
                # 如果没有 user-specified，则使用第一个（按第一个结构归一化的）
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


class TMscoreWrapper:
    """TMscore包装器 - 计算 GDT-TS, GDT-HA, MaxSub"""
    
    def __init__(self, tmscore_path: str = "./tools/TMscore"):
        self.tmscore_path = tmscore_path
        
        # 检查 TMscore 是否存在
        if not Path(tmscore_path).exists():
            raise FileNotFoundError(
                f"TMscore 未找到: {tmscore_path}\n"
                f"请确保工具已正确安装在 tools/ 目录"
            )
    
    def calculate_metrics(self, pred_pdb: str, native_pdb: str, d0: Optional[float] = None) -> Dict:
        """
        使用 TMscore 计算 GDT-TS, GDT-HA, MaxSub
        
        返回格式：
        {
            'gdt_ts': float,
            'gdt_ha': float,
            'maxsub': float,
            'gdt_p1': float,  # %(d<1)
            'gdt_p2': float,  # %(d<2)
            'gdt_p4': float,  # %(d<4)
            'gdt_p8': float,  # %(d<8)
            'raw_output': str
        }
        """
        try:
            # 运行 TMscore
            cmd = [
                self.tmscore_path,
                pred_pdb,
                native_pdb,
            ]
            # 仅当显式指定 d0 时传入 -d 参数；否则使用 TMscore 默认 d0
            if d0 is not None:
                cmd.extend(["-d", str(d0)])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {
                    'error': f"TMscore failed: {result.stderr}",
                    'returncode': result.returncode
                }
            
            output = result.stdout
            
            # 解析输出
            metrics = {}
            
            # GDT-TS-score= 1.0000 %(d<1)=1.0000 %(d<2)=1.0000 %(d<4)=1.0000 %(d<8)=1.0000
            gdt_ts_match = re.search(
                r'GDT-TS-score=\s*([\d.]+)\s+%\(d<1\)=([\d.]+)\s+%\(d<2\)=([\d.]+)\s+%\(d<4\)=([\d.]+)\s+%\(d<8\)=([\d.]+)',
                output
            )
            if gdt_ts_match:
                metrics['gdt_ts'] = float(gdt_ts_match.group(1))
                metrics['gdt_p1'] = float(gdt_ts_match.group(2))
                metrics['gdt_p2'] = float(gdt_ts_match.group(3))
                metrics['gdt_p4'] = float(gdt_ts_match.group(4))
                metrics['gdt_p8'] = float(gdt_ts_match.group(5))
            
            # GDT-HA-score= 0.9359 %(d<0.5)=0.7436 %(d<1)=1.0000 %(d<2)=1.0000 %(d<4)=1.0000
            gdt_ha_match = re.search(
                r'GDT-HA-score=\s*([\d.]+)',
                output
            )
            if gdt_ha_match:
                metrics['gdt_ha'] = float(gdt_ha_match.group(1))
            
            # MaxSub-score= 0.9836  (d0= 3.50)
            maxsub_match = re.search(r'MaxSub-score=\s*([\d.]+)', output)
            if maxsub_match:
                metrics['maxsub'] = float(maxsub_match.group(1))
            
            metrics['raw_output'] = output
            
            return metrics
            
        except subprocess.TimeoutExpired:
            return {'error': 'TMscore timeout (>30s)'}
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
                           tmscore_wrapper: TMscoreWrapper,
                           lddt_calculator: LDDTCalculator,
                           clash_calculator,  # MolProbityClashCalculator or None
                           logger: logging.Logger,
                           tmscore_d0: Optional[float] = None) -> Dict[str, Any]:
    """评估单个结构对"""
    try:
        # 使用US-align计算 RMSD 和 TM-score
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
        
        # 使用TMscore计算 GDT-TS, GDT-HA, MaxSub
        tmscore_metrics = tmscore_wrapper.calculate_metrics(
            str(pred_pdb),
            str(native_pdb),
            d0=tmscore_d0,
        )
        
        # 构建结果
        result = {
            'sample_name': sample_name,
            'status': 'success',
            'pred_pdb': str(pred_pdb),
            'native_pdb': str(native_pdb),
            'rmsd': metrics.get('rmsd'),
            'tm_score': metrics.get('tm_score'),
            'aligned_length': metrics.get('aligned_length'),
            'seq_identity': metrics.get('seq_identity')
        }
        
        # 添加TMscore的指标
        if 'error' not in tmscore_metrics:
            result['gdt_ts'] = tmscore_metrics.get('gdt_ts')
            result['gdt_ha'] = tmscore_metrics.get('gdt_ha')
            result['maxsub'] = tmscore_metrics.get('maxsub')
            result['gdt_p1'] = tmscore_metrics.get('gdt_p1')
            result['gdt_p2'] = tmscore_metrics.get('gdt_p2')
            result['gdt_p4'] = tmscore_metrics.get('gdt_p4')
            result['gdt_p8'] = tmscore_metrics.get('gdt_p8')
        else:
            logger.debug(f"样本 {sample_name}: TMscore计算失败: {tmscore_metrics['error']}")
        
        # 计算lDDT
        lddt_metrics = lddt_calculator.calculate(str(pred_pdb), str(native_pdb))
        if 'error' not in lddt_metrics:
            result['lddt'] = lddt_metrics.get('lddt')
            result['lddt_std'] = lddt_metrics.get('lddt_std')
            result['lddt_num_residues'] = lddt_metrics.get('num_residues')
        else:
            logger.debug(f"样本 {sample_name}: lDDT计算失败: {lddt_metrics['error']}")
        
        # 计算clash score (如果 clash_calculator 可用)
        if clash_calculator is not None:
            clash_metrics = clash_calculator.calculate(str(pred_pdb))
            if 'error' not in clash_metrics:
                result['clash_score'] = clash_metrics.get('clash_score')
                result['num_clashes'] = clash_metrics.get('num_clashes')
                # 使用 MolProbity 的 total_contacts
                result['total_contacts'] = clash_metrics.get('total_contacts', 0)
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
            
            # GDT-HA统计
            gdt_ha_values = [r['gdt_ha'] for r in successful_results if r.get('gdt_ha') is not None]
            if gdt_ha_values:
                f.write("GDT-HA统计 (High Accuracy):\n")
                f.write("-" * 40 + "\n")
                f.write(f"  平均值: {np.mean(gdt_ha_values):.4f}\n")
                f.write(f"  中位数: {np.median(gdt_ha_values):.4f}\n")
                f.write(f"  标准差: {np.std(gdt_ha_values):.4f}\n")
                f.write(f"  最小值: {np.min(gdt_ha_values):.4f}\n")
                f.write(f"  最大值: {np.max(gdt_ha_values):.4f}\n\n")
            
            # MaxSub统计
            maxsub_values = [r['maxsub'] for r in successful_results if r.get('maxsub') is not None]
            if maxsub_values:
                f.write("MaxSub统计:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  平均值: {np.mean(maxsub_values):.4f}\n")
                f.write(f"  中位数: {np.median(maxsub_values):.4f}\n")
                f.write(f"  标准差: {np.std(maxsub_values):.4f}\n")
                f.write(f"  最小值: {np.min(maxsub_values):.4f}\n")
                f.write(f"  最大值: {np.max(maxsub_values):.4f}\n\n")
            
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
使用示例（从 rhofold/ 目录运行）:

python scripts/evaluate_structures.py \\
    --pred_dir results/single_diffold_output/merged_pdb_files \\
    --native_dir benchmark_data/RNA-benchmark/single/pdb \\
    --output_dir results/single_diffold_output/evaluation_results \\
    --usalign_path tools/USalign \\
    --tmscore_path tools/TMscore

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
    parser.add_argument("--usalign_path", default="tools/USalign",
                       help="US-align可执行文件路径")
    parser.add_argument("--tmscore_path", default="tools/TMscore",
                       help="TMscore可执行文件路径")
    parser.add_argument("--log_level", default="INFO",
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help="日志级别")
    parser.add_argument("--tmscore_d0", type=float, default=None,
                       help="TMscore 中用于 GDT/MaxSub 计算的 d0 参数（默认: 不指定，使用 TMscore 内置默认）")
    
    args = parser.parse_args()
    
    # 验证目录存在
    pred_dir = Path(args.pred_dir)
    native_dir = Path(args.native_dir)
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    
    if not pred_dir.exists():
        print(f"❌ 预测目录不存在: {pred_dir}")
        return 1
    
    if not native_dir.exists():
        print(f"❌ 真实目录不存在: {native_dir}")
        return 1
    
    # 设置日志
    logger = setup_logging(str(output_dir), args.log_level)
    logger.info("=" * 60)
    logger.info("开始结构评估")
    logger.info("=" * 60)
    logger.info(f"预测目录: {pred_dir}")
    logger.info(f"真实目录: {native_dir}")
    logger.info(f"输出目录: {output_dir}")
    if args.tmscore_d0 is None:
        logger.info("TMscore d0: 默认（使用 TMscore 内置 d0）")
    else:
        logger.info(f"TMscore d0: {args.tmscore_d0}")
    logger.info("=" * 60)
    
    try:
        # 创建US-align包装器
        usalign_wrapper = USalignWrapper(usalign_path=args.usalign_path)
        logger.info(f"✅ US-align已就绪: {args.usalign_path}")
        
        # 创建TMscore包装器
        tmscore_wrapper = TMscoreWrapper(tmscore_path=args.tmscore_path)
        logger.info(f"✅ TMscore已就绪: {args.tmscore_path}")
        
        # 创建lDDT计算器
        lddt_calculator = LDDTCalculator()
        logger.info("✅ lDDT计算器已就绪")
        
        # 创建Clash计算器（使用 MolProbity probe）
        # 检查是否提供了 probe_path 参数
        if hasattr(args, 'probe_path') and args.probe_path:
            probe_path = Path(args.probe_path)
        else:
            # 尝试默认位置
            probe_path = None
            for possible_path in ["tools/probe", "../tools/probe", "./tools/probe"]:
                if Path(possible_path).exists():
                    probe_path = Path(possible_path)
                    break
        
        if probe_path and not probe_path.exists():
            logger.error(f"❌ MolProbity probe 未找到: {probe_path}")
            logger.error("请确保 probe 工具已安装或使用 --probe_path 参数指定路径")
            logger.warning("⚠️  继续评估但不计算 Clash Score")
            clash_calculator = None
        else:
            clash_calculator = MolProbityClashCalculator(probe_path=str(probe_path))
            logger.info(f"✅ 使用 MolProbity probe 计算 Clash Score: {probe_path}")
        
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
                tmscore_wrapper=tmscore_wrapper,
                lddt_calculator=lddt_calculator,
                clash_calculator=clash_calculator,
                logger=logger,
                tmscore_d0=args.tmscore_d0,
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

