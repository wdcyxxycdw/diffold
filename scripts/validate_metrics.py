#!/usr/bin/env python3
"""
完整的RNA结构指标验证工具
- 使用 US-align 计算: RMSD, TM-score, GDT-TS (权威实现)
- 使用 BioPython 计算: lDDT, Clash score (标准算法)
- 与自己的实现进行对比验证
"""

import argparse
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
import warnings

warnings.filterwarnings('ignore')


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
                'num_residues': len(lddt_scores),
                'source': 'BioPython (标准算法)'
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
                'clash_details': clash_details,
                'source': 'BioPython'
            }
            
        except Exception as e:
            return {'error': f'Clash calculation failed: {str(e)}', 'clash_score': 0.0}


class USalignWrapper:
    """US-align 包装器 - 计算 RMSD, TM-score (d0=5Å), GDT-TS"""
    
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
        使用 US-align 计算 RMSD, TM-score (d0=5Å), GDT-TS
        
        注意：TM-score 使用固定的 d0=5Å，使不同长度的结构具有可比性
        
        返回格式：
        {
            'rmsd': float,
            'tm_score': float,  # 使用 d0=5Å
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
                "-d", "5"       # 固定 d0=5Å 用于 TM-score 计算
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
            
            # TM-score (使用 d0=5 的那个，即第3个 TM-score 值)
            tm_matches = re.findall(r'TM-score=\s*([\d.]+)', output)
            if len(tm_matches) >= 3:
                # 第3个是 scaled by user-specified d0=5.00 的值
                metrics['tm_score'] = float(tm_matches[2])
            elif tm_matches:
                # 如果只有2个或1个，使用第一个作为后备
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


def load_my_results(results_file: str) -> List[Dict]:
    """加载我的实现的结果"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # 处理不同的文件格式
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # 可能是 detailed_metrics 格式
        return list(data.values())
    else:
        raise ValueError(f"不支持的结果文件格式: {type(data)}")


def compare_single_sample(sample_name: str,
                         my_metrics: Dict,
                         usalign_metrics: Dict,
                         lddt_metrics: Dict,
                         clash_metrics: Dict) -> Dict:
    """对比单个样本的所有指标"""
    
    comparison = {
        'sample': sample_name,
        # 我的实现
        'my_rmsd': my_metrics.get('avg_rmsd', my_metrics.get('rmsd')),
        'my_tm_score': my_metrics.get('avg_tm_score', my_metrics.get('tm_score')),
        'my_lddt': my_metrics.get('avg_lddt', my_metrics.get('lddt')),
        'my_clash_score': my_metrics.get('avg_clash_score', my_metrics.get('clash_score')),
    }
    
    # US-align 结果 (RMSD, TM-score, GDT-TS)
    if 'error' in usalign_metrics:
        comparison['usalign_error'] = usalign_metrics['error']
        comparison['usalign_rmsd'] = None
        comparison['usalign_tm_score'] = None
        comparison['usalign_gdt_ts'] = None
    else:
        comparison['usalign_rmsd'] = usalign_metrics.get('rmsd')
        comparison['usalign_tm_score'] = usalign_metrics.get('tm_score')
        comparison['usalign_gdt_ts'] = usalign_metrics.get('gdt_ts')
        comparison['usalign_aligned_length'] = usalign_metrics.get('aligned_length')
        
        # 计算 RMSD 差异
        if comparison['my_rmsd'] and comparison['usalign_rmsd']:
            comparison['rmsd_diff'] = abs(comparison['my_rmsd'] - comparison['usalign_rmsd'])
            comparison['rmsd_rel_error'] = comparison['rmsd_diff'] / comparison['usalign_rmsd']
        
        # 计算 TM-score 差异
        if comparison['my_tm_score'] and comparison['usalign_tm_score']:
            comparison['tm_diff'] = abs(comparison['my_tm_score'] - comparison['usalign_tm_score'])
            comparison['tm_rel_error'] = comparison['tm_diff'] / comparison['usalign_tm_score']
    
    # lDDT 结果 (BioPython标准算法)
    if 'error' not in lddt_metrics:
        comparison['ref_lddt'] = lddt_metrics.get('lddt')
        comparison['ref_lddt_num_residues'] = lddt_metrics.get('num_residues')
        
        # 计算 lDDT 差异
        if comparison['my_lddt'] and comparison['ref_lddt']:
            comparison['lddt_diff'] = abs(comparison['my_lddt'] - comparison['ref_lddt'])
            comparison['lddt_rel_error'] = comparison['lddt_diff'] / comparison['ref_lddt']
    else:
        comparison['ref_lddt_error'] = lddt_metrics['error']
    
    # Clash score 结果 (BioPython)
    if 'error' not in clash_metrics:
        comparison['ref_clash_score'] = clash_metrics.get('clash_score')
        comparison['ref_num_clashes'] = clash_metrics.get('num_clashes')
        comparison['ref_total_pairs'] = clash_metrics.get('total_pairs')
        
        # 计算 Clash score 差异
        if comparison['my_clash_score'] and comparison['ref_clash_score']:
            comparison['clash_diff'] = abs(comparison['my_clash_score'] - comparison['ref_clash_score'])
            if comparison['ref_clash_score'] > 0:
                comparison['clash_rel_error'] = comparison['clash_diff'] / comparison['ref_clash_score']
    else:
        comparison['ref_clash_error'] = clash_metrics['error']
    
    return comparison


def validate_batch(my_results_file: str,
                  pred_dir: str,
                  native_dir: str,
                  usalign_path: str,
                  output_csv: str = "validation_report.csv",
                  output_json: str = "validation_report.json",
                  pred_suffix: str = "",
                  native_suffix: str = "") -> pd.DataFrame:
    """批量验证所有指标"""
    
    print("=" * 70)
    print("完整的RNA结构指标验证工具")
    print("=" * 70)
    print(f"我的结果: {my_results_file}")
    print(f"预测目录: {pred_dir}")
    print(f"真实目录: {native_dir}")
    print(f"US-align: {usalign_path}")
    print()
    print("将计算以下指标:")
    print("  - RMSD, TM-score, GDT-TS (US-align 权威实现)")
    print("  - lDDT (BioPython 标准算法)")
    print("  - Clash score (BioPython)")
    print("=" * 70)
    print()
    
    # 初始化所有计算器
    usalign = USalignWrapper(usalign_path)
    lddt_calc = LDDTCalculator(inclusion_radius=15.0)
    clash_calc = ClashScoreCalculator(clash_threshold=2.0)
    
    my_results = load_my_results(my_results_file)
    
    # 创建样本名称到指标的映射
    my_metrics_dict = {}
    for result in my_results:
        sample_name = result.get('sample_name', result.get('sample', result.get('name')))
        if sample_name:
            my_metrics_dict[sample_name] = result
    
    print(f"加载了 {len(my_metrics_dict)} 个样本的指标\n")
    
    # 批量验证
    comparisons = []
    
    for i, (sample_name, my_metrics) in enumerate(my_metrics_dict.items(), 1):
        print(f"[{i}/{len(my_metrics_dict)}] {sample_name}")
        
        # 构建PDB文件路径（支持自定义后缀）
        pred_pdb = Path(pred_dir) / f"{sample_name}{pred_suffix}.pdb"
        native_pdb = Path(native_dir) / f"{sample_name}{native_suffix}.pdb"
        
        # 检查文件是否存在
        if not pred_pdb.exists():
            print(f"  ✗ 预测PDB不存在: {pred_pdb}")
            continue
        
        if not native_pdb.exists():
            print(f"  ✗ 真实PDB不存在: {native_pdb}")
            continue
        
        # 1. 运行 US-align (RMSD, TM-score, GDT-TS)
        print(f"  🔄 US-align...", end=" ")
        usalign_metrics = usalign.calculate_metrics(str(pred_pdb), str(native_pdb))
        if 'error' not in usalign_metrics:
            print(f"✓ RMSD={usalign_metrics.get('rmsd', 0):.3f}, TM={usalign_metrics.get('tm_score', 0):.3f}, GDT-TS={usalign_metrics.get('gdt_ts', 0):.3f}")
        else:
            print(f"✗ {usalign_metrics['error']}")
        
        # 2. 计算 lDDT
        print(f"  🔄 lDDT...", end=" ")
        lddt_metrics = lddt_calc.calculate(str(pred_pdb), str(native_pdb))
        if 'error' not in lddt_metrics:
            print(f"✓ lDDT={lddt_metrics.get('lddt', 0):.3f}")
        else:
            print(f"✗ {lddt_metrics['error']}")
        
        # 3. 计算 Clash score
        print(f"  🔄 Clash...", end=" ")
        clash_metrics = clash_calc.calculate(str(pred_pdb))
        if 'error' not in clash_metrics:
            print(f"✓ Clash={clash_metrics.get('clash_score', 0):.3f} ({clash_metrics.get('num_clashes', 0)} clashes)")
        else:
            print(f"✗ {clash_metrics['error']}")
        
        # 对比所有指标
        comparison = compare_single_sample(
            sample_name, 
            my_metrics, 
            usalign_metrics,
            lddt_metrics,
            clash_metrics
        )
        comparisons.append(comparison)
        
        # 显示对比结果
        errors = []
        if 'error' not in usalign_metrics:
            rmsd_diff = comparison.get('rmsd_diff', 0)
            tm_diff = comparison.get('tm_diff', 0)
            if rmsd_diff > 0.1 or tm_diff > 0.01:
                errors.append(f"RMSD差={rmsd_diff:.4f}, TM差={tm_diff:.4f}")
        
        if 'lddt_diff' in comparison and comparison['lddt_diff'] > 0.05:
            errors.append(f"lDDT差={comparison['lddt_diff']:.4f}")
        
        if errors:
            print(f"  ⚠️  差异: {'; '.join(errors)}")
        else:
            print(f"  ✅ 所有指标一致")
        
        print()
    
    # 转换为DataFrame
    df = pd.DataFrame(comparisons)
    
    # 保存结果
    df.to_csv(output_csv, index=False)
    print(f"\n✅ CSV报告已保存: {output_csv}")
    
    with open(output_json, 'w') as f:
        json.dump(comparisons, f, indent=2, default=str)
    print(f"✅ JSON报告已保存: {output_json}")
    
    # 统计
    print("\n" + "=" * 70)
    print("验证统计")
    print("=" * 70)
    
    # 检查是否有成功的验证结果
    if 'usalign_rmsd' not in df.columns or df.empty:
        print("⚠️  没有成功验证的样本")
        return df
    
    successful = df[df['usalign_rmsd'].notna()]
    
    if len(successful) > 0:
        print(f"成功验证: {len(successful)}/{len(df)} 个样本\n")
        
        # RMSD统计 (US-align 作为参考)
        if 'rmsd_diff' in successful.columns and successful['rmsd_diff'].notna().any():
            print("📊 RMSD 差异 (vs US-align):")
            print(f"  平均差异: {successful['rmsd_diff'].mean():.4f} Å")
            print(f"  最大差异: {successful['rmsd_diff'].max():.4f} Å")
            print(f"  中位数差异: {successful['rmsd_diff'].median():.4f} Å")
            print(f"  平均相对误差: {successful['rmsd_rel_error'].mean()*100:.2f}%")
            rmsd_ok = successful['rmsd_diff'].mean() < 0.1
            print(f"  结论: {'✅ 正确' if rmsd_ok else '⚠️ 需要检查'}")
        
        # TM-score统计 (US-align 作为参考)
        if 'tm_diff' in successful.columns and successful['tm_diff'].notna().any():
            print(f"\n📊 TM-score 差异 (vs US-align):")
            print(f"  平均差异: {successful['tm_diff'].mean():.4f}")
            print(f"  最大差异: {successful['tm_diff'].max():.4f}")
            print(f"  中位数差异: {successful['tm_diff'].median():.4f}")
            print(f"  平均相对误差: {successful['tm_rel_error'].mean()*100:.2f}%")
            tm_ok = successful['tm_diff'].mean() < 0.01
            print(f"  结论: {'✅ 正确' if tm_ok else '⚠️ 需要检查'}")
        
        # GDT-TS统计 (仅显示US-align的值，因为我们可能没有实现)
        if 'usalign_gdt_ts' in successful.columns and successful['usalign_gdt_ts'].notna().any():
            print(f"\n📊 GDT-TS (US-align 参考值):")
            print(f"  平均: {successful['usalign_gdt_ts'].mean():.4f}")
            print(f"  中位数: {successful['usalign_gdt_ts'].median():.4f}")
            print(f"  范围: [{successful['usalign_gdt_ts'].min():.4f}, {successful['usalign_gdt_ts'].max():.4f}]")
        
        # lDDT统计 (BioPython 标准算法作为参考)
        if 'lddt_diff' in successful.columns and successful['lddt_diff'].notna().any():
            print(f"\n📊 lDDT 差异 (vs BioPython标准算法):")
            print(f"  平均差异: {successful['lddt_diff'].mean():.4f}")
            print(f"  最大差异: {successful['lddt_diff'].max():.4f}")
            print(f"  中位数差异: {successful['lddt_diff'].median():.4f}")
            if successful['lddt_rel_error'].notna().any():
                print(f"  平均相对误差: {successful['lddt_rel_error'].mean()*100:.2f}%")
            lddt_ok = successful['lddt_diff'].mean() < 0.05
            print(f"  结论: {'✅ 正确' if lddt_ok else '⚠️ 需要检查'}")
        
        # Clash score统计 (BioPython作为参考)
        if 'clash_diff' in successful.columns and successful['clash_diff'].notna().any():
            print(f"\n📊 Clash score 差异 (vs BioPython):")
            print(f"  平均差异: {successful['clash_diff'].mean():.4f}")
            print(f"  最大差异: {successful['clash_diff'].max():.4f}")
            print(f"  中位数差异: {successful['clash_diff'].median():.4f}")
            clash_ok = successful['clash_diff'].mean() < 0.05
            print(f"  结论: {'✅ 正确' if clash_ok else '⚠️ 需要检查'}")
        
        # 总体结论
        print("\n" + "=" * 70)
        rmsd_ok = successful['rmsd_diff'].mean() < 0.1 if 'rmsd_diff' in successful.columns else True
        tm_ok = successful['tm_diff'].mean() < 0.01 if 'tm_diff' in successful.columns else True
        lddt_ok = successful['lddt_diff'].mean() < 0.05 if 'lddt_diff' in successful.columns else True
        
        if rmsd_ok and tm_ok and lddt_ok:
            print("✅ 总体结论: 所有指标实现正确！")
        else:
            print("⚠️  总体结论: 部分指标存在差异，建议检查实现")
            
            # 显示差异最大的样本
            if 'rmsd_diff' in successful.columns:
                print("\n差异最大的样本（前5个）:")
                cols = ['sample', 'my_rmsd', 'usalign_rmsd', 'rmsd_diff', 
                       'my_tm_score', 'usalign_tm_score', 'tm_diff']
                available_cols = [c for c in cols if c in successful.columns]
                top_diff = successful.nlargest(5, 'rmsd_diff')[available_cols]
                print(top_diff.to_string(index=False))
        
        print("=" * 70)
    else:
        print("⚠️  没有成功验证的样本")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="完整的RNA结构指标验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
功能：
  - RMSD, TM-score, GDT-TS: 使用 US-align (权威实现)
  - lDDT: 使用 BioPython (标准算法)
  - Clash score: 使用 BioPython
  - 与你的实现进行全面对比验证

使用示例:

1. 验证 Diffold 结果:
   python scripts/validate_metrics_usalign.py \\
       --my_results merged_results.json \\
       --pred_dir results/pdb_files \\
       --native_dir benchmark_data/casp16/pdb \\
       --usalign_path ./USalign/USalign

2. 验证 RhoFold 结果:
   python scripts/validate_metrics_usalign.py \\
       --my_results rhofold_merged_results.json \\
       --pred_dir rhofold_results/pdb_files \\
       --native_dir benchmark_data/casp16/pdb \\
       --output_csv rhofold_validation.csv

注意:
  - 需要先编译 US-align
  - 需要安装 biopython: pip install biopython
        """
    )
    
    parser.add_argument("--my_results", required=True,
                       help="我的结果JSON文件")
    parser.add_argument("--pred_dir", required=True,
                       help="预测PDB目录")
    parser.add_argument("--native_dir", required=True,
                       help="真实PDB目录")
    parser.add_argument("--usalign_path", default="./USalign/USalign",
                       help="US-align可执行文件路径 (默认: ./USalign/USalign)")
    parser.add_argument("--output_csv", default="validation_report.csv",
                       help="输出CSV文件 (默认: validation_report.csv)")
    parser.add_argument("--output_json", default="validation_report.json",
                       help="输出JSON文件 (默认: validation_report.json)")
    parser.add_argument("--pred_suffix", default="",
                       help="预测PDB文件名后缀 (例如: _best)")
    parser.add_argument("--native_suffix", default="",
                       help="真实PDB文件名后缀")
    
    args = parser.parse_args()
    
    try:
        validate_batch(
            args.my_results,
            args.pred_dir,
            args.native_dir,
            args.usalign_path,
            args.output_csv,
            args.output_json,
            args.pred_suffix,
            args.native_suffix
        )
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

