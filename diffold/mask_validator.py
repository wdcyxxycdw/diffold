"""
Mask验证和监控机制
用于确保EDM batch处理中mask的一致性和正确性
"""

import torch
import warnings
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class MaskValidator:
    """Mask验证器，确保batch中不同长度序列的正确处理"""
    
    def __init__(self, 
                 enable_warnings: bool = True,
                 enable_logging: bool = True,
                 strict_mode: bool = False):
        """
        初始化Mask验证器
        
        Args:
            enable_warnings: 是否启用警告
            enable_logging: 是否启用日志记录
            strict_mode: 严格模式，发现问题时抛出异常
        """
        self.enable_warnings = enable_warnings
        self.enable_logging = enable_logging
        self.strict_mode = strict_mode
        
    def validate_batch_consistency(self,
                                 tokens: torch.Tensor,
                                 sequences: List[str],
                                 coordinates: torch.Tensor,
                                 missing_atom_mask: torch.Tensor,
                                 seq_lengths: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        验证batch数据的一致性
        
        Args:
            tokens: token张量 [batch_size, msa_depth, max_seq_len]
            sequences: 序列列表
            coordinates: 坐标张量 [batch_size, max_atoms, 3]
            missing_atom_mask: 缺失原子mask [batch_size, max_atoms]
            seq_lengths: 序列长度 [batch_size]
            
        Returns:
            验证结果字典
        """
        batch_size = tokens.shape[0]
        results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # 1. 检查batch维度一致性
        expected_shapes = {
            'tokens': (batch_size, tokens.shape[1], tokens.shape[2]),
            'coordinates': (batch_size, coordinates.shape[1], 3),
            'missing_atom_mask': (batch_size, missing_atom_mask.shape[1]),
            'sequences': batch_size
        }
        
        if len(sequences) != batch_size:
            error_msg = f"序列数量({len(sequences)})与batch_size({batch_size})不匹配"
            results['errors'].append(error_msg)
            results['is_valid'] = False
            
        # 2. 检查序列长度变化
        actual_seq_lengths = [len(seq) for seq in sequences]
        max_seq_len = max(actual_seq_lengths)
        min_seq_len = min(actual_seq_lengths)
        
        results['statistics']['seq_lengths'] = actual_seq_lengths
        results['statistics']['seq_length_variance'] = max_seq_len - min_seq_len
        results['statistics']['max_seq_len'] = max_seq_len
        results['statistics']['min_seq_len'] = min_seq_len
        
        if max_seq_len != min_seq_len:
            msg = f"检测到不同序列长度: 最长{max_seq_len}, 最短{min_seq_len}, 差异{max_seq_len - min_seq_len}"
            results['warnings'].append(msg)
            if self.enable_warnings:
                warnings.warn(msg)
        
        # 3. 检查padding正确性
        if seq_lengths is not None:
            for i, (expected_len, actual_len) in enumerate(zip(seq_lengths.tolist(), actual_seq_lengths)):
                if expected_len != actual_len:
                    error_msg = f"序列{i}长度不匹配: 期望{expected_len}, 实际{actual_len}"
                    results['errors'].append(error_msg)
                    results['is_valid'] = False
        
        # 4. 检查missing_atom_mask覆盖率
        valid_atoms_per_seq = []
        for i in range(batch_size):
            valid_atoms = (~missing_atom_mask[i]).sum().item()
            total_atoms = missing_atom_mask[i].shape[0]
            coverage = valid_atoms / total_atoms if total_atoms > 0 else 0
            valid_atoms_per_seq.append(coverage)
            
            if coverage < 0.1:  # 有效原子少于10%
                warning_msg = f"序列{i}有效原子比例过低: {coverage:.2%}"
                results['warnings'].append(warning_msg)
                if self.enable_warnings:
                    warnings.warn(warning_msg)
        
        results['statistics']['atom_coverage'] = valid_atoms_per_seq
        results['statistics']['avg_atom_coverage'] = sum(valid_atoms_per_seq) / len(valid_atoms_per_seq)
        
        # 5. 检查坐标有效性
        zero_coord_counts = []
        for i in range(batch_size):
            coords = coordinates[i][~missing_atom_mask[i]]  # 只检查有效原子
            if coords.shape[0] > 0:
                zero_coords = (coords.abs().sum(dim=1) < 1e-6).sum().item()
                zero_coord_counts.append(zero_coords)
                
                if zero_coords > coords.shape[0] * 0.5:  # 超过50%的坐标为零
                    warning_msg = f"序列{i}存在大量零坐标: {zero_coords}/{coords.shape[0]}"
                    results['warnings'].append(warning_msg)
            else:
                zero_coord_counts.append(0)
        
        results['statistics']['zero_coord_counts'] = zero_coord_counts
        
        # 严格模式下，有错误就抛出异常
        if self.strict_mode and not results['is_valid']:
            raise ValueError(f"Mask验证失败: {results['errors']}")
        
        # 记录日志
        if self.enable_logging:
            logger.info(f"Batch mask验证完成: valid={results['is_valid']}, "
                       f"warnings={len(results['warnings'])}, errors={len(results['errors'])}")
        
        return results
    
    def validate_edm_inputs(self,
                           atom_mask: torch.Tensor,
                           missing_atom_mask: torch.Tensor,
                           molecule_atom_lens: torch.Tensor,
                           mask: torch.Tensor) -> Dict[str, Any]:
        """
        验证传递给EDM的mask参数一致性
        
        Args:
            atom_mask: 原子存在mask [batch_size, max_atoms]
            missing_atom_mask: 缺失原子mask [batch_size, max_atoms]
            molecule_atom_lens: 分子原子长度 [batch_size, num_molecules]
            mask: 序列mask [batch_size, max_seq_len]
            
        Returns:
            验证结果字典
        """
        results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        batch_size = atom_mask.shape[0]
        
        # 1. 检查mask维度一致性
        if atom_mask.shape != missing_atom_mask.shape:
            error_msg = f"atom_mask({atom_mask.shape})与missing_atom_mask({missing_atom_mask.shape})形状不匹配"
            results['errors'].append(error_msg)
            results['is_valid'] = False
        
        # 2. 检查mask逻辑一致性
        # atom_mask为True表示原子存在，missing_atom_mask为True表示原子缺失
        # 理论上 atom_mask 应该与 ~missing_atom_mask 基本一致
        consistency_scores = []
        for i in range(batch_size):
            expected_atom_mask = ~missing_atom_mask[i]
            actual_atom_mask = atom_mask[i]
            
            # 计算一致性
            consistent = (expected_atom_mask == actual_atom_mask).float().mean().item()
            consistency_scores.append(consistent)
            
            if consistent < 0.95:  # 一致性低于95%
                warning_msg = f"序列{i}的atom_mask与missing_atom_mask一致性较低: {consistent:.2%}"
                results['warnings'].append(warning_msg)
        
        results['statistics']['mask_consistency_scores'] = consistency_scores
        results['statistics']['avg_mask_consistency'] = sum(consistency_scores) / len(consistency_scores)
        
        # 3. 检查分子原子长度的合理性
        molecule_lens = molecule_atom_lens.sum(dim=-1)  # 每个序列的总原子数
        atom_counts = atom_mask.sum(dim=-1)  # 实际存在的原子数
        
        len_ratios = []
        for i in range(batch_size):
            if molecule_lens[i] > 0:
                ratio = atom_counts[i].float() / molecule_lens[i].float()
                len_ratios.append(ratio.item())
                
                if ratio < 0.5 or ratio > 1.5:  # 偏差超过50%
                    warning_msg = f"序列{i}分子长度与实际原子数偏差较大: {ratio:.2f}"
                    results['warnings'].append(warning_msg)
            else:
                len_ratios.append(0.0)
        
        results['statistics']['length_ratios'] = len_ratios
        
        # 记录统计信息
        results['statistics']['total_atoms'] = atom_mask.sum().item()
        results['statistics']['total_missing'] = missing_atom_mask.sum().item()
        results['statistics']['valid_sequences'] = (mask.sum(dim=-1) > 0).sum().item()
        
        if self.enable_logging:
            logger.info(f"EDM输入mask验证: valid={results['is_valid']}, "
                       f"avg_consistency={results['statistics']['avg_mask_consistency']:.3f}")
        
        return results
    
    def monitor_loss_computation(self,
                               logits: torch.Tensor,
                               labels: torch.Tensor,
                               mask: torch.Tensor,
                               loss_type: str = "cross_entropy") -> Dict[str, Any]:
        """
        监控损失计算中的mask使用情况
        
        Args:
            logits: 预测logits
            labels: 真实标签
            mask: 应用的mask
            loss_type: 损失类型
            
        Returns:
            监控结果
        """
        results = {
            'statistics': {},
            'warnings': []
        }
        
        # 计算mask覆盖率
        if mask.dim() == logits.dim() - 1:  # mask维度比logits少1（通常是最后的类别维度）
            mask_flat = mask.flatten()
            valid_ratio = mask_flat.float().mean().item()
        else:
            valid_ratio = 1.0  # 无法确定，假设全部有效
        
        results['statistics']['mask_coverage'] = valid_ratio
        results['statistics']['total_elements'] = labels.numel()
        results['statistics']['valid_elements'] = mask.sum().item() if mask.dtype == torch.bool else mask.sum().item()
        
        # 检查标签范围
        if labels.dtype in [torch.long, torch.int]:
            unique_labels = torch.unique(labels[mask] if mask.dtype == torch.bool else labels)
            num_classes = logits.shape[-1] if logits.dim() > 1 else 1
            
            invalid_labels = (unique_labels < 0) | (unique_labels >= num_classes)
            if invalid_labels.any():
                warning_msg = f"{loss_type}损失中检测到无效标签: {unique_labels[invalid_labels].tolist()}"
                results['warnings'].append(warning_msg)
        
        # 检查logits是否包含NaN或Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            warning_msg = f"{loss_type}损失的logits包含NaN或Inf"
            results['warnings'].append(warning_msg)
        
        if self.enable_logging and len(results['warnings']) > 0:
            logger.warning(f"损失计算监控发现问题: {results['warnings']}")
        
        return results
    
    def create_batch_report(self, 
                          validation_results: List[Dict[str, Any]], 
                          step: int = 0) -> str:
        """
        创建batch处理报告
        
        Args:
            validation_results: 验证结果列表
            step: 训练步数
            
        Returns:
            格式化的报告字符串
        """
        report = [f"\n📊 Batch处理报告 (Step {step})"]
        report.append("=" * 50)
        
        total_warnings = sum(len(result.get('warnings', [])) for result in validation_results)
        total_errors = sum(len(result.get('errors', [])) for result in validation_results)
        
        report.append(f"总体状态: {'✅ 正常' if total_errors == 0 else '❌ 有错误'}")
        report.append(f"警告数量: {total_warnings}")
        report.append(f"错误数量: {total_errors}")
        
        # 统计信息汇总
        if validation_results:
            for i, result in enumerate(validation_results):
                if 'statistics' in result:
                    stats = result['statistics']
                    report.append(f"\n验证阶段 {i+1}:")
                    
                    if 'seq_length_variance' in stats:
                        report.append(f"  序列长度变化: {stats['seq_length_variance']}")
                    
                    if 'avg_atom_coverage' in stats:
                        report.append(f"  平均原子覆盖率: {stats['avg_atom_coverage']:.2%}")
                    
                    if 'avg_mask_consistency' in stats:
                        report.append(f"  Mask一致性: {stats['avg_mask_consistency']:.2%}")
        
        # 警告和错误详情
        if total_warnings > 0:
            report.append(f"\n⚠️ 警告详情:")
            for i, result in enumerate(validation_results):
                for warning in result.get('warnings', []):
                    report.append(f"  - {warning}")
        
        if total_errors > 0:
            report.append(f"\n❌ 错误详情:")
            for i, result in enumerate(validation_results):
                for error in result.get('errors', []):
                    report.append(f"  - {error}")
        
        return "\n".join(report)

# 全局验证器实例
default_validator = MaskValidator(
    enable_warnings=True,
    enable_logging=True,
    strict_mode=False
)

def validate_batch_for_edm(tokens: torch.Tensor,
                          sequences: List[str],
                          coordinates: torch.Tensor,
                          missing_atom_mask: torch.Tensor,
                          seq_lengths: Optional[torch.Tensor] = None,
                          validator: Optional[MaskValidator] = None) -> bool:
    """
    为EDM调用验证batch数据的便捷函数
    
    Returns:
        是否验证通过
    """
    if validator is None:
        validator = default_validator
    
    result = validator.validate_batch_consistency(
        tokens, sequences, coordinates, missing_atom_mask, seq_lengths
    )
    
    return result['is_valid'] 