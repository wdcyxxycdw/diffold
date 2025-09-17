"""
RhoFold模型输出处理模块
提供PDB文件生成和输出验证功能
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

def rhofold_coords_to_pdb(predicted_coords: torch.Tensor, 
                         sequence: str,
                         output_path: str, 
                         confidence: Optional[torch.Tensor] = None,
                         model_instance: Optional[object] = None,
                         logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    将RhoFold预测的坐标转换为PDB文件
    优先使用RhoFold内置的export_pdb_file方法
    
    Args:
        predicted_coords: 预测的原子坐标 [seq_len, atom_types, 3] 或 [total_atoms, 3]
        sequence: RNA序列
        output_path: 输出PDB文件路径
        confidence: 置信度分数 (可选)
        model_instance: RhoFold模型实例 (可选，用于调用内置方法)
        logger_instance: 日志器实例 (可选)
        
    Returns:
        bool: 是否成功保存
    """
    try:
        # 如果有logger，使用它；否则创建一个简单的logger
        logger = logger_instance if logger_instance else logging.getLogger(__name__)
        
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 优先使用RhoFold内置的export_pdb_file方法
        if model_instance is not None and hasattr(model_instance, 'structure_module'):
            try:
                # 转换坐标为numpy数组
                if isinstance(predicted_coords, torch.Tensor):
                    coords_np = predicted_coords.detach().cpu().numpy()
                else:
                    coords_np = predicted_coords
                
                # 处理置信度
                confidence_np = None
                if confidence is not None:
                    if isinstance(confidence, torch.Tensor):
                        confidence_np = confidence.detach().cpu().numpy()
                    else:
                        confidence_np = confidence
                
                # 如果坐标是3D的，取最后一个维度或者压缩
                if len(coords_np.shape) == 3:
                    if coords_np.shape[0] == 1:  # batch维度
                        coords_np = coords_np.squeeze(0)
                    else:
                        coords_np = coords_np.reshape(-1, 3)  # 展平所有原子
                
                # 调用RhoFold内置的export_pdb_file方法
                # 先尝试不同的参数组合
                try:
                    # 尝试原始inference_rf.py中的参数格式
                    model_instance.structure_module.converter.export_pdb_file(
                        sequence,
                        coords_np,
                        path=output_path,
                        chain_id=None,
                        confidence=confidence_np,
                        logger=logger
                    )
                except TypeError as te:
                    # 如果参数不匹配，尝试更简单的调用
                    try:
                        model_instance.structure_module.converter.export_pdb_file(
                            sequence,
                            coords_np,
                            output_path,
                            None,
                            confidence_np,
                            logger
                        )
                    except Exception as te2:
                        raise Exception(f"PDB导出参数错误: {te}, {te2}")
                
                logger.debug(f"使用RhoFold内置方法成功保存PDB文件: {output_path}")
                return True
                
            except Exception as e:
                logger.warning(f"RhoFold内置方法保存失败: {e}，尝试使用自定义方法")
                # 如果内置方法失败，继续使用自定义方法
        
        # 如果没有模型实例或内置方法失败，使用自定义的PDB写入方法
        return _write_pdb_custom(predicted_coords, sequence, output_path, confidence, logger)
        
    except Exception as e:
        if logger_instance:
            logger_instance.error(f"保存RhoFold PDB文件失败: {e}")
        else:
            print(f"错误: 保存RhoFold PDB文件失败: {e}")
        return False

def _write_pdb_custom(predicted_coords: torch.Tensor, 
                     sequence: str,
                     output_path: str, 
                     confidence: Optional[torch.Tensor] = None,
                     logger: Optional[logging.Logger] = None) -> bool:
    """
    自定义的PDB写入方法（作为备选方案）
    """
    try:
        # 转换坐标为numpy数组
        if isinstance(predicted_coords, torch.Tensor):
            coords_np = predicted_coords.detach().cpu().numpy()
        elif isinstance(predicted_coords, (list, tuple)):
            # 如果是列表或元组，尝试取第一个元素或转换
            if len(predicted_coords) > 0:
                if isinstance(predicted_coords[0], torch.Tensor):
                    coords_np = predicted_coords[0].detach().cpu().numpy()
                else:
                    coords_np = np.array(predicted_coords)
            else:
                raise ValueError("坐标列表为空")
        else:
            coords_np = predicted_coords
        
        # 处理置信度
        confidence_np = None
        if confidence is not None:
            if isinstance(confidence, torch.Tensor):
                confidence_np = confidence.detach().cpu().numpy()
            else:
                confidence_np = confidence
        
        # RNA原子类型映射 (标准原子顺序)
        # 每个核苷酸通常有27个原子
        rna_atoms = [
            # 磷酸基团
            "P", "OP1", "OP2",
            # 糖基
            "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
            # 碱基原子（通用）- 实际会根据碱基类型调整
            "N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9",
            "O2", "N2", "O4", "O6", "N4"
        ]
        
        # 碱基特异性原子映射
        base_atoms = {
            'A': ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
            'U': ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
            'G': ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
            'C': ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"]
        }
        
        # 处理坐标形状
        if len(coords_np.shape) == 3:
            # [seq_len, atom_types, 3] -> [seq_len * atom_types, 3]
            seq_len, atom_types, _ = coords_np.shape
            coords_flat = coords_np.reshape(-1, 3)
        else:
            # [total_atoms, 3]
            coords_flat = coords_np
            seq_len = len(sequence)
            atom_types = coords_flat.shape[0] // seq_len if seq_len > 0 else 27
        
        # 写入PDB文件
        with open(output_path, 'w') as f:
            # PDB头部
            f.write("HEADER    RNA STRUCTURE PREDICTION\n")
            f.write("TITLE     RHOFOLD PREDICTED STRUCTURE\n")
            f.write("MODEL     1\n")
            
            atom_index = 1
            residue_index = 1
            
            for i, nucleotide in enumerate(sequence):
                if i >= seq_len:
                    break
                    
                # 选择该残基的原子
                start_atom = i * atom_types
                end_atom = min((i + 1) * atom_types, coords_flat.shape[0])
                residue_coords = coords_flat[start_atom:end_atom]
                
                # 获取该碱基的原子名称
                if nucleotide in base_atoms:
                    # 使用完整的原子列表：磷酸 + 糖 + 碱基
                    atom_names = rna_atoms[:12] + base_atoms[nucleotide]  # 前12个是磷酸和糖原子
                else:
                    # 使用通用原子名称
                    atom_names = rna_atoms
                
                # 限制原子数量以匹配坐标数量
                num_atoms = min(len(atom_names), residue_coords.shape[0])
                
                for j in range(num_atoms):
                    coord = residue_coords[j]
                    atom_name = atom_names[j] if j < len(atom_names) else f"X{j}"
                    
                    # 获取置信度（B-factor）
                    b_factor = 50.0  # 默认B-factor
                    if confidence_np is not None:
                        if len(confidence_np.shape) == 1 and i < len(confidence_np):
                            b_factor = confidence_np[i]
                        elif len(confidence_np.shape) == 2 and i < confidence_np.shape[0] and j < confidence_np.shape[1]:
                            b_factor = confidence_np[i, j]
                    
                    # 转换置信度为合理的B-factor范围
                    if b_factor > 1.0:
                        b_factor = min(b_factor, 100.0)
                    else:
                        b_factor = b_factor * 100.0
                    
                    # 写入ATOM记录
                    f.write(f"ATOM  {atom_index:5d} {atom_name:^4s} {nucleotide:>3s} A{residue_index:4d}    "
                           f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{1.00:6.2f}{b_factor:6.2f}           "
                           f"{atom_name[0]:>2s}\n")
                    
                    atom_index += 1
                
                residue_index += 1
            
            f.write("ENDMDL\n")
            f.write("END\n")
        
        if logger:
            logger.debug(f"使用自定义方法成功保存PDB文件: {output_path}")
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"自定义PDB保存方法失败: {e}")
        return False

def validate_rhofold_output(predicted_coords: torch.Tensor, 
                           sequence: str,
                           confidence: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    验证RhoFold模型输出的有效性
    
    Args:
        predicted_coords: 预测的原子坐标
        sequence: RNA序列
        confidence: 置信度分数 (可选)
        
    Returns:
        Dict: 验证结果
    """
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        # 检查坐标
        if predicted_coords is None:
            validation_result['valid'] = False
            validation_result['errors'].append("预测坐标为None")
            return validation_result
        
        # 转换为numpy数组进行检查
        if isinstance(predicted_coords, torch.Tensor):
            coords_np = predicted_coords.detach().cpu().numpy()
        else:
            coords_np = predicted_coords
        
        # 检查坐标维度
        if len(coords_np.shape) != 2 and len(coords_np.shape) != 3:
            validation_result['valid'] = False
            validation_result['errors'].append(f"坐标维度错误: {coords_np.shape}")
            return validation_result
        
        if coords_np.shape[-1] != 3:
            validation_result['valid'] = False
            validation_result['errors'].append(f"坐标最后一维应为3，实际为: {coords_np.shape[-1]}")
            return validation_result
        
        # 检查序列
        if not sequence or not isinstance(sequence, str):
            validation_result['valid'] = False
            validation_result['errors'].append("序列为空或格式错误")
            return validation_result
        
        # 检查RNA序列的有效性
        valid_nucleotides = set('AUCG')
        invalid_nucleotides = set(sequence.upper()) - valid_nucleotides
        if invalid_nucleotides:
            validation_result['warnings'].append(f"序列包含非标准核苷酸: {invalid_nucleotides}")
        
        # 检查坐标和序列长度的一致性
        seq_len = len(sequence)
        if len(coords_np.shape) == 3:
            coord_seq_len = coords_np.shape[0]
            atom_types = coords_np.shape[1]
        else:
            # 假设每个核苷酸有27个原子
            atom_types = 27
            coord_seq_len = coords_np.shape[0] // atom_types
        
        if coord_seq_len != seq_len:
            validation_result['warnings'].append(
                f"坐标序列长度({coord_seq_len})与输入序列长度({seq_len})不匹配"
            )
        
        # 检查坐标是否包含NaN或Inf
        if np.any(np.isnan(coords_np)) or np.any(np.isinf(coords_np)):
            validation_result['valid'] = False
            validation_result['errors'].append("坐标包含NaN或Inf值")
        
        # 检查坐标范围是否合理（通常RNA结构坐标在-100到100Å范围内）
        coord_range = np.max(np.abs(coords_np))
        if coord_range > 1000:
            validation_result['warnings'].append(f"坐标范围异常大: {coord_range:.2f}Å")
        
        # 检查置信度
        if confidence is not None:
            if isinstance(confidence, torch.Tensor):
                confidence_np = confidence.detach().cpu().numpy()
            else:
                confidence_np = confidence
            
            if np.any(np.isnan(confidence_np)) or np.any(np.isinf(confidence_np)):
                validation_result['warnings'].append("置信度包含NaN或Inf值")
            
            if np.any(confidence_np < 0) or np.any(confidence_np > 100):
                validation_result['warnings'].append("置信度值超出合理范围[0, 100]")
        
        # 添加信息
        validation_result['info'] = {
            'coord_shape': coords_np.shape,
            'sequence_length': seq_len,
            'estimated_atoms_per_residue': atom_types,
            'coord_range': float(coord_range),
            'has_confidence': confidence is not None
        }
        
    except Exception as e:
        validation_result['valid'] = False
        validation_result['errors'].append(f"验证过程发生错误: {str(e)}")
    
    return validation_result

def save_rhofold_metrics(metrics: Dict[str, Any], 
                        output_path: str,
                        logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    保存RhoFold相关的指标到文件
    
    Args:
        metrics: 指标字典
        output_path: 输出文件路径
        logger_instance: 日志器实例 (可选)
        
    Returns:
        bool: 是否成功保存
    """
    try:
        import json
        
        logger = logger_instance if logger_instance else logging.getLogger(__name__)
        
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 转换torch张量为可序列化的格式
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                serializable_metrics[key] = value.detach().cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        # 保存到JSON文件
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2, default=str)
        
        logger.debug(f"成功保存RhoFold指标: {output_path}")
        return True
        
    except Exception as e:
        if logger_instance:
            logger_instance.error(f"保存RhoFold指标失败: {e}")
        else:
            print(f"错误: 保存RhoFold指标失败: {e}")
        return False

def extract_rhofold_features(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    从RhoFold输出中提取有用的特征
    
    Args:
        output: RhoFold模型输出字典
        
    Returns:
        Dict: 提取的特征
    """
    features = {}
    
    try:
        # 提取坐标预测
        if 'cord_tns_pred' in output:
            coord_pred = output['cord_tns_pred']
            if isinstance(coord_pred, (list, tuple)):
                features['predicted_coords'] = coord_pred[-1]  # 最后一次迭代
                features['coord_iterations'] = len(coord_pred)
            else:
                features['predicted_coords'] = coord_pred
                features['coord_iterations'] = 1
        
        # 提取置信度
        if 'plddt' in output:
            plddt = output['plddt']
            features['confidence'] = plddt
            # 安全地计算平均置信度
            if isinstance(plddt, torch.Tensor):
                features['avg_confidence'] = torch.mean(plddt).item()
            elif isinstance(plddt, (list, tuple)) and len(plddt) > 0:
                if isinstance(plddt[0], torch.Tensor):
                    features['avg_confidence'] = torch.mean(plddt[0]).item()
                else:
                    features['avg_confidence'] = float(plddt[0]) if len(plddt) == 1 else sum(plddt) / len(plddt)
            else:
                features['avg_confidence'] = 0.0
        
        # 提取二级结构
        if 'ss' in output:
            features['secondary_structure'] = output['ss']
        
        # 提取距离预测
        if 'p' in output:
            features['dist_p'] = output['p']
        if 'c4_' in output:
            features['dist_c'] = output['c4_']
        if 'n' in output:
            features['dist_n'] = output['n']
        
        # 提取其他特征
        for key in ['single', 'pair']:
            if key in output:
                features[f'{key}_features'] = output[key]
        
    except Exception as e:
        print(f"警告: 提取RhoFold特征时发生错误: {e}")
    
    return features

# 为了向后兼容，创建别名
def diffold_coords_to_pdb(*args, **kwargs):
    """向后兼容的别名"""
    return rhofold_coords_to_pdb(*args, **kwargs)

def validate_diffold_output(*args, **kwargs):
    """向后兼容的别名"""
    return validate_rhofold_output(*args, **kwargs)
