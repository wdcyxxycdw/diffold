import torch
from typing import List, Optional, Tuple, Union
from copy import deepcopy
import logging
from alphafold3_pytorch.inputs import (
    Alphafold3Input, 
    alphafold3_inputs_to_batched_atom_input,
    BatchedAtomInput
)

from rhofold.utils.constants import ATOM_NAMES_PER_RESD

# 设置日志
logger = logging.getLogger(__name__)


def expand_windowed_atompairs(windowed_atompairs: torch.Tensor, total_atoms: int, window_size: int = 27) -> torch.Tensor:
    """
    将窗口化的atompair张量展开成完整的原子对矩阵
    
    参数:
        windowed_atompairs: [batch, num_windows, window_size, window_size*2, feature_dim]
        total_atoms: 总原子数
        window_size: 窗口大小，默认27
    
    返回:
        expanded: [batch, total_atoms, total_atoms, feature_dim]
    """
    batch_size, num_windows, w, w2, feature_dim = windowed_atompairs.shape
    device = windowed_atompairs.device
    
    assert w == window_size, f"窗口大小不匹配: 期望{window_size}, 实际{w}"
    assert w2 == window_size * 2, f"窗口宽度不匹配: 期望{window_size*2}, 实际{w2}"
    
    # 创建完整的原子对矩阵
    expanded = torch.zeros(batch_size, total_atoms, total_atoms, feature_dim, device=device)
    
    # 每个窗口的步长是window_size（不重叠）
    processed_windows = 0
    for window_idx in range(num_windows):
        # 计算当前窗口的行范围：连续的window_size个原子
        start_i = window_idx * window_size
        end_i = min(start_i + window_size, total_atoms)
        actual_size_i = end_i - start_i
        
        # 如果窗口起始位置超出范围，停止处理
        if start_i >= total_atoms:
            logger.debug(f"窗口{window_idx}: 起始位置{start_i}超出原子数{total_atoms}，停止处理")
            break
        
        # 计算当前窗口的列范围：从当前位置开始的window_size*2个原子
        start_j = start_i  # 列也从当前窗口位置开始
        end_j = min(start_j + window_size * 2, total_atoms)
        actual_size_j = end_j - start_j
        
        # 边界检查：确保有有效数据可处理
        if actual_size_i > 0 and actual_size_j > 0:
            # 确保不超出窗口张量的维度
            safe_size_i = min(actual_size_i, w)
            safe_size_j = min(actual_size_j, w2)
            
            if safe_size_i > 0 and safe_size_j > 0:
                # 从窗口张量中提取对应部分（只提取有效范围）
                window_data = windowed_atompairs[:, window_idx, :safe_size_i, :safe_size_j, :]
                
                # 将数据复制到完整矩阵的对应位置
                expanded[:, start_i:start_i+safe_size_i, start_j:start_j+safe_size_j, :] = window_data
                
                processed_windows += 1
                logger.debug(f"窗口{window_idx}: 行[{start_i}:{start_i+safe_size_i}] x 列[{start_j}:{start_j+safe_size_j}] "
                           f"(实际大小: {safe_size_i}x{safe_size_j})")
            else:
                logger.debug(f"窗口{window_idx}: 安全大小为0，跳过处理")
        else:
            logger.debug(f"窗口{window_idx}: 实际大小为0，跳过处理")
    
    logger.debug(f"展开窗口化张量: {windowed_atompairs.shape} -> {expanded.shape}")
    logger.debug(f"窗口逻辑: 处理了{processed_windows}/{num_windows}个窗口, 步长={window_size}, 窗口大小={window_size}x{window_size*2}")
    return expanded


def rewindow_atompairs(expanded_atompairs: torch.Tensor, window_size: int = 27) -> torch.Tensor:
    """
    将完整的原子对矩阵重新转换为窗口化表示
    
    参数:
        expanded_atompairs: [batch, total_atoms, total_atoms, feature_dim]
        window_size: 窗口大小，默认27
    
    返回:
        windowed: [batch, num_windows, window_size, window_size*2, feature_dim]
    """
    batch_size, total_atoms, _, feature_dim = expanded_atompairs.shape
    device = expanded_atompairs.device
    
    # 计算窗口数量（步长为window_size，不重叠）
    num_windows = (total_atoms + window_size - 1) // window_size
    
    # 创建窗口化张量
    windowed = torch.zeros(batch_size, num_windows, window_size, window_size * 2, feature_dim, device=device)
    
    processed_windows = 0
    for window_idx in range(num_windows):
        # 计算当前窗口的行范围：连续的window_size个原子
        start_i = window_idx * window_size
        end_i = min(start_i + window_size, total_atoms)
        actual_size_i = end_i - start_i
        
        # 如果窗口起始位置超出范围，停止处理
        if start_i >= total_atoms:
            logger.debug(f"重新窗口化-窗口{window_idx}: 起始位置{start_i}超出原子数{total_atoms}，停止处理")
            break
        
        # 计算当前窗口的列范围：从当前位置开始的window_size*2个原子
        start_j = start_i  # 列也从当前窗口位置开始
        end_j = min(start_j + window_size * 2, total_atoms)
        actual_size_j = end_j - start_j
        
        # 边界检查：确保有有效数据可处理
        if actual_size_i > 0 and actual_size_j > 0:
            # 确保不超出窗口张量的维度
            safe_size_i = min(actual_size_i, window_size)
            safe_size_j = min(actual_size_j, window_size * 2)
            
            if safe_size_i > 0 and safe_size_j > 0:
                # 从完整矩阵中提取对应部分（只提取有效范围）
                matrix_data = expanded_atompairs[:, start_i:start_i+safe_size_i, start_j:start_j+safe_size_j, :]
                
                # 将数据复制到窗口张量的对应位置（注意：超出部分保持为0）
                windowed[:, window_idx, :safe_size_i, :safe_size_j, :] = matrix_data
                
                processed_windows += 1
                logger.debug(f"重新窗口化-窗口{window_idx}: 行[{start_i}:{start_i+safe_size_i}] x 列[{start_j}:{start_j+safe_size_j}] "
                           f"-> 窗口位置[0:{safe_size_i}, 0:{safe_size_j}]")
            else:
                logger.debug(f"重新窗口化-窗口{window_idx}: 安全大小为0，跳过处理")
        else:
            logger.debug(f"重新窗口化-窗口{window_idx}: 实际大小为0，跳过处理")
    
    logger.debug(f"重新窗口化张量: {expanded_atompairs.shape} -> {windowed.shape}")
    logger.debug(f"重新窗口化逻辑: 处理了{processed_windows}/{num_windows}个窗口, 步长={window_size}, 窗口大小={window_size}x{window_size*2}")
    return windowed

# 添加lens_to_mask函数的导入或实现
def lens_to_mask(lens: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Convert a Tensor of lengths to a mask Tensor."""
    device = lens.device
    if max_len is None:
        max_len = int(lens.amax().item())
    arange = torch.arange(max_len, device=device)
    return arange < lens.unsqueeze(-1)


def align_input(af_in, seq):
    """
    对齐AlphaFold3输入和目标坐标
    将AlphaFold3格式转换为RhoFold+格式，删除每个碱基最后一个原子（桥连氧）
    
    参数:
        af_in: BatchedAtomInput，AlphaFold3的输入对象
        seq: List[str] 或 str，RNA序列
    
    返回:
        (对齐后的BatchedAtomInput对象, 对齐后的atom_mask)
    """
    # 获取每个碱基的原子数量
    ATOMS_PER_BASE = {
        "A": len(ATOM_NAMES_PER_RESD["A"]),  # 22
        "G": len(ATOM_NAMES_PER_RESD["G"]),  # 23  
        "U": len(ATOM_NAMES_PER_RESD["U"]),  # 20
        "C": len(ATOM_NAMES_PER_RESD["C"])   # 20
    }
    
    # 处理输入序列格式
    if isinstance(seq, str):
        sequences = [seq]
    elif isinstance(seq, list):
        sequences = seq
    else:
        raise ValueError(f"不支持的序列格式: {type(seq)}")
    
    logger.debug(f"开始对齐输入，序列数量: {len(sequences)}")
    
    # 🔍 记录输入数据的基本信息
    if af_in.atom_pos is not None:
        logger.debug(f"🔍 AlphaFold3输入原子数: {af_in.atom_pos.shape[1]}")
    if af_in.molecule_atom_lens is not None:
        total_expected_atoms = af_in.molecule_atom_lens.sum(dim=-1)
        logger.debug(f"🔍 molecule_atom_lens期望原子数: {total_expected_atoms.tolist()}")
    
    # 深拷贝af_in以避免修改原始数据
    af_in_aligned = deepcopy(af_in)
    
    # 🔍 先检查molecule_atom_lens来理解AlphaFold3的数据结构
    if af_in.molecule_atom_lens is not None:
        logger.debug(f"molecule_atom_lens: {af_in.molecule_atom_lens}")
        
        # AlphaFold3可能将每个碱基作为独立分子处理
        # 根据molecule_atom_lens计算每个分子的keep_indices
        keep_indices = []
        current_atom_idx = 0
        
        batch_size = af_in.molecule_atom_lens.shape[0]
        for batch_idx in range(batch_size):
            molecule_lens = af_in.molecule_atom_lens[batch_idx]
            seq_idx = 0
            
            for mol_idx, mol_len in enumerate(molecule_lens):
                if mol_len == 0:
                    continue  # 跳过空分子
                    
                # 尝试从sequence推断碱基类型
                if seq_idx < len(sequences[batch_idx] if batch_idx < len(sequences) else sequences[0]):
                    base = sequences[batch_idx][seq_idx] if batch_idx < len(sequences) else sequences[0][seq_idx]
                    seq_idx += 1
                    
                    if base in ATOMS_PER_BASE:
                        expected_atoms = ATOMS_PER_BASE[base]
                        actual_atoms = mol_len.item()
                        
                        logger.debug(f"分子{mol_idx}: 碱基={base}, 预期原子={expected_atoms}, 实际原子={actual_atoms}")
                        
                        # 保留该分子的前 (actual_atoms - 1) 个原子
                        # 注意：使用实际原子数而不是预期原子数
                        if actual_atoms > 1:
                            mol_keep_indices = list(range(current_atom_idx, current_atom_idx + actual_atoms - 1))
                        else:
                            mol_keep_indices = list(range(current_atom_idx, current_atom_idx + actual_atoms))
                        
                        keep_indices.extend(mol_keep_indices)
                        current_atom_idx += actual_atoms
                    else:
                        logger.warning(f"未知碱基: {base}，保留所有原子")
                        mol_keep_indices = list(range(current_atom_idx, current_atom_idx + mol_len.item()))
                        keep_indices.extend(mol_keep_indices)
                        current_atom_idx += mol_len.item()
                else:
                    # 如果没有对应的序列信息，保留所有原子
                    mol_keep_indices = list(range(current_atom_idx, current_atom_idx + mol_len.item()))
                    keep_indices.extend(mol_keep_indices)
                    current_atom_idx += mol_len.item()
    else:
        # 回退到原始逻辑
        logger.debug("使用回退逻辑：基于序列计算keep_indices")
        keep_indices = []
        current_atom_idx = 0
        
        for seq_idx, sequence in enumerate(sequences):
            logger.debug(f"处理序列[{seq_idx}]: {sequence[:20]}..." if len(sequence) > 20 else f"处理序列[{seq_idx}]: {sequence}")
            
            for base_idx, base in enumerate(sequence):
                if base not in ATOMS_PER_BASE:
                    logger.warning(f"未知碱基: {base}，跳过")
                    continue
                    
                atoms_count = ATOMS_PER_BASE[base]
                
                # 保留该碱基的前 (atoms_count - 1) 个原子，删除最后一个原子（桥连氧）
                base_keep_indices = list(range(current_atom_idx, current_atom_idx + atoms_count - 1))
                keep_indices.extend(base_keep_indices)
                
                # 更新当前原子索引（使用AlphaFold3的完整原子数）
                current_atom_idx += atoms_count
    
    keep_indices = torch.tensor(keep_indices, dtype=torch.long)
    logger.debug(f"保留的原子数量: {len(keep_indices)}, 原始原子数量: {current_atom_idx}")

    # 获取设备信息
    device = af_in.atom_inputs.device if af_in.atom_inputs is not None else torch.device('cpu')
    keep_indices = keep_indices.to(device)
    
    # 对所有与原子数相关的张量进行裁切，需要根据每个张量的实际原子数量来处理
    
    # 1. atom_inputs: [batch_size, num_atoms, feature_dim]
    if af_in_aligned.atom_inputs is not None:
        original_shape = af_in_aligned.atom_inputs.shape
        actual_atoms = original_shape[1]
        # 只使用在范围内的索引
        valid_keep_indices = keep_indices[keep_indices < actual_atoms]
        if len(valid_keep_indices) > 0:
            af_in_aligned.atom_inputs = af_in_aligned.atom_inputs[:, valid_keep_indices, :]
            logger.debug(f"atom_inputs: {original_shape} -> {af_in_aligned.atom_inputs.shape}")
        else:
            logger.warning(f"atom_inputs: 没有有效的keep_indices，原子数 {actual_atoms}")
    
    # 2. atompair_inputs: 处理窗口化的原子对输入 [batch_size, num_windows, window_size, window_size*2, feature_dim]
    if af_in_aligned.atompair_inputs is not None:
        original_shape = af_in_aligned.atompair_inputs.shape
        logger.debug(f"处理atompair_inputs: {original_shape}")
        
        # 检测是否为窗口化格式 (5维张量)
        if len(original_shape) == 5:
            # 窗口化格式: [batch, num_windows, window_size, window_size*2, feature_dim]
            batch_size, num_windows, w, w2, feature_dim = original_shape
            window_size = 27  # 固定窗口大小
            
            logger.debug(f"检测到窗口化格式: batch={batch_size}, windows={num_windows}, w={w}, w2={w2}, features={feature_dim}")
            
            # 估算原始总原子数
            original_total_atoms = current_atom_idx
            
            # 1. 展开窗口化张量
            try:
                expanded_atompairs = expand_windowed_atompairs(
                    af_in_aligned.atompair_inputs, 
                    original_total_atoms, 
                    window_size
                )
                logger.debug(f"展开成功: {original_shape} -> {expanded_atompairs.shape}")
                
                # 2. 对展开的张量进行裁切
                if len(keep_indices) > 0:
                    # 确保索引在有效范围内
                    max_valid_idx = expanded_atompairs.shape[1] - 1
                    valid_keep_indices = keep_indices[keep_indices <= max_valid_idx]
                    
                    if len(valid_keep_indices) > 0:
                        # 裁切原子对矩阵的两个维度
                        cropped_atompairs = expanded_atompairs[:, valid_keep_indices, :, :]
                        cropped_atompairs = cropped_atompairs[:, :, valid_keep_indices, :]
                        logger.debug(f"裁切成功: {expanded_atompairs.shape} -> {cropped_atompairs.shape}")
                        
                        # 3. 重新窗口化
                        new_total_atoms = len(valid_keep_indices)
                        windowed_atompairs = rewindow_atompairs(cropped_atompairs, window_size)
                        
                        af_in_aligned.atompair_inputs = windowed_atompairs
                        logger.debug(f"重新窗口化成功: {cropped_atompairs.shape} -> {windowed_atompairs.shape}")
                    else:
                        logger.warning(f"atompair_inputs: 没有有效的keep_indices进行裁切")
                else:
                    logger.warning(f"atompair_inputs: keep_indices为空")
                    
            except Exception as e:
                logger.error(f"窗口化处理失败: {e}")
                logger.warning("跳过atompair_inputs的处理")
                
        elif len(original_shape) == 4:
            # 标准格式: [batch_size, num_atoms, num_atoms, feature_dim]
            actual_atoms = original_shape[1]
            valid_keep_indices = keep_indices[keep_indices < actual_atoms]
            if len(valid_keep_indices) > 0:
                # 同时裁切两个原子维度
                af_in_aligned.atompair_inputs = af_in_aligned.atompair_inputs[:, valid_keep_indices, :, :]
                af_in_aligned.atompair_inputs = af_in_aligned.atompair_inputs[:, :, valid_keep_indices, :]
                logger.debug(f"标准格式裁切: {original_shape} -> {af_in_aligned.atompair_inputs.shape}")
            else:
                logger.warning(f"atompair_inputs: 没有有效的keep_indices，原子数 {actual_atoms}")
        else:
            logger.warning(f"不支持的atompair_inputs格式: {original_shape}")
            logger.debug(f"跳过atompair_inputs处理")
    
    # 4. 更新molecule_atom_lens以反映新的原子数量
    if af_in_aligned.molecule_atom_lens is not None:
        original_lens = af_in_aligned.molecule_atom_lens.clone()
        
        # 根据新的理解更新molecule_atom_lens
        # 每个分子（碱基）减少1个原子（如果原子数>1）
        for batch_idx in range(af_in_aligned.molecule_atom_lens.shape[0]):
            molecule_lens = af_in_aligned.molecule_atom_lens[batch_idx]
            seq_idx = 0
            
            for mol_idx in range(len(molecule_lens)):
                if molecule_lens[mol_idx] == 0:
                    continue
                    
                # 尝试从sequence推断碱基类型
                if seq_idx < len(sequences[batch_idx] if batch_idx < len(sequences) else sequences[0]):
                    base = sequences[batch_idx][seq_idx] if batch_idx < len(sequences) else sequences[0][seq_idx]
                    seq_idx += 1
                    
                    if base in ATOMS_PER_BASE and molecule_lens[mol_idx] > 1:
                        # 每个碱基减少1个原子（桥连氧）
                        af_in_aligned.molecule_atom_lens[batch_idx, mol_idx] -= 1
                        logger.debug(f"分子{mol_idx}({base}): {molecule_lens[mol_idx]} -> {af_in_aligned.molecule_atom_lens[batch_idx, mol_idx]}")
        
        logger.debug(f"molecule_atom_lens: {original_lens} -> {af_in_aligned.molecule_atom_lens}")
    
    # 5. 处理其他可能与原子索引相关的字段
    
    # atom_parent_ids: [batch_size, num_atoms] 
    if af_in_aligned.atom_parent_ids is not None:
        original_shape = af_in_aligned.atom_parent_ids.shape
        actual_atoms = original_shape[1]
        # 只使用在范围内的索引
        valid_keep_indices = keep_indices[keep_indices < actual_atoms]
        if len(valid_keep_indices) > 0:
            af_in_aligned.atom_parent_ids = af_in_aligned.atom_parent_ids[:, valid_keep_indices]
            logger.debug(f"atom_parent_ids: {original_shape} -> {af_in_aligned.atom_parent_ids.shape}")
        else:
            logger.warning(f"atom_parent_ids: 没有有效的keep_indices，原子数 {actual_atoms}")
    
    # molecule_atom_indices: 可能需要重新映射索引
    if af_in_aligned.molecule_atom_indices is not None:
        original_shape = af_in_aligned.molecule_atom_indices.shape
        
        # 基于实际的keep_indices创建索引映射
        if len(keep_indices) > 0:
            max_original_idx = max(current_atom_idx, keep_indices.max().item() + 1)
            old_to_new_mapping = torch.full((max_original_idx,), -1, dtype=torch.long, device=device)
            old_to_new_mapping[keep_indices] = torch.arange(len(keep_indices), device=device)
            
            # 重新映射molecule_atom_indices
            flat_indices = af_in_aligned.molecule_atom_indices.flatten()
            valid_mask = (flat_indices >= 0) & (flat_indices < len(old_to_new_mapping))
            
            new_flat_indices = flat_indices.clone()
            new_flat_indices[valid_mask] = old_to_new_mapping[flat_indices[valid_mask]]
            
            af_in_aligned.molecule_atom_indices = new_flat_indices.reshape(af_in_aligned.molecule_atom_indices.shape)
            logger.debug(f"molecule_atom_indices: {original_shape} -> {af_in_aligned.molecule_atom_indices.shape}")
        else:
            logger.warning(f"molecule_atom_indices: 没有有效的keep_indices进行重映射")
    
    # 6. 生成对齐后的atom_mask
    # 基于新的molecule_atom_lens生成atom_mask
    aligned_atom_mask = None
    if af_in_aligned.molecule_atom_lens is not None:
        # 计算新的总原子数
        total_atoms_aligned = af_in_aligned.molecule_atom_lens.sum(dim=-1)
        max_atoms_aligned = len(keep_indices)
        
        # 生成对齐后的atom_mask
        aligned_atom_mask = lens_to_mask(total_atoms_aligned, max_len=max_atoms_aligned)
        logger.debug(f"生成对齐后的atom_mask: {aligned_atom_mask.shape}")
    
    logger.debug("输入对齐完成")
    return af_in_aligned, aligned_atom_mask

def process_alphafold3_input(
    ss_rna: Optional[List[str]] = None,
    atom_pos: Optional[List[torch.Tensor]] = None,
    atoms_per_window: int = 27,
    add_atom_ids: bool = False,
    add_atompair_ids: bool = False,
    directed_bonds: bool = False,
    custom_atoms: Optional[List[str]] = None,
    custom_bonds: Optional[List[str]] = None,
    **kwargs
) ->  Tuple[BatchedAtomInput, torch.Tensor]:
    """
    处理AlphaFold3输入序列，返回预处理后的BatchedAtomInput对象
    
    参数:
        proteins: 蛋白质序列列表，例如 ['MLEI', 'AG']
        ss_dna: 单链DNA序列列表，例如 ['ACGT']
        ss_rna: 单链RNA序列列表，例如 ['ACGU']
        ds_dna: 双链DNA序列列表，例如 ['ACGT']
        ds_rna: 双链RNA序列列表，例如 ['ACGU']
        ligands: 配体SMILES列表，例如 ['CCO', 'CC(=O)O']
        metal_ions: 金属离子列表，例如 ['Na', 'Ca']
        atom_pos: 原子坐标列表，每个元素为 [num_atoms, 3] 的张量
        atoms_per_window: 每个窗口的原子数，用于内存优化
        add_atom_ids: 是否添加原子ID嵌入
        add_atompair_ids: 是否添加原子对ID嵌入
        directed_bonds: 是否使用有向键
        custom_atoms: 自定义原子类型列表
        custom_bonds: 自定义键类型列表
        remove_extra_oxygens: 是否删除额外的氧原子以匹配RhoFold格式
        **kwargs: 其他传递给Alphafold3Input的参数
    
    返回:
        BatchedAtomInput: 预处理后的批量原子输入对象
        atom_mask: 原子掩码张量
    
    示例:
        # 保持原始AlphaFold3格式
        batch_input, atom_mask = process_alphafold3_input(
            ss_rna=['AGUC'],
            remove_extra_oxygens=False
        )
    """
    
    # 设置默认值
    ss_rna = ss_rna or []
    
    # 创建Alphafold3Input对象
    alphafold3_input = Alphafold3Input(
        ss_rna=ss_rna,
        atom_pos=atom_pos,
        add_atom_ids=add_atom_ids,
        add_atompair_ids=add_atompair_ids,
        directed_bonds=directed_bonds,
        custom_atoms=custom_atoms,
        custom_bonds=custom_bonds,
        **kwargs
    )
    
    # 转换为BatchedAtomInput
    batched_input = alphafold3_inputs_to_batched_atom_input(
        alphafold3_input,
        atoms_per_window=atoms_per_window
    )
    
    # 🐛 BUG修复：不要直接覆盖atom_pos，这会导致数据不一致
    # batched_input.atom_pos=atom_pos[0]  # 移除这行有问题的代码
    
    # 🔍 调试信息：比较AlphaFold3处理前后的原子数量
    original_atom_count = atom_pos[0].shape[0] if atom_pos else 0
    processed_atom_count = batched_input.atom_pos.shape[1] if batched_input.atom_pos is not None else 0
    
    logger.debug(f"🔍 原子数量对比:")
    logger.debug(f"  用户输入原子数: {original_atom_count}")
    logger.debug(f"  AlphaFold3处理后原子数: {processed_atom_count}")
    if original_atom_count != processed_atom_count:
        logger.warning(f"⚠️ 原子数量不匹配！用户输入={original_atom_count}, AlphaFold3处理后={processed_atom_count}")

    # 从molecule_atom_lens计算total_atoms
    molecule_atom_lens = batched_input.molecule_atom_lens
    total_atoms = molecule_atom_lens.sum(dim=-1)
    
    # 获取处理后的维度信息
    atom_num_in_coords = batched_input.atom_inputs.shape[1]

    # 生成atom_mask，使用AlphaFold3处理后的原子数量
    atom_mask = lens_to_mask(total_atoms, max_len=atom_num_in_coords)

    logger.debug(f"📊 最终尺寸:")
    logger.debug(f"  atom_pos: {batched_input.atom_pos.shape if batched_input.atom_pos is not None else 'None'}")
    logger.debug(f"  atom_mask: {atom_mask.shape}")
    logger.debug(f"  molecule_atom_lens总数: {total_atoms.tolist()}")

    return batched_input, atom_mask


def process_multiple_alphafold3_inputs(
    input_list: List[dict],
    atoms_per_window: int = 27,
) -> Tuple[BatchedAtomInput, torch.Tensor]:
    """
    处理多个AlphaFold3输入，返回批量处理后的结果
    
    参数:
        input_list: 输入字典列表，每个字典包含序列等参数
        atoms_per_window: 每个窗口的原子数
        return_atom_mask: 是否同时返回atom_mask
    
    返回:
        BatchedAtomInput: 批量处理后的输入对象
        或者 (BatchedAtomInput, atom_mask): 如果return_atom_mask=True
    
    示例:
        inputs = [
            {'ss_rna': ['ACGU', 'ACGU'], 'atom_pos': [torch.randn(120, 3), torch.randn(120, 3)]}
        ]
        batch_input = process_multiple_alphafold3_inputs(inputs)
        
        # 同时返回atom_mask
        batch_input, atom_mask = process_multiple_alphafold3_inputs(
            inputs, return_atom_mask=True
        )
    """
    
    alphafold3_inputs = []
    
    for input_dict in input_list:
        alphafold3_input = Alphafold3Input(
            ss_rna=input_dict.get('ss_rna', []),
            atom_pos=input_dict.get('atom_pos'),
            add_atom_ids=input_dict.get('add_atom_ids', False),
            add_atompair_ids=input_dict.get('add_atompair_ids', False),
            directed_bonds=input_dict.get('directed_bonds', False),
            custom_atoms=input_dict.get('custom_atoms'),
            custom_bonds=input_dict.get('custom_bonds'),
            **{k: v for k, v in input_dict.items() if k not in [
                'ss_rna', 'atom_pos', 'add_atom_ids', 
                'add_atompair_ids', 'directed_bonds', 'custom_atoms', 'custom_bonds'
            ]}
        )
        alphafold3_inputs.append(alphafold3_input)
    
    # 批量转换
    batched_input = alphafold3_inputs_to_batched_atom_input(
        alphafold3_inputs,
        atoms_per_window=atoms_per_window
    )
    
    # 从molecule_atom_lens计算total_atoms
    molecule_atom_lens = batched_input.molecule_atom_lens
    total_atoms = molecule_atom_lens.sum(dim=-1)
    
    # 获取atom_seq_len（原子序列的最大长度）
    atom_seq_len = batched_input.atom_inputs.shape[1]
    
    if batched_input.atom_pos is not None:
        # 获取处理后的维度信息
        atom_num_in_coords = batched_input.atom_pos.shape[1]

        # 生成atom_mask，使用atom_pos的实际原子数量
        atom_mask = lens_to_mask(total_atoms, max_len=atom_num_in_coords)

        logger.debug(f"Multiple inputs - atom_pos: {batched_input.atom_pos.shape}")
        logger.debug(f"Multiple inputs - atom_mask: {atom_mask.shape}")

    else:
        # 如果没有atom_pos，使用默认的atom_mask生成
        atom_mask = lens_to_mask(total_atoms, max_len=atom_seq_len)
    
    return batched_input, atom_mask



def get_input_shapes(batched_input: BatchedAtomInput) -> dict:
    """
    获取BatchedAtomInput各个字段的形状信息
    
    参数:
        batched_input: BatchedAtomInput对象
    
    返回:
        dict: 包含各字段形状的字典
    """
    shapes = {}
    input_dict = batched_input.dict()
    
    for key, value in input_dict.items():
        if value is not None:
            if isinstance(value, torch.Tensor):
                shapes[key] = list(value.shape)
            elif isinstance(value, (list, tuple)):
                shapes[key] = f"list/tuple of length {len(value)}"
            else:
                shapes[key] = str(type(value))
        else:
            shapes[key] = "None"
    
    return shapes


def print_input_summary(batched_input: BatchedAtomInput, atom_mask: Optional[torch.Tensor] = None):
    """
    打印BatchedAtomInput的摘要信息
    
    参数:
        batched_input: BatchedAtomInput对象
        atom_mask: 可选的atom_mask张量
    """
    logger.info("=== AlphaFold3 Input Summary ===")
    shapes = get_input_shapes(batched_input)
    
    # 主要维度信息
    if 'atom_inputs' in shapes and isinstance(batched_input.atom_inputs, torch.Tensor):
        batch_size, num_atoms, atom_dim = batched_input.atom_inputs.shape
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Total atoms: {num_atoms}")
        logger.info(f"Atom feature dim: {atom_dim}")
    
    if 'molecule_atom_lens' in shapes and isinstance(batched_input.molecule_atom_lens, torch.Tensor):
        batch_size, num_tokens = batched_input.molecule_atom_lens.shape
        logger.info(f"Number of tokens: {num_tokens}")
    
    # 如果提供了atom_mask，显示其信息
    if atom_mask is not None:
        logger.info(f"Atom mask shape: {list(atom_mask.shape)}")
        total_valid_atoms = atom_mask.sum(dim=-1)
        logger.info(f"Valid atoms per batch: {total_valid_atoms.tolist()}")
    
    logger.info("\n=== Field Shapes ===")
    for key, shape in shapes.items():
        logger.info(f"{key}: {shape}")



if __name__ == "__main__":
    af_input = process_alphafold3_input(
        ss_rna=['ACGU'],
        atom_pos=[torch.randn(120, 3)]
    )