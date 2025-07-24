import torch
from typing import List, Optional, Tuple, Union
from copy import deepcopy
import logging
from alphafold3_pytorch.inputs import (
    Alphafold3Input, 
    alphafold3_inputs_to_batched_atom_input,
    BatchedAtomInput
)

# 设置日志
logger = logging.getLogger(__name__)

# 添加lens_to_mask函数的导入或实现
def lens_to_mask(lens: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Convert a Tensor of lengths to a mask Tensor."""
    device = lens.device
    if max_len is None:
        max_len = int(lens.amax().item())
    arange = torch.arange(max_len, device=device)
    return arange < lens.unsqueeze(-1)


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
        # 处理RNA输入并自动删除额外氧原子
        batch_input, atom_mask = process_alphafold3_input(
            ss_rna=['AGUC'],
            remove_extra_oxygens=True
        )
        
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
    
    # 从molecule_atom_lens计算total_atoms
    molecule_atom_lens = batched_input.molecule_atom_lens
    total_atoms = molecule_atom_lens.sum(dim=-1)
    
    # 获取atom_seq_len（原子序列的最大长度）
    atom_seq_len = batched_input.atom_inputs.shape[1]
    
    # 【关键修复】稳健的atom_pos维度处理
    logger.debug(f"原始 atom_pos 维度: {batched_input.atom_pos.shape if batched_input.atom_pos is not None else 'None'}")
    
    # 检查atom_pos是否存在
    if batched_input.atom_pos is None:
        raise ValueError("batched_input.atom_pos 不能为 None")
    
    # 确保atom_pos是3维张量 [batch_size, num_atoms, 3]
    if batched_input.atom_pos.dim() == 4:
        # 如果是4维，压缩多余的维度
        if batched_input.atom_pos.shape[0] == 1:
            batched_input.atom_pos = batched_input.atom_pos.squeeze(0)
        elif batched_input.atom_pos.shape[1] == 1:
            batched_input.atom_pos = batched_input.atom_pos.squeeze(1)
        else:
            # 如果两个维度都不是1，选择保留最后3个维度
            batched_input.atom_pos = batched_input.atom_pos.view(-1, batched_input.atom_pos.shape[-2], batched_input.atom_pos.shape[-1])
    elif batched_input.atom_pos.dim() == 2:
        # 如果是2维，添加batch维度
        batched_input.atom_pos = batched_input.atom_pos.unsqueeze(0)
    elif batched_input.atom_pos.dim() != 3:
        raise ValueError(f"不支持的atom_pos维度: {batched_input.atom_pos.shape}")
    
    logger.debug(f"处理后 atom_pos 维度: {batched_input.atom_pos.shape}")
    
    # 获取处理后的维度信息
    atom_num_in_coords = batched_input.atom_pos.shape[1]
    batch_num = batched_input.atom_pos.shape[0]

    # 生成atom_mask，使用atom_pos的实际原子数量
    atom_mask = lens_to_mask(total_atoms, max_len=atom_num_in_coords)

    logger.debug(f"atom_pos: {batched_input.atom_pos.shape}")
    logger.debug(f"atom_mask: {atom_mask.shape}")
    
    # 如果atom_inputs的原子数量大于atom_pos的原子数量，需要填充
    if atom_seq_len > atom_num_in_coords:
        logger.debug(f"填充 atom_pos: {atom_num_in_coords} -> {atom_seq_len}")
        padding_atoms = atom_seq_len - atom_num_in_coords
        batched_input.atom_pos = torch.cat([
            batched_input.atom_pos, 
            torch.zeros(batch_num, padding_atoms, 3, device=batched_input.atom_pos.device)
        ], dim=1)
        atom_mask = torch.cat([
            atom_mask, 
            torch.zeros(batch_num, padding_atoms, device=atom_mask.device, dtype=atom_mask.dtype)
        ], dim=1)

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
    
    # 【添加原子数量对齐处理】与process_alphafold3_input保持一致
    logger.debug(f"Multiple inputs - 原始 atom_pos 维度: {batched_input.atom_pos.shape if batched_input.atom_pos is not None else 'None'}")
    
    if batched_input.atom_pos is not None:
        # 确保atom_pos存在的断言，帮助类型检查
        assert batched_input.atom_pos is not None
        
        # 确保atom_pos是3维张量 [batch_size, num_atoms, 3]
        if batched_input.atom_pos.dim() == 4:
            # 如果是4维，压缩多余的维度
            if batched_input.atom_pos.shape[0] == 1:
                batched_input.atom_pos = batched_input.atom_pos.squeeze(0)
            elif batched_input.atom_pos.shape[1] == 1:
                batched_input.atom_pos = batched_input.atom_pos.squeeze(1)
            else:
                # 如果两个维度都不是1，选择保留最后3个维度
                batched_input.atom_pos = batched_input.atom_pos.view(-1, batched_input.atom_pos.shape[-2], batched_input.atom_pos.shape[-1])
        elif batched_input.atom_pos.dim() == 2:
            # 如果是2维，添加batch维度
            batched_input.atom_pos = batched_input.atom_pos.unsqueeze(0)
        elif batched_input.atom_pos.dim() != 3:
            raise ValueError(f"不支持的atom_pos维度: {batched_input.atom_pos.shape}")
        
        logger.debug(f"Multiple inputs - 处理后 atom_pos 维度: {batched_input.atom_pos.shape}")
        
        # 获取处理后的维度信息
        atom_num_in_coords = batched_input.atom_pos.shape[1]
        batch_num = batched_input.atom_pos.shape[0]

        # 生成atom_mask，使用atom_pos的实际原子数量
        atom_mask = lens_to_mask(total_atoms, max_len=atom_num_in_coords)

        logger.debug(f"Multiple inputs - atom_pos: {batched_input.atom_pos.shape}")
        logger.debug(f"Multiple inputs - atom_mask: {atom_mask.shape}")
        
        # 如果atom_inputs的原子数量大于atom_pos的原子数量，需要填充
        if atom_seq_len > atom_num_in_coords:
            logger.debug(f"Multiple inputs - 填充 atom_pos: {atom_num_in_coords} -> {atom_seq_len}")
            padding_atoms = atom_seq_len - atom_num_in_coords
            
            # 确保batched_input.atom_pos不为None
            assert batched_input.atom_pos is not None
            batched_input.atom_pos = torch.cat([
                batched_input.atom_pos, 
                torch.zeros(batch_num, padding_atoms, 3, device=batched_input.atom_pos.device)
            ], dim=1)
            atom_mask = torch.cat([
                atom_mask, 
                torch.zeros(batch_num, padding_atoms, device=atom_mask.device, dtype=atom_mask.dtype)
            ], dim=1)
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