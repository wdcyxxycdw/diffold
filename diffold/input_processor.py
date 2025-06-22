import torch
from typing import List, Optional, Tuple, Union
from alphafold3_pytorch.inputs import (
    Alphafold3Input, 
    alphafold3_inputs_to_batched_atom_input,
    BatchedAtomInput
)

# 添加lens_to_mask函数的导入或实现
def lens_to_mask(lens: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Convert a Tensor of lengths to a mask Tensor."""
    device = lens.device
    if max_len is None:
        max_len = int(lens.amax().item())
    arange = torch.arange(max_len, device=device)
    return arange < lens.unsqueeze(-1)

def process_alphafold3_input(
    proteins: Optional[List[str]] = None,
    ss_dna: Optional[List[str]] = None,
    ss_rna: Optional[List[str]] = None,
    ds_dna: Optional[List[str]] = None,
    ds_rna: Optional[List[str]] = None,
    ligands: Optional[List[str]] = None,
    metal_ions: Optional[List[str]] = None,
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
        return_atom_mask: 是否同时返回atom_mask
        **kwargs: 其他传递给Alphafold3Input的参数
    
    返回:
        BatchedAtomInput: 预处理后的批量原子输入对象
        或者 (BatchedAtomInput, atom_mask): 如果return_atom_mask=True
    
    示例:
        # 简单蛋白质输入
        batch_input = process_alphafold3_input(
            proteins=['AG', 'MLEI']
        )
        
        # 带坐标的输入
        mock_atompos = [
            torch.randn(5, 3),  # 丙氨酸5个原子
            torch.randn(4, 3)   # 甘氨酸4个原子
        ]
        batch_input = process_alphafold3_input(
            proteins=['AG'],
            atom_pos=mock_atompos
        )
        
        # 多种分子类型
        batch_input = process_alphafold3_input(
            proteins=['MLEI'],
            ss_rna=['ACGU'],
            ligands=['CCO'],
            metal_ions=['Na']
        )
        
        # 同时返回atom_mask
        batch_input, atom_mask = process_alphafold3_input(
            proteins=['AG'],
            return_atom_mask=True
        )
    """
    
    # 设置默认值
    proteins = proteins or []
    ss_dna = ss_dna or []
    ss_rna = ss_rna or []
    ds_dna = ds_dna or []
    ds_rna = ds_rna or []
    ligands = ligands or []
    metal_ions = metal_ions or []
    
    # 创建Alphafold3Input对象
    alphafold3_input = Alphafold3Input(
        proteins=proteins,
        ss_dna=ss_dna,
        ss_rna=ss_rna,
        ds_dna=ds_dna,
        ds_rna=ds_rna,
        ligands=ligands,
        metal_ions=metal_ions,
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
    
    # 生成atom_mask
    atom_mask = lens_to_mask(total_atoms, max_len=atom_seq_len)
    
    return batched_input, atom_mask




def process_multiple_alphafold3_inputs(
    input_list: List[dict],
    atoms_per_window: int = 27,
    return_atom_mask: bool = False
) -> Union[BatchedAtomInput, Tuple[BatchedAtomInput, torch.Tensor]]:
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
            {'proteins': ['AG'], 'ligands': ['CCO']},
            {'proteins': ['MLEI'], 'ss_rna': ['ACGU']},
            {'proteins': ['GAVL']}
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
            proteins=input_dict.get('proteins', []),
            ss_dna=input_dict.get('ss_dna', []),
            ss_rna=input_dict.get('ss_rna', []),
            ds_dna=input_dict.get('ds_dna', []),
            ds_rna=input_dict.get('ds_rna', []),
            ligands=input_dict.get('ligands', []),
            metal_ions=input_dict.get('metal_ions', []),
            atom_pos=input_dict.get('atom_pos'),
            add_atom_ids=input_dict.get('add_atom_ids', False),
            add_atompair_ids=input_dict.get('add_atompair_ids', False),
            directed_bonds=input_dict.get('directed_bonds', False),
            custom_atoms=input_dict.get('custom_atoms'),
            custom_bonds=input_dict.get('custom_bonds'),
            **{k: v for k, v in input_dict.items() if k not in [
                'proteins', 'ss_dna', 'ss_rna', 'ds_dna', 'ds_rna', 
                'ligands', 'metal_ions', 'atom_pos', 'add_atom_ids', 
                'add_atompair_ids', 'directed_bonds', 'custom_atoms', 'custom_bonds'
            ]}
        )
        alphafold3_inputs.append(alphafold3_input)
    
    # 批量转换
    batched_input = alphafold3_inputs_to_batched_atom_input(
        alphafold3_inputs,
        atoms_per_window=atoms_per_window
    )
    
    # 如果需要返回atom_mask，则计算它
    if return_atom_mask:
        # 从molecule_atom_lens计算total_atoms
        molecule_atom_lens = batched_input.molecule_atom_lens
        total_atoms = molecule_atom_lens.sum(dim=-1)
        
        # 获取atom_seq_len（原子序列的最大长度）
        atom_seq_len = batched_input.atom_inputs.shape[1]
        
        # 生成atom_mask
        atom_mask = lens_to_mask(total_atoms, max_len=atom_seq_len)
        
        return batched_input, atom_mask
    
    return batched_input


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
    print("=== AlphaFold3 Input Summary ===")
    shapes = get_input_shapes(batched_input)
    
    # 主要维度信息
    if 'atom_inputs' in shapes and isinstance(batched_input.atom_inputs, torch.Tensor):
        batch_size, num_atoms, atom_dim = batched_input.atom_inputs.shape
        print(f"Batch size: {batch_size}")
        print(f"Total atoms: {num_atoms}")
        print(f"Atom feature dim: {atom_dim}")
    
    if 'molecule_atom_lens' in shapes and isinstance(batched_input.molecule_atom_lens, torch.Tensor):
        batch_size, num_tokens = batched_input.molecule_atom_lens.shape
        print(f"Number of tokens: {num_tokens}")
    
    # 如果提供了atom_mask，显示其信息
    if atom_mask is not None:
        print(f"Atom mask shape: {list(atom_mask.shape)}")
        total_valid_atoms = atom_mask.sum(dim=-1)
        print(f"Valid atoms per batch: {total_valid_atoms.tolist()}")
    
    print("\n=== Field Shapes ===")
    for key, shape in shapes.items():
        print(f"{key}: {shape}")


# 示例使用函数
def example_usage():
    """展示如何使用这些函数的示例"""
    
    print("=== 示例1: 简单蛋白质输入 ===")
    result = process_alphafold3_input(
        proteins=['AG']
    )
    # 确保result是BatchedAtomInput类型
    if isinstance(result, BatchedAtomInput):
        print_input_summary(result)
    
    print("\n=== 示例2: 带坐标和atom_mask的输入 ===")
    mock_atompos = [
        torch.randn(5, 3),  # 丙氨酸
        torch.randn(4, 3)   # 甘氨酸
    ]
    result = process_alphafold3_input(
        proteins=['AG'],
        atom_pos=mock_atompos,
    )
    if isinstance(result, tuple):
        batch_input, atom_mask = result
        print_input_summary(batch_input, atom_mask)
        print(atom_mask)
    
    print("\n=== 示例3: 多种分子类型 ===")
    result = process_alphafold3_input(
        proteins=['MLEI'],
        ss_rna=['ACGU'],
        ligands=['CCO'],
        metal_ions=['Na'],
    )
    if isinstance(result, tuple):
        batch_input, atom_mask = result
        print_input_summary(batch_input, atom_mask)
    
    print("\n=== 示例4: 批量处理多个输入 ===")
    inputs = [
        {'proteins': ['AG']},
        {'proteins': ['MLEI'], 'ligands': ['CCO']},
        {'proteins': ['GAVL'], 'ss_rna': ['ACGU']}
    ]
    result = process_multiple_alphafold3_inputs(
        inputs
    )
    if isinstance(result, tuple):
        batch_input, atom_mask = result
        print_input_summary(batch_input, atom_mask)


if __name__ == "__main__":
    example_usage() 