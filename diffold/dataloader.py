import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Union, Tuple
import logging
from pathlib import Path
import pickle
import re

# 导入RhoFold相关模块
from rhofold.utils.converter import RNAConverter
from rhofold.utils.constants import RNA_CONSTANTS
from rhofold.utils.alphabet import get_features, read_fas

# from .input_processor import process_alphafold3_input  # 测试时注释掉

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAtomMaskGenerator:
    """集成的缺失原子掩码生成器"""
    
    def __init__(self):
        self.rna_constants = RNA_CONSTANTS
        self.residue_names = self.rna_constants.RESD_NAMES  # ['A', 'G', 'U', 'C']
        self.atom_names_per_residue = self.rna_constants.ATOM_NAMES_PER_RESD
        self.max_atoms_per_residue = self.rna_constants.ATOM_NUM_MAX  # 23
    
    def parse_pdb_file(self, pdb_file: Union[str, Path]) -> Dict[int, Dict]:
        """解析PDB文件，提取所有原子信息"""
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB文件不存在: {pdb_file}")
        
        residues = {}
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    # 解析PDB ATOM行
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    res_num = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    # 创建残基条目
                    if res_num not in residues:
                        residues[res_num] = {
                            'res_name': res_name,
                            'atoms': {}
                        }
                    
                    # 添加原子坐标
                    residues[res_num]['atoms'][atom_name] = [x, y, z]
        
        if not residues:
            raise ValueError(f"PDB文件中未找到原子数据: {pdb_file}")
        
        return residues
    
    def get_sequence_from_pdb(self, residues: Dict[int, Dict]) -> str:
        """从解析的残基信息中提取序列"""
        sorted_residues = sorted(residues.keys())
        sequence = ""
        
        for res_num in sorted_residues:
            res_name = residues[res_num]['res_name']
            if res_name in self.residue_names:
                sequence += res_name
            else:
                logger.warning(f"未知的残基类型: {res_name} (位置 {res_num})")
                sequence += 'N'  # 用N表示未知残基
        
        return sequence
    
    def generate_residue_level_mask(
        self, 
        pdb_file: Union[str, Path],
        target_sequence: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        生成残基级别的缺失原子掩码
        
        参数:
            pdb_file: PDB文件路径
            target_sequence: 目标序列，如果提供则与PDB序列进行对齐
            
        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
                - coordinates: 坐标张量 [num_residues, max_atoms_per_residue, 3]
                - missing_mask: 缺失原子掩码 [num_residues, max_atoms_per_residue]
                - residue_mask: 残基掩码 [num_residues]，True表示有效残基
                - sequence: RNA序列字符串
        """
        # 解析PDB文件
        residues = self.parse_pdb_file(pdb_file)
        pdb_sequence = self.get_sequence_from_pdb(residues)
        
        # 使用目标序列或PDB序列
        sequence = target_sequence if target_sequence is not None else pdb_sequence
        num_residues = len(sequence)
        
        # 初始化输出张量 - 缺失原子的坐标默认填充为0
        coordinates = torch.zeros(num_residues, int(self.max_atoms_per_residue), 3, dtype=torch.float32)
        missing_mask = torch.ones(num_residues, int(self.max_atoms_per_residue), dtype=torch.bool)  # 默认全部缺失
        residue_mask = torch.zeros(num_residues, dtype=torch.bool)
        
        # 获取排序后的残基编号
        sorted_residue_nums = sorted(residues.keys())
        
        # 对齐PDB残基和目标序列
        pdb_idx = 0
        for seq_idx in range(num_residues):
            if pdb_idx >= len(sorted_residue_nums):
                break
                
            res_type = sequence[seq_idx]
            if res_type not in list(self.atom_names_per_residue.keys()):
                logger.warning(f"未知残基类型: {res_type}，跳过")
                continue
            
            # 找到匹配的PDB残基
            while pdb_idx < len(sorted_residue_nums):
                res_num = sorted_residue_nums[pdb_idx]
                res_info = residues[res_num]
                pdb_res_type = res_info['res_name']
                
                if pdb_res_type == res_type:
                    # 找到匹配的残基
                    residue_mask[seq_idx] = True
                    expected_atoms = list(self.atom_names_per_residue[res_type])
                    actual_atoms = res_info['atoms']
                    
                    # 检查每个预期的原子
                    for atom_idx, atom_name in enumerate(expected_atoms):
                        if atom_name in actual_atoms:
                            # 原子存在 - 填充真实坐标并标记为非缺失
                            coords = actual_atoms[atom_name]
                            coordinates[seq_idx, atom_idx] = torch.tensor(coords, dtype=torch.float32)
                            missing_mask[seq_idx, atom_idx] = False
                        # 如果原子不存在，坐标保持为0，missing_mask保持为True
                    
                    pdb_idx += 1
                    break
                else:
                    # 跳过不匹配的PDB残基
                    pdb_idx += 1
        
        return coordinates, missing_mask, residue_mask, sequence

class RNA3DDataset(Dataset):
    """
    用于RhoFold训练的RNA 3D结构数据集
    
    数据文件组织结构:
    RNA3D_DATA/
    ├── pdb/              # 所有PDB文件
    ├── seq/              # 所有序列文件
    ├── rMSA/             # 所有MSA文件
    └── list/             # 交叉验证分割文件
        ├── fold-X_train_ids
        └── valid_fold-X
    """
    
    def __init__(
        self,
        data_dir: str,
        fold: int = 0,
        split: str = "train",
        max_length: int = 512,
        use_msa: bool = True,
        cache_dir: Optional[str] = None,
        force_reload: bool = False,
        enable_missing_atom_mask: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据根目录路径 (RNA3D_DATA)
            fold: 交叉验证折数 (0-9)
            split: 数据集分割 ("train", "valid")
            max_length: 最大序列长度
            use_msa: 是否使用MSA信息
            cache_dir: 缓存目录，用于存储预处理的数据
            force_reload: 是否强制重新加载数据
            enable_missing_atom_mask: 是否启用缺失原子掩码生成
        """
        self.data_dir = Path(data_dir)
        self.fold = fold
        self.split = split
        self.max_length = max_length
        self.use_msa = use_msa
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.force_reload = force_reload
        self.enable_missing_atom_mask = enable_missing_atom_mask
        
        # 设置子目录路径
        self.pdb_dir = self.data_dir / "pdb"
        self.seq_dir = self.data_dir / "sequences"
        self.msa_dir = self.data_dir / "rMSA"
        self.list_dir = self.data_dir / "list"
        
        # 验证目录是否存在
        for dir_path in [self.pdb_dir, self.seq_dir, self.list_dir]:
            if not dir_path.exists():
                raise FileNotFoundError(f"必需的目录不存在: {dir_path}")
        
        # 创建缓存目录
        self.cache_dir.mkdir(exist_ok=True)
        
        # 初始化RNA转换器和缺失原子掩码生成器
        self.rna_converter = RNAConverter()
        if self.enable_missing_atom_mask:
            self.missing_atom_generator = MissingAtomMaskGenerator()
        
        # 加载数据
        self._load_data()
        
        logger.info(f"加载了 {len(self.samples)} 个样本 (fold: {fold}, split: {split})")
    
    def _load_split_ids(self) -> List[str]:
        """加载指定fold和split的样本ID列表"""
        if self.split == "train":
            split_file = self.list_dir / f"fold-{self.fold}_train_ids"
        elif self.split == "valid":
            split_file = self.list_dir / f"valid_fold-{self.fold}"
        else:
            raise ValueError(f"不支持的split类型: {self.split}. 支持的类型: train, valid")
        
        if not split_file.exists():
            raise FileNotFoundError(f"分割文件不存在: {split_file}")
        
        with open(split_file, 'r') as f:
            sample_ids = [line.strip() for line in f if line.strip()]
        
        logger.info(f"从 {split_file} 加载了 {len(sample_ids)} 个样本ID")
        return sample_ids
    
    def _load_data(self):
        """加载数据样本"""
        # 缓存文件路径
        cache_file = self.cache_dir / f"fold_{self.fold}_{self.split}_samples.pkl"
        
        if cache_file.exists() and not self.force_reload:
            logger.info(f"从缓存加载数据: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.samples = pickle.load(f)
            return
        
        logger.info(f"扫描数据目录: {self.data_dir}")
        
        # 加载指定fold和split的样本ID
        sample_ids = self._load_split_ids()
        self.samples = []
        
        # 统计文件存在情况
        found_pdb = found_seq = found_msa = 0
        
        for sample_id in sample_ids:
            # 查找对应的文件
            pdb_file = self.pdb_dir / f"{sample_id}.pdb"
            seq_file = self.seq_dir / f"{sample_id}.fasta"
            msa_file = self.msa_dir / f"{sample_id}.a3m"
            
            # 检查必需文件是否存在
            if not pdb_file.exists():
                logger.warning(f"缺少PDB文件: {pdb_file}")
                continue
            found_pdb += 1
            
            if not seq_file.exists():
                logger.warning(f"缺少序列文件: {seq_file}")
                continue
            found_seq += 1
            
            # MSA文件是可选的
            msa_file_path = None
            if msa_file.exists():
                msa_file_path = str(msa_file)
                found_msa += 1
            
            sample = {
                'name': sample_id,
                'pdb_file': str(pdb_file),
                'seq_file': str(seq_file),
                'msa_file': msa_file_path
            }
            
            self.samples.append(sample)
        
        logger.info(f"文件统计: PDB {found_pdb}, SEQ {found_seq}, MSA {found_msa}")
        
        # 过滤过长序列
        valid_samples = []
        for sample in self.samples:
            try:
                seq = self._load_sequence(sample['seq_file'])
                if len(seq) <= self.max_length:
                    valid_samples.append(sample)
                else:
                    logger.warning(f"序列过长，跳过: {sample['name']} (长度: {len(seq)})")
            except Exception as e:
                logger.warning(f"加载序列失败，跳过: {sample['name']}, 错误: {e}")
        
        self.samples = valid_samples
        
        # 缓存样本信息
        with open(cache_file, 'wb') as f:
            pickle.dump(self.samples, f)
        
        logger.info(f"扫描完成，找到 {len(self.samples)} 个有效样本")
    
    def _load_sequence(self, seq_file: str) -> str:
        """加载RNA序列"""
        with open(seq_file, 'r') as f:
            lines = f.readlines()
        
        # 处理FASTA格式
        sequence = ""
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                continue
            sequence += line.upper()
        
        # 验证RNA序列
        valid_chars = set('AUGC')
        sequence = ''.join(c for c in sequence if c in valid_chars)
        
        return sequence
    
    def _load_msa(self, msa_file: str) -> Optional[List[str]]:
        """加载MSA信息"""
        if not msa_file or not os.path.exists(msa_file):
            return None
        
        try:
            with open(msa_file, 'r') as f:
                lines = f.readlines()
            
            sequences = []
            current_seq = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq.upper())
                        current_seq = ""
                else:
                    current_seq += line
            
            if current_seq:
                sequences.append(current_seq.upper())
            
            return sequences
        except Exception as e:
            logger.warning(f"加载MSA文件失败: {msa_file}, 错误: {e}")
            return None
    
    def _parse_pdb_coordinates(self, pdb_file: str) -> torch.Tensor:
        """解析PDB文件中的原子坐标"""
        atoms_data = []
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    # 解析PDB ATOM行
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    res_num = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    atoms_data.append({
                        'atom_name': atom_name,
                        'res_name': res_name,
                        'res_num': res_num,
                        'coords': [x, y, z]
                    })
        
        if not atoms_data:
            raise ValueError(f"PDB文件中未找到原子数据: {pdb_file}")
        
        # 按残基号排序
        atoms_data.sort(key=lambda x: x['res_num'])
        
        # 提取坐标
        coords_list = []
        for atom in atoms_data:
            coords_list.append(atom['coords'])
        
        return torch.tensor(coords_list, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor, List[str], int, None]]:
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 使用RhoFold的get_features函数处理token
        try:
            # 创建临时的fasta文件路径（实际就是seq文件）
            fas_fpath = sample['seq_file']
            msa_fpath = sample['msa_file'] if (self.use_msa and sample['msa_file']) else None
            
            if msa_fpath and os.path.exists(msa_fpath):
                # 使用MSA进行特征提取
                features = get_features(fas_fpath, msa_fpath, msa_depth=64)
                tokens = features['tokens'].squeeze(0)  # 移除batch维度
                rna_fm_tokens = features['rna_fm_tokens'].squeeze(0)  # 移除batch维度
                sequence = features['seq']
            else:
                # 只有序列，没有MSA
                sequence = read_fas(fas_fpath)[0][1]
                
                # 创建一个只包含序列本身的"MSA"
                with open(fas_fpath, 'r') as f:
                    fas_content = f.read()
                
                # 使用单序列作为MSA
                features = get_features(fas_fpath, fas_fpath, msa_depth=1)
                tokens = features['tokens'].squeeze(0)
                rna_fm_tokens = features['rna_fm_tokens'].squeeze(0)
                
        except Exception as e:
            logger.warning(f"使用get_features失败: {e}，使用后备方案")
            # 后备方案：直接读取序列
            sequence = self._load_sequence(sample['seq_file'])
            tokens = torch.zeros(len(sequence) + 2, dtype=torch.long)  # 简单填充
            rna_fm_tokens = None
        
        # 处理坐标和缺失原子掩码
        if self.enable_missing_atom_mask:
            try:
                # 使用缺失原子掩码生成器
                residue_coordinates, residue_missing_mask, residue_mask, pdb_sequence = \
                    self.missing_atom_generator.generate_residue_level_mask(
                        sample['pdb_file'], target_sequence=sequence
                    )
                
                # 转换为平展格式
                # 提取所有预期原子的坐标和掩码，包括缺失的原子（用0坐标填充）
                flat_coordinates = []
                flat_missing_mask = []
                
                for res_idx in range(len(sequence)):
                    res_coords = residue_coordinates[res_idx]  # [max_atoms_per_residue, 3]
                    res_mask = residue_missing_mask[res_idx]   # [max_atoms_per_residue]
                    
                    # 获取该残基类型的预期原子数
                    res_type = sequence[res_idx]
                    if res_type in ['A', 'G', 'U', 'C']:
                        expected_atoms = len(self.missing_atom_generator.atom_names_per_residue[res_type])
                        # 取预期的原子数量，缺失原子已经用0坐标填充，mask为True
                        flat_coordinates.append(res_coords[:expected_atoms])
                        flat_missing_mask.append(res_mask[:expected_atoms])
            
                if flat_coordinates:
                    coordinates = torch.cat(flat_coordinates, dim=0)  # [N_atoms, 3]
                    missing_atom_mask = torch.cat(flat_missing_mask, dim=0)  # [N_atoms]
                else:
                    # 如果没有有效残基，创建空张量
                    coordinates = torch.zeros((0, 3), dtype=torch.float32)
                    missing_atom_mask = torch.zeros(0, dtype=torch.bool)
                
                result = {
                    'name': sample['name'],
                    'sequence': sequence,
                    'tokens': tokens,
                    'rna_fm_tokens': rna_fm_tokens,
                    'coordinates': coordinates,  # [N_atoms, 3] - 平展格式
                    'missing_atom_mask': missing_atom_mask,  # [N_atoms] - 平展格式
                    'seq_length': len(sequence),
                    'num_atoms': coordinates.shape[0]  # 添加原子数量信息
                }
                
            except Exception as e:
                logger.error(f"生成缺失原子掩码失败: {sample['pdb_file']}, 错误: {e}")
                # 返回空的坐标和掩码
                coordinates = torch.zeros((0, 3), dtype=torch.float32)
                missing_atom_mask = torch.zeros(0, dtype=torch.bool)
                
                result = {
                    'name': sample['name'],
                    'sequence': sequence,
                    'tokens': tokens,
                    'rna_fm_tokens': rna_fm_tokens,
                    'coordinates': coordinates,
                    'missing_atom_mask': missing_atom_mask,
                    'seq_length': len(sequence),
                    'num_atoms': 0
                }
        else:
            # 传统的坐标加载方式
            try:
                flat_coordinates = self._parse_pdb_coordinates(sample['pdb_file'])
            except Exception as e:
                logger.error(f"解析PDB文件失败: {sample['pdb_file']}, 错误: {e}")

            result = {
                'name': sample['name'],
                'sequence': sequence,
                'tokens': tokens,
                'rna_fm_tokens': rna_fm_tokens,
                'coordinates': flat_coordinates,
                'seq_length': len(sequence)
            }
        
        return result
    
    def get_sample_by_name(self, sample_name: str) -> Optional[Dict[str, Union[str, torch.Tensor, List[str], int, None]]]:
        """
        根据样本名称获取特定数据样本
        
        Args:
            sample_name: 样本名称，如 "3meiA"
            
        Returns:
            数据样本字典，如果未找到则返回None
        """
        # 查找匹配的样本
        target_sample = None
        for sample in self.samples:
            if sample['name'] == sample_name:
                target_sample = sample
                break
        
        if target_sample is None:
            logger.warning(f"未找到样本: {sample_name}")
            logger.info(f"可用样本列表: {[s['name'] for s in self.samples[:10]]}...")  # 显示前10个样本名称
            return None
        
        # 使用相同的逻辑处理这个样本（复用__getitem__的逻辑）
        idx = self.samples.index(target_sample)
        return self.__getitem__(idx)
    
    def list_sample_names(self, max_count: int = None) -> List[str]:
        """
        列出所有可用的样本名称
        
        Args:
            max_count: 最大返回数量，None表示返回所有
            
        Returns:
            样本名称列表
        """
        names = [sample['name'] for sample in self.samples]
        if max_count is not None:
            names = names[:max_count]
        return names



def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List, None]]:
    """
    自定义collate函数，用于处理不同长度的序列
    支持缺失原子掩码功能
    """
    batch_size = len(batch)
    
    # 收集基本信息
    names = [item['name'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    seq_lengths = [item['seq_length'] for item in batch]
    
    # 找到最大长度
    max_seq_len = max(seq_lengths)
    
    # 检查tokens维度并处理
    token_dims = [item['tokens'].dim() for item in batch]
    if all(dim == 1 for dim in token_dims):
        # 1维tokens: [seq_length]
        max_token_len = max(item['tokens'].size(0) for item in batch)
        padded_tokens = torch.zeros(batch_size, max_token_len, dtype=torch.long)
        for i, item in enumerate(batch):
            tokens = item['tokens']
            padded_tokens[i, :tokens.size(0)] = tokens
    elif all(dim == 2 for dim in token_dims):
        # 2维tokens: [msa_depth, seq_length]
        max_msa_depth = max(item['tokens'].size(0) for item in batch)
        max_token_len = max(item['tokens'].size(1) for item in batch)
        padded_tokens = torch.zeros(batch_size, max_msa_depth, max_token_len, dtype=torch.long)
        for i, item in enumerate(batch):
            tokens = item['tokens']
            padded_tokens[i, :tokens.size(0), :tokens.size(1)] = tokens
    else:
        raise ValueError(f"不一致的tokens维度: {token_dims}")
    
    # 处理rna_fm tokens
    rna_fm_tokens = None
    if any(item['rna_fm_tokens'] is not None for item in batch):
        # 检查rna_fm_tokens的维度
        rna_fm_dims = [item['rna_fm_tokens'].dim() if item['rna_fm_tokens'] is not None else 0 for item in batch]
        valid_dims = [dim for dim in rna_fm_dims if dim > 0]
        
        if len(valid_dims) > 0:
            if all(dim == 1 for dim in valid_dims):
                # 1维rna_fm_tokens: [seq_length]
                max_rna_fm_len = max(
                    item['rna_fm_tokens'].size(0) if item['rna_fm_tokens'] is not None else 0 
                    for item in batch
                )
                if max_rna_fm_len > 0:
                    rna_fm_tokens = torch.zeros(batch_size, max_rna_fm_len, dtype=torch.long)
                    for i, item in enumerate(batch):
                        if item['rna_fm_tokens'] is not None:
                            tokens = item['rna_fm_tokens']
                            rna_fm_tokens[i, :tokens.size(0)] = tokens
            elif all(dim == 2 for dim in valid_dims):
                # 2维rna_fm_tokens: [msa_depth, seq_length]
                max_msa_depth = max(
                    item['rna_fm_tokens'].size(0) if item['rna_fm_tokens'] is not None else 0 
                    for item in batch
                )
                max_msa_len = max(
                    item['rna_fm_tokens'].size(1) if item['rna_fm_tokens'] is not None else 0 
                    for item in batch
                )
                
                if max_msa_depth > 0 and max_msa_len > 0:
                    rna_fm_tokens = torch.zeros(batch_size, max_msa_depth, max_msa_len, dtype=torch.long)
                    for i, item in enumerate(batch):
                        if item['rna_fm_tokens'] is not None:
                            tokens = item['rna_fm_tokens']
                            rna_fm_tokens[i, :tokens.size(0), :tokens.size(1)] = tokens
    
    # 检查是否启用了缺失原子掩码功能
    has_missing_atom_mask = 'missing_atom_mask' in batch[0]
    
    if has_missing_atom_mask:
        # 处理平展格式的坐标和掩码
        # 坐标形状: [N_atoms, 3]，掩码形状: [N_atoms]
        max_atoms = max(item['num_atoms'] for item in batch)
        
        # 填充坐标张量和掩码
        padded_coords = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
        missing_atom_masks = torch.ones(batch_size, max_atoms, dtype=torch.bool)  # 默认为True（缺失）
        atom_masks = torch.zeros(batch_size, max_atoms, dtype=torch.bool)  # 原子存在掩码
        
        for i, item in enumerate(batch):
            coords = item['coordinates']  # [N_atoms, 3]
            missing_mask = item['missing_atom_mask']  # [N_atoms]
            num_atoms = item['num_atoms']
            
            if num_atoms > 0:
                padded_coords[i, :num_atoms] = coords
                missing_atom_masks[i, :num_atoms] = missing_mask
                atom_masks[i, :num_atoms] = True  # 标记这些位置有原子数据
        
        result = {
            'names': names,
            'sequences': sequences,
            'tokens': padded_tokens,    #[batch_size, msa_depth, seq_length]
            'rna_fm_tokens': rna_fm_tokens,  #[batch_size, seq_length]
            'coordinates': padded_coords,  # [batch_size, max_atoms, 3] - 平展格式
            'missing_atom_masks': missing_atom_masks,  # [batch_size, max_atoms] - 平展格式
            'atom_masks': atom_masks,  # [batch_size, max_atoms] - 原子存在掩码
            'seq_lengths': torch.tensor(seq_lengths, dtype=torch.long),
            'num_atoms': torch.tensor([item['num_atoms'] for item in batch], dtype=torch.long)
        }
    else:
        # 传统的平展坐标处理
        # 处理坐标 - 找到最大原子数
        max_atoms = max(item['coordinates'].size(0) for item in batch)
        padded_coords = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
        coord_masks = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
        
        for i, item in enumerate(batch):
            coords = item['coordinates']
            num_atoms = coords.size(0)
            padded_coords[i, :num_atoms] = coords
            coord_masks[i, :num_atoms] = True
        
        # 创建序列mask
        seq_masks = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        for i, seq_len in enumerate(seq_lengths):
            seq_masks[i, :seq_len] = True
        
        result = {
            'names': names,
            'sequences': sequences,
            'tokens': padded_tokens,
            'rna_fm_tokens': rna_fm_tokens,
            'coordinates': padded_coords,
            'coord_masks': coord_masks,
            'seq_masks': seq_masks,
            'seq_lengths': torch.tensor(seq_lengths, dtype=torch.long)
        }
    
    return result


class RNA3DDataLoader:
    """
    RNA3D数据加载器包装类，提供便捷的接口
    支持交叉验证和按文件类型分组的数据结构
    """
    
    def __init__(
        self,
        data_dir: str = "/home/wdcyx/rhofold/RNA3D_DATA",
        batch_size: int = 4,
        max_length: int = 512,
        use_msa: bool = True,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        force_reload: bool = False,
        enable_missing_atom_mask: bool = True
    ):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据根目录 (默认: /home/wdcyx/rhofold/RNA3D_DATA)
            batch_size: 批次大小
            max_length: 最大序列长度
            use_msa: 是否使用MSA
            num_workers: 数据加载进程数
            cache_dir: 缓存目录
            force_reload: 是否强制重新加载
            enable_missing_atom_mask: 是否启用缺失原子掩码生成
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_msa = use_msa
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.force_reload = force_reload
        self.enable_missing_atom_mask = enable_missing_atom_mask
        
    def get_dataloader(
        self, 
        fold: int = 0, 
        split: str = "train", 
        shuffle: bool = True,
        sampler=None
    ) -> DataLoader:
        """
        获取指定fold和split的数据加载器
        
        Args:
            fold: 交叉验证折数 (0-9)
            split: 数据分割 ("train", "valid")
            shuffle: 是否随机打乱
            sampler: 可选的采样器
            
        Returns:
            PyTorch DataLoader对象
        """
        dataset = RNA3DDataset(
            data_dir=self.data_dir,
            fold=fold,
            split=split,
            max_length=self.max_length,
            use_msa=self.use_msa,
            cache_dir=self.cache_dir,
            force_reload=self.force_reload,
            enable_missing_atom_mask=self.enable_missing_atom_mask
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def get_train_dataloader(self, fold: int = 0, sampler=None) -> DataLoader:
        """获取训练数据加载器"""
        return self.get_dataloader(fold=fold, split="train", shuffle=True, sampler=sampler)
    
    def get_valid_dataloader(self, fold: int = 0, sampler=None) -> DataLoader:
        """获取验证数据加载器"""
        return self.get_dataloader(fold=fold, split="valid", shuffle=False, sampler=sampler)
    
    def get_all_folds_dataloaders(self) -> Dict[int, Dict[str, DataLoader]]:
        """
        获取所有折数的数据加载器
        
        Returns:
            Dict[int, Dict[str, DataLoader]]: 例如 {0: {'train': DataLoader, 'valid': DataLoader}}
        """
        all_loaders = {}
        for fold in range(10):  # 假设有10个折数
            try:
                train_loader = self.get_train_dataloader(fold)
                valid_loader = self.get_valid_dataloader(fold)
                all_loaders[fold] = {'train': train_loader, 'valid': valid_loader}
            except Exception as e:
                logger.warning(f"跳过Fold {fold}: {e}")
                continue
        return all_loaders


def create_data_loaders(
    data_dir: str,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 4,
    fold: int = 0,
    use_msa: bool = True,
    use_all_folds: bool = False,
    cache_dir: Optional[str] = None,
    force_reload: bool = False,
    world_size: int = 1,
    local_rank: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    便利函数：一次性创建训练和验证数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        max_length: 最大序列长度
        num_workers: 数据加载进程数
        fold: 交叉验证折数
        use_msa: 是否使用MSA
        use_all_folds: 是否使用所有折数的数据
        cache_dir: 缓存目录
        force_reload: 是否强制重新加载数据
        
    Returns:
        (train_loader, valid_loader): 训练和验证数据加载器
    """
    logger.info(f"创建数据加载器 - fold: {fold}, batch_size: {batch_size}, max_length: {max_length}")
    
    if use_all_folds:
        # 使用所有折数的数据
        logger.info("使用所有折数的训练数据...")
        
        data_loader = RNA3DDataLoader(
            data_dir=data_dir,
            batch_size=batch_size,
            max_length=max_length,
            use_msa=use_msa,
            num_workers=num_workers,
            cache_dir=cache_dir,
            force_reload=force_reload,
            enable_missing_atom_mask=True
        )
        
        all_loaders = data_loader.get_all_folds_dataloaders()
        
        # 合并所有折数的训练数据加载器
        from torch.utils.data import ConcatDataset
        
        all_train_datasets = []
        all_valid_datasets = []
        
        for fold_num, loaders in all_loaders.items():
            train_dataset = loaders['train'].dataset
            valid_dataset = loaders['valid'].dataset
            all_train_datasets.append(train_dataset)
            all_valid_datasets.append(valid_dataset)
            
            # 安全获取数据集大小
            if hasattr(train_dataset, '__len__') and hasattr(valid_dataset, '__len__'):
                train_size = len(train_dataset)
                valid_size = len(valid_dataset)
                logger.info(f"Fold {fold_num}: 训练样本 {train_size}, 验证样本 {valid_size}")
            else:
                logger.info(f"Fold {fold_num}: 已添加到合并数据集")
        
        # 合并数据集
        combined_train_dataset = ConcatDataset(all_train_datasets)
        combined_valid_dataset = ConcatDataset(all_valid_datasets)
        
        # 创建新的数据加载器
        from torch.utils.data import DataLoader
        if world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(combined_train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=True)
            valid_sampler = DistributedSampler(combined_valid_dataset, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=False)
        else:
            train_sampler = None
            valid_sampler = None
        train_loader = DataLoader(
            combined_train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        valid_loader = DataLoader(
            combined_valid_dataset,
            batch_size=batch_size,
            shuffle=(valid_sampler is None),
            sampler=valid_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        logger.info(f"合并后训练样本数: {len(combined_train_dataset)}")
        logger.info(f"合并后验证样本数: {len(combined_valid_dataset)}")
        
    else:
        # 使用单个折数的数据
        data_loader = RNA3DDataLoader(
            data_dir=data_dir,
            batch_size=batch_size,
            max_length=max_length,
            use_msa=use_msa,
            num_workers=num_workers,
            cache_dir=cache_dir,
            force_reload=force_reload,
            enable_missing_atom_mask=True
        )
        
        from torch.utils.data import DataLoader
        if world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            train_dataset = data_loader.get_train_dataloader(fold=fold).dataset
            valid_dataset = data_loader.get_valid_dataloader(fold=fold).dataset
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=True)
            valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=False)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=train_sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=valid_sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
        else:
            train_loader = data_loader.get_train_dataloader(fold=fold)
            valid_loader = data_loader.get_valid_dataloader(fold=fold)
        
        # 安全获取数据集大小
        if hasattr(train_loader.dataset, '__len__'):
            train_size = len(train_loader.dataset)
            logger.info(f"Fold {fold} - 训练样本数: {train_size}")
        
        if hasattr(valid_loader.dataset, '__len__'):
            valid_size = len(valid_loader.dataset)
            logger.info(f"Fold {fold} - 验证样本数: {valid_size}")
    
    return train_loader, valid_loader


# 使用示例和测试函数
def test_missing_atom_filling():
    """测试缺失原子填充功能"""
    logger.info("测试缺失原子填充功能...")
    
    # 创建数据加载器
    data_loader = RNA3DDataLoader(
        data_dir="/home/yamanashi/diffold/processed_data",
        batch_size=1,
        max_length=512,
        use_msa=False,
        num_workers=0,
        force_reload=False,
        enable_missing_atom_mask=True
    )
    
    try:
        train_loader = data_loader.get_train_dataloader(fold=0)
        logger.info(f"训练样本数: {len(train_loader.dataset)}")
        
        # 获取一个样本进行详细检查
        for batch_idx, batch in enumerate(train_loader):
            sample_name = batch['names'][0]
            sequence = batch['sequences'][0]
            coordinates = batch['coordinates'][0]  # [max_atoms, 3]
            missing_mask = batch['missing_atom_masks'][0]  # [max_atoms]
            atom_mask = batch['atom_masks'][0]  # [max_atoms]
            num_atoms = batch['num_atoms'][0].item()
            
            logger.info(f"\n=== 样本详细信息: {sample_name} ===")
            logger.info(f"序列: {sequence}")
            logger.info(f"序列长度: {len(sequence)}")
            logger.info(f"总原子数: {num_atoms}")
            logger.info(f"坐标张量形状: {coordinates.shape}")
            logger.info(f"缺失原子掩码形状: {missing_mask.shape}")
            
            # 检查每个残基的原子情况
            atom_idx = 0
            for res_idx, res_type in enumerate(sequence):
                if res_type in ['A', 'G', 'U', 'C']:
                    expected_atoms = len(train_loader.dataset.missing_atom_generator.atom_names_per_residue[res_type])
                    
                    # 提取该残基的原子坐标和掩码
                    res_coords = coordinates[atom_idx:atom_idx + expected_atoms]  # [expected_atoms, 3]
                    res_missing = missing_mask[atom_idx:atom_idx + expected_atoms]  # [expected_atoms]
                    res_atom_mask = atom_mask[atom_idx:atom_idx + expected_atoms]  # [expected_atoms]
                    
                    # 统计该残基的原子情况
                    present_atoms = (~res_missing & res_atom_mask).sum().item()
                    missing_atoms = (res_missing & res_atom_mask).sum().item()
                    zero_coords = (res_coords.abs().sum(dim=1) == 0).sum().item()
                    
                    if missing_atoms > 0:  # 只显示有缺失原子的残基
                        logger.info(f"  残基 {res_idx+1} ({res_type}): 预期 {expected_atoms} 个原子")
                        logger.info(f"    - 存在原子: {present_atoms}")
                        logger.info(f"    - 缺失原子: {missing_atoms}")
                        logger.info(f"    - 零坐标原子: {zero_coords}")
                        logger.info(f"    - 缺失原子掩码: {res_missing.tolist()}")
                    
                    atom_idx += expected_atoms
            
            # 只检查第一个样本
            break
            
        logger.info("✓ 缺失原子填充功能测试完成!")
        
    except Exception as e:
        logger.error(f"✗ 缺失原子填充功能测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_dataloader():
    """测试数据加载器功能"""
    logger.info("测试RNA3D数据加载器...")
    
    # 创建数据加载器
    data_loader = RNA3DDataLoader(
        data_dir="/home/wdcyx/rhofold/RNA3D_DATA",
        batch_size=2,
        max_length=512,
        use_msa=True,
        num_workers=0,  # 测试时使用0避免多进程问题
        force_reload=False
    )
    
    try:
        # 测试单个fold
        logger.info("=== 测试单个fold ===")
        train_loader = data_loader.get_train_dataloader(fold=0)
        valid_loader = data_loader.get_valid_dataloader(fold=0)
        
        logger.info(f"Fold 0 - 训练样本数: {len(train_loader.dataset)}")
        logger.info(f"Fold 0 - 验证样本数: {len(valid_loader.dataset)}")
        
        # 测试一个batch
        for batch_idx, batch in enumerate(train_loader):
            logger.info(f"\nBatch {batch_idx + 1}:")
            logger.info(f"  - 样本名称: {batch['names']}")
            logger.info(f"  - 序列长度: {batch['seq_lengths'].tolist()}")
            logger.info(f"  - Tokens形状: {batch['tokens'].shape}")
            if batch['rna_fm_tokens'] is not None:
                logger.info(f"  - rMSA Tokens形状: {batch['rna_fm_tokens'].shape}")
            logger.info(f"  - 坐标形状: {batch['coordinates'].shape}")
            
            # 检查是否有缺失原子掩码功能
            if 'missing_atom_masks' in batch:
                logger.info(f"  - 缺失原子掩码形状: {batch['missing_atom_masks'].shape}")
                logger.info(f"  - 原子掩码形状: {batch['atom_masks'].shape}")
                logger.info(f"  - 总原子数: {batch['num_atoms'].tolist()}")
            else:
                logger.info(f"  - 坐标mask形状: {batch['coord_masks'].shape}")
            
            # 只测试第一个batch
            break
        
        # 测试所有fold的统计信息
        logger.info("\n=== 测试所有fold统计信息 ===")
        all_loaders = data_loader.get_all_folds_dataloaders()
        
        total_train = total_valid = 0
        for fold, loaders in all_loaders.items():
            train_size = len(loaders['train'].dataset)
            valid_size = len(loaders['valid'].dataset)
            total_train += train_size
            total_valid += valid_size
        logger.info(f"\n总计: 训练样本 {total_train}, 验证样本 {total_valid}")
        logger.info(f"可用fold数: {len(all_loaders)}")
        logger.info("\n✓ 数据加载器测试成功!")
        
    except Exception as e:
        logger.error(f"✗ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 测试基本数据加载器功能
    test_dataloader()
    
    print("\n" + "="*50 + "\n")
    
    # 测试缺失原子填充功能
    test_missing_atom_filling()
