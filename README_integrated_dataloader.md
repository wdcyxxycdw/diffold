# 集成缺失原子掩码功能的DataLoader

我已经成功将缺失原子掩码生成功能整合到现有的RNA3DDataset和RNA3DDataLoader中，以支持批次处理。

## 🎯 整合内容

### 1. MissingAtomMaskGenerator类
已集成到 `diffold/dataloader.py` 中，提供：
- PDB文件解析功能
- 缺失原子检测
- 残基级别坐标和掩码生成

### 2. 增强的RNA3DDataset
- 新增 `enable_missing_atom_mask` 参数（默认True）
- 支持残基级别的坐标输出格式
- 自动生成缺失原子掩码和残基掩码
- 保持向后兼容性

### 3. 改进的collate_fn
- 自动检测数据格式（传统 vs 缺失原子掩码）
- 支持残基级别的批次填充
- 生成统一的批次掩码

## 🚀 使用方法

### 启用缺失原子掩码功能

```python
from diffold.dataloader import RNA3DDataLoader

# 创建支持缺失原子掩码的dataloader
dataloader = RNA3DDataLoader(
    data_dir="/path/to/RNA3D_DATA",
    batch_size=4,
    enable_missing_atom_mask=True  # 启用缺失原子掩码
)

train_loader = dataloader.get_train_dataloader(fold=0)

# 在训练循环中使用
for batch in train_loader:
    coords = batch['coordinates']          # [B, R, A, 3] - 残基级坐标
    missing_masks = batch['missing_atom_masks']  # [B, R, A] - 缺失原子掩码
    residue_masks = batch['residue_masks']      # [B, R] - 有效残基掩码
    
    # 使用掩码计算损失
    loss = compute_masked_loss(pred_coords, coords, missing_masks)
```

### 传统模式（向后兼容）

```python
# 禁用缺失原子掩码，使用传统格式
dataloader = RNA3DDataLoader(
    data_dir="/path/to/RNA3D_DATA",
    enable_missing_atom_mask=False
)

# 返回平展格式的坐标 [B, N_atoms, 3]
```

## 📊 数据格式

### 启用缺失原子掩码时

**单个样本:**
```python
sample = {
    'name': str,                    # 样本名称
    'sequence': str,                # RNA序列
    'coordinates': torch.Tensor,    # [num_residues, max_atoms_per_residue, 3]
    'missing_atom_mask': torch.Tensor,  # [num_residues, max_atoms_per_residue]
    'residue_mask': torch.Tensor,   # [num_residues]
    'tokens': torch.Tensor,         # 序列tokens
    'seq_length': int              # 序列长度
}
```

**批次数据:**
```python
batch = {
    'names': List[str],
    'sequences': List[str],
    'coordinates': torch.Tensor,        # [B, max_residues, max_atoms_per_residue, 3]
    'missing_atom_masks': torch.Tensor, # [B, max_residues, max_atoms_per_residue]
    'residue_masks': torch.Tensor,      # [B, max_residues]
    'tokens': torch.Tensor,
    'seq_lengths': torch.Tensor
}
```

### 传统模式时

**批次数据:**
```python
batch = {
    'names': List[str],
    'coordinates': torch.Tensor,    # [B, max_atoms, 3] - 平展格式
    'coord_masks': torch.Tensor,    # [B, max_atoms] - 填充掩码
    'seq_masks': torch.Tensor,      # [B, max_seq_len] - 序列掩码
    # ... 其他字段
}
```

## 💡 使用示例

### 损失函数实现

```python
def compute_masked_loss(pred_coords, target_coords, missing_masks):
    """
    计算带掩码的损失函数
    
    Args:
        pred_coords: [B, R, A, 3] 预测坐标
        target_coords: [B, R, A, 3] 目标坐标
        missing_masks: [B, R, A] 缺失原子掩码 (True=缺失)
    """
    mse_loss = nn.MSELoss(reduction='none')(pred_coords, target_coords)
    
    # 只计算存在原子的损失
    valid_atom_mask = ~missing_masks  # False -> True (存在)
    masked_loss = mse_loss * valid_atom_mask.unsqueeze(-1).float()
    
    # 计算平均损失
    total_valid = valid_atom_mask.sum()
    return masked_loss.sum() / total_valid if total_valid > 0 else torch.tensor(0.0)
```

### 提取有效原子坐标

```python
def extract_valid_coords(coords, missing_masks):
    """
    提取所有存在的原子坐标
    
    Args:
        coords: [B, R, A, 3] 坐标张量
        missing_masks: [B, R, A] 缺失原子掩码
    
    Returns:
        List[Tensor]: 每个样本的有效原子坐标 [N_valid, 3]
    """
    valid_coords_list = []
    for i in range(coords.shape[0]):
        sample_coords = coords[i]  # [R, A, 3]
        sample_mask = ~missing_masks[i]  # [R, A]
        sample_valid_coords = sample_coords[sample_mask]  # [N_valid, 3]
        valid_coords_list.append(sample_valid_coords)
    
    return valid_coords_list
```

## ⚙️ 配置参数

### RNA3DDataset参数

- `enable_missing_atom_mask: bool = True` - 是否启用缺失原子掩码功能
- 其他参数保持不变

### RNA3DDataLoader参数

- `enable_missing_atom_mask: bool = True` - 传递给数据集的参数
- 其他参数保持不变

## 🔍 技术细节

### 原子配置
- 基于RhoFold的RNA_CONSTANTS
- 支持A, G, U, C四种RNA残基
- 每个残基最多23个原子

### 序列对齐
- 支持PDB序列与目标序列的自动对齐
- 处理序列不匹配的情况
- 保持残基级别的对应关系

### 批次处理
- 自动填充到批次中的最大长度
- 生成对应的掩码信息
- 支持不同长度序列的高效处理

## 📋 测试

运行测试脚本验证功能：

```bash
python test_integrated_dataloader.py
```

测试内容包括：
- 单个样本加载测试
- 批次处理测试
- 掩码形状验证
- 神经网络兼容性测试
- 损失计算示例

## 🔄 向后兼容性

- 默认启用缺失原子掩码功能
- 可通过`enable_missing_atom_mask=False`使用传统模式
- 现有代码无需修改即可获得新功能
- collate_fn自动检测数据格式

## 🎁 优势

1. **自动缺失检测**: 无需手动标记缺失原子
2. **批次优化**: 支持高效的批次处理
3. **内存优化**: 残基级别的数据组织
4. **训练友好**: 直接支持掩码损失计算
5. **灵活使用**: 支持传统和新格式切换 