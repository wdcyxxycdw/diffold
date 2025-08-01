# Diffold推理脚本使用说明

这个脚本提供了完整的Diffold模型推理流程，包括坐标预测、PDB文件生成和结构优化。

## 功能特性

- ✅ **完整的推理流程** - 从FASTA序列到优化后的PDB结构
- ✅ **PDB文件生成** - 自动将Diffold预测坐标转换为标准PDB格式
- ✅ **结构优化** - 集成Amber relaxation进行结构优化
- ✅ **详细日志** - 完整的推理过程记录
- ✅ **多种输出格式** - PDB、NPZ、Numpy等多种格式
- ✅ **错误处理** - 优雅的错误处理和恢复机制

## 使用方法

### 基本用法

```bash
python inference_diffold.py \
    --input_fas ./example/input/3owzA/3owzA.fasta \
    --output_dir ./example/output/diffold_3owzA \
    --checkpoint_path ./checkpoints/diffold_best.pt \
    --relax_steps 1000
```

### 使用配置文件

```bash
python inference_diffold.py \
    --config_file config_diffold_inference.yaml \
    --input_fas ./your_sequence.fasta \
    --output_dir ./your_output
```

### 跳过优化

```bash
python inference_diffold.py \
    --input_fas ./example/input/3owzA/3owzA.fasta \
    --output_dir ./example/output/diffold_3owzA \
    --relax_steps 0  # 设置为0跳过优化
```

## 参数说明

### 基本参数

- `--device`: 设备类型 (默认自动检测)
  - `cuda:0`: 使用GPU 0
  - `cpu`: 使用CPU
  - 不指定: 自动检测

- `--checkpoint_path`: Diffold模型检查点路径
- `--rhofold_checkpoint`: RhoFold预训练模型路径

### 输入输出

- `--input_fas`: 输入FASTA文件路径
- `--input_a3m`: 输入MSA文件路径 (可选)
- `--output_dir`: 输出目录路径

### 优化参数

- `--relax_steps`: Amber优化步数 (默认1000)
  - `0`: 跳过优化
  - `>0`: 执行指定步数的优化

### 配置

- `--config_file`: 配置文件路径 (YAML格式)

## 输出文件

推理完成后，输出目录将包含以下文件：

### 主要输出

1. **`diffold_unrelaxed_model.pdb`** - 未优化的PDB结构文件
2. **`diffold_relaxed_1000_model.pdb`** - 优化后的PDB结构文件 (如果执行了优化)
3. **`diffold_coordinates.npz`** - 预测坐标数据 (NumPy压缩格式)
4. **`diffold_confidence.npy`** - 置信度数据 (如果模型输出包含)

### 日志文件

- **`diffold_inference.log`** - 详细的推理日志

## 文件格式说明

### PDB文件

生成的PDB文件包含标准的ATOM记录，可以直接用PyMOL、VMD等软件查看：

```
REMARK 250 MODEL: DIFFOLD_PREDICTION
REMARK 250 SEQUENCE: AUGC
REMARK 250 CHAIN: A
REMARK 250 ATOMS: 84
ATOM      1  C4'  A   A   1      12.345  23.456  34.567  1.00 50.00           C
ATOM      2  C1'  A   A   1      13.456  24.567  35.678  1.00 50.00           C
...
END
```

### NPZ文件

`diffold_coordinates.npz` 包含以下数据：

```python
import numpy as np

data = np.load('diffold_coordinates.npz')
predicted_coords = data['predicted_coords']  # 预测坐标
atom_mask = data['atom_mask']                # 原子掩码
sequence = data['sequence']                   # 序列
validation = data['validation']              # 验证结果
```

## 配置文件示例

创建 `config_diffold_inference.yaml`:

```yaml
# 基本参数
device: "cuda:0"
checkpoint_path: "./checkpoints/diffold_best.pt"
rhofold_checkpoint: "./pretrained/model_20221010_params.pt"

# 输入输出
input_fas: "./example/input/3owzA/3owzA.fasta"
input_a3m: null
output_dir: "./example/output/diffold_3owzA"

# 优化参数
relax_steps: 1000

# Diffold模型配置
model_config:
  diffusion_steps: 1000
  noise_schedule: "cosine"
  loss_diffusion_weight: 1.0
  loss_confidence_weight: 0.1
  loss_distogram_weight: 0.1
  num_sample_steps: 32
  nucleotide_loss_weight: 1.0
  ligand_loss_weight: 1.0
```

## 与训练脚本集成

在你的训练代码中，可以在每个epoch后调用推理：

```python
# 在训练循环中
if epoch % save_every == 0:
    # 保存检查点
    checkpoint_path = f"./checkpoints/diffold_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    # 运行推理
    import subprocess
    subprocess.run([
        'python', 'inference_diffold.py',
        '--input_fas', './example/input/test.fasta',
        '--output_dir', f'./outputs/epoch_{epoch}',
        '--checkpoint_path', checkpoint_path,
        '--relax_steps', '500'
    ])
```

## 错误处理

脚本包含完整的错误处理机制：

- ✅ **输入验证** - 检查FASTA文件格式和序列有效性
- ✅ **模型加载** - 优雅处理检查点加载失败
- ✅ **推理错误** - 捕获并记录推理过程中的错误
- ✅ **PDB生成** - 处理坐标转换和文件写入错误
- ✅ **优化错误** - 如果Amber优化失败，继续执行但不中断流程

## 性能优化建议

1. **GPU使用**: 如果有GPU，设置 `--device cuda:0` 可显著加速推理
2. **批次处理**: 对于多个序列，可以修改脚本支持批次处理
3. **内存管理**: 对于长序列，可能需要调整批次大小
4. **优化步数**: 根据需求调整 `relax_steps`，更多步数=更好质量但更慢

## 依赖项

- Python 3.7+
- PyTorch
- NumPy
- tqdm
- PyYAML (如果使用配置文件)
- OpenMM (用于Amber优化)

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 解决方案: 使用 `--device cpu` 或减少批次大小

2. **Amber优化失败**
   - 解决方案: 设置 `--relax_steps 0` 跳过优化

3. **PDB文件生成失败**
   - 检查输出目录权限
   - 验证输入序列格式

4. **模型加载失败**
   - 检查检查点文件路径
   - 确认模型架构匹配

### 日志分析

查看 `diffold_inference.log` 文件获取详细的错误信息：

```bash
tail -f ./example/output/diffold_3owzA/diffold_inference.log
``` 