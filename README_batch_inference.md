# 批量推理和指标计算

这个脚本用于对验证集样本进行批量推理，计算结构预测指标并输出到文件。

## 功能特点

- 🚀 **批量处理**: 自动处理验证集中的所有样本
- 📊 **多指标计算**: 计算RMSD、TM-score、lDDT、Clash Score等指标
- 💾 **多格式输出**: 支持JSON、CSV和文本报告格式
- 📈 **统计报告**: 自动生成详细的统计报告
- 🔍 **错误处理**: 完善的错误处理和日志记录
- ⚡ **进度监控**: 实时显示处理进度和预估时间

## 使用方法

### 1. 基本用法

```bash
python batch_inference_metrics.py \
    --checkpoint_path ./checkpoints/best_model.pt \
    --data_dir ./processed_data \
    --output_dir ./batch_inference_output \
    --fold 3
```

### 2. 使用示例脚本

```bash
# 给脚本添加执行权限
chmod +x run_batch_inference.sh

# 运行批量推理
./run_batch_inference.sh
```

### 3. 参数说明

#### 必需参数
- `--checkpoint_path`: 模型检查点路径

#### 可选参数
- `--data_dir`: 数据目录路径 (默认: `./processed_data`)
- `--output_dir`: 输出目录路径 (默认: `./batch_inference_output`)
- `--fold`: 验证集折数 (默认: 3)
- `--rhofold_checkpoint`: RhoFold检查点路径 (默认: `./pretrained/model_20221010_params.pt`)
- `--max_sequence_length`: 最大序列长度 (默认: 256)
- `--num_workers`: 数据加载器工作进程数 (默认: 4)
- `--use_msa`: 是否使用MSA (默认: True)
- `--device`: 计算设备 (默认: auto)
- `--log_level`: 日志级别 (默认: INFO)
- `--max_samples`: 最大处理样本数，用于测试 (默认: None)

## 输出文件

脚本会在输出目录中生成以下文件：

### 1. 结果文件
- `batch_inference_results.json`: JSON格式的详细结果
- `batch_inference_results.csv`: CSV格式的结果表格
- `detailed_metrics.json`: 每个样本的详细指标数据

### 2. 结构文件
- `pdb_files/`: 包含所有预测的PDB结构文件
  - `{sample_name}_predicted.pdb`: 每个样本的预测结构

### 3. 报告文件
- `batch_inference_report.txt`: 统计报告
- `batch_inference.log`: 详细日志

## 输出格式

### JSON格式示例
```json
[
  {
    "sample_name": "6vrd_B",
    "status": "success",
    "sequence_length": 76,
    "sequence": "AUGCUAUGCUAUGCUAUGCUA...",
    "predicted_coords_shape": [1, 1520, 3],
    "target_coords_shape": [1, 1520, 3],
    "pdb_file_path": "./batch_inference_output/pdb_files/6vrd_B_predicted.pdb",
    "avg_rmsd": 2.341,
    "avg_tm_score": 0.678,
    "avg_lddt": 73.2,
    "avg_clash_score": 3.2,
    "detailed_metrics": {
      "rmsd_values": [2.341],
      "tm_scores": [0.678],
      "lddt_scores": [73.2],
      "clash_scores": [3.2]
    }
  }
]
```

### CSV格式示例
```csv
sample_name,status,sequence_length,avg_rmsd,avg_tm_score,avg_lddt,avg_clash_score,pdb_file_path
6vrd_B,success,76,2.341,0.678,73.2,3.2,./batch_inference_output/pdb_files/6vrd_B_predicted.pdb
```

### 报告格式示例
```
批量推理指标计算报告
==================================================

总样本数: 1170
成功样本数: 1150
失败样本数: 20
成功率: 98.29%

指标统计:
------------------------------
avg_rmsd:
  平均值: 2.3412
  中位数: 2.1234
  标准差: 0.5678
  最小值: 0.1234
  最大值: 5.6789

avg_tm_score:
  平均值: 0.6789
  中位数: 0.7123
  标准差: 0.1234
  最小值: 0.2345
  最大值: 0.9876
```

## 新增功能

### 1. PDB文件保存
- 每个样本的预测结构都会保存为PDB文件
- 文件命名格式: `{sample_name}_predicted.pdb`
- 保存位置: `{output_dir}/pdb_files/`

### 2. 详细指标记录
- 每个样本的原始指标值都会记录
- 保存在 `detailed_metrics.json` 文件中
- 包含RMSD、TM-score、lDDT、Clash Score的原始值

### 3. 序列信息记录
- 每个样本的RNA序列都会记录
- 便于后续分析和验证

## 计算的指标

### 1. RMSD (Root Mean Square Deviation)
- **单位**: Å
- **范围**: [0, +∞)
- **说明**: 基础几何偏差测量

### 2. TM-score (Template Modeling Score)
- **单位**: 无量纲
- **范围**: (0, 1]
- **说明**: RNA结构全局相似性

### 3. lDDT (local Distance Difference Test)
- **单位**: 0-100分
- **范围**: [0, 100]
- **说明**: 局部结构质量评估

### 4. Clash Score
- **单位**: 百分比
- **范围**: [0, 100]
- **说明**: 物理合理性检查

## 使用建议

### 1. 测试运行
首次使用时，建议先用少量样本测试：

```bash
python batch_inference_metrics.py \
    --checkpoint_path ./checkpoints/best_model.pt \
    --max_samples 10  # 只处理10个样本进行测试
```

### 2. 正式运行
测试无误后，移除`--max_samples`参数进行完整运行：

```bash
python batch_inference_metrics.py \
    --checkpoint_path ./checkpoints/best_model.pt \
    --data_dir ./processed_data \
    --output_dir ./batch_inference_output \
    --fold 3
```

### 3. 性能优化
- 使用GPU加速：`--device cuda`
- 调整工作进程数：`--num_workers 8`
- 使用混合精度：在脚本中启用`torch.cuda.amp`

## 故障排除

### 1. 内存不足
- 减少`--num_workers`
- 使用`--max_samples`分批处理

### 2. 模型加载失败
- 检查检查点路径是否正确
- 确认检查点文件完整性

### 3. 数据加载失败
- 检查数据目录结构
- 确认验证集列表文件存在

### 4. 指标计算错误
- 检查坐标格式是否正确
- 查看详细日志文件

## 注意事项

1. **数据格式**: 确保processed_data目录结构正确
2. **模型兼容性**: 确保检查点与当前代码版本兼容
3. **内存使用**: 大批量处理时注意内存使用情况
4. **时间估算**: 根据样本数量和硬件配置估算处理时间

## 扩展功能

可以根据需要扩展以下功能：

1. **多GPU支持**: 添加分布式处理
2. **结果可视化**: 添加图表生成功能
3. **指标筛选**: 添加指标阈值筛选
4. **结果分析**: 添加更详细的统计分析 