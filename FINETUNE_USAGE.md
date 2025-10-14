# 🎯 Diffold Fine-tuning 使用指南

## 📋 更新说明

已修复 fine-tuning 与增强优化器的兼容性问题。现在 `AdaptiveOptimizer` 同时支持：
- **标准训练模式**: 传入 `model` 参数
- **微调模式**: 传入 `param_groups` 参数（分层学习率）

---

## ✅ 修复内容

### 1. `diffold/advanced_optimizers.py`
- ✅ `AdaptiveOptimizer` 支持参数组模式
- ✅ 新增 `_create_optimizer_from_param_groups` 方法
- ✅ 自动打印各参数组的学习率配置
- ✅ 兼容标准训练和微调两种模式

### 2. `train.py`
- ✅ `setup_optimizer_and_scheduler` 优先获取参数组
- ✅ 增强优化器根据模式自动选择
- ✅ 完全向后兼容，不影响现有训练流程

---

## 🚀 使用方法

### 场景1: 小数据集微调（冻结RhoFold）

```bash
python train.py \
  --config config.yaml \
  --finetune \
  --pretrained_checkpoint ./checkpoints/best_model.pt \
  --freeze_strategy rhofold_only \
  --learning_rate 5e-5 \
  --epochs 20
```

**预期日志输出**：
```
🎯 启用微调模式
🎯 微调策略: rhofold_only
🔒 已冻结RhoFold所有模块，只训练扩散部分
🎯 使用高级优化器
🎯 使用微调模式（6个参数组）
📊 参数组学习率配置:
  rhofold_backbone: LR=5.00e-06, 参数=8,234,567, WD=1.00e-05
  diffusion_module: LR=5.00e-05, 参数=2,345,678, WD=1.00e-05
  confidence_head: LR=5.00e-05, 参数=123,456, WD=1.00e-05
  ...
```

### 场景2: 中等数据集，渐进式解冻

**配置文件** (`config.yaml`):
```yaml
finetune:
  enable_finetuning: true
  pretrained_checkpoint: "./checkpoints/best_model.pt"
  freeze_strategy: "rhofold_backbone"
  gradual_unfreeze:
    enable: true
    unfreeze_every: 5
    unfreeze_order: "top_down"
  learning_rate_scaling:
    enable: true
    backbone_lr_ratio: 0.1
    head_lr_ratio: 1.0

training:
  num_epochs: 30
  learning_rate: 1e-4
  validate_every: 1  # ⚠️ 微调建议每轮验证
  early_stopping_patience: 8  # ⚠️ 更严格的early stopping

enhanced_features:
  enable_enhanced_training: true
  optimizer:
    use_advanced_optimizer: true  # ✅ 现在支持微调！
    optimizer_name: "adamw"
    scheduler_type: "warmup_cosine"
```

**运行**:
```bash
python train.py --config config.yaml
```

### 场景3: E2Eformer细粒度控制

```bash
python train.py \
  --config config.yaml \
  --finetune \
  --enable_layer_control \
  --freeze_e2eformer_blocks 0 1 2 \
  --trainable_e2eformer_blocks 3 4 5 6 7 \
  --e2eformer_block_lr_ratios "3:0.05,4:0.08,5:0.1,6:0.15,7:0.2"
```

**日志输出示例**：
```
🎯 启用E2Eformer块细粒度控制
🔒 冻结E2Eformer块: [0, 1, 2]
🔓 指定可训练E2Eformer块: [3, 4, 5, 6, 7]
📊 参数组学习率配置:
  e2eformer_block_3: LR=5.00e-06, 参数=456,789, WD=1.00e-05
  e2eformer_block_4: LR=8.00e-06, 参数=456,789, WD=1.00e-05
  e2eformer_block_5: LR=1.00e-05, 参数=456,789, WD=1.00e-05
  e2eformer_block_6: LR=1.50e-05, 参数=456,789, WD=1.00e-05
  e2eformer_block_7: LR=2.00e-05, 参数=456,789, WD=1.00e-05
```

---

## 📊 Fine-tuning vs 预训练对比

### 训练流程差异

| 步骤 | 预训练 | Fine-tuning |
|------|--------|-------------|
| **1. 初始化** | 随机初始化 | 加载预训练权重 |
| **2. 参数冻结** | 无 | 部分冻结（freeze_strategy） |
| **3. 学习率** | 1e-3 ~ 1e-4 | **1e-5 ~ 1e-4** (小10-100倍) |
| **4. 分层学习率** | 不需要 | **推荐使用** |
| **5. Warmup** | 可选 | **推荐** |
| **6. 训练轮数** | 100+ | **10-30** |
| **7. 验证频率** | 每5轮 | **每轮** |
| **8. Early stopping** | patience=20 | **patience=5-10** |
| **9. 风险** | 过拟合风险低 | **过拟合风险高** |

### 是否需要验证集？

**答：是的，必须需要！而且比预训练更重要！**

#### 为什么？

1. **监控过拟合** 🔍
   - 小数据集更容易过拟合
   - 需要验证集及时发现过拟合信号
   
2. **Early Stopping** ⏹️
   - 基于验证loss决定何时停止
   - 防止在训练集上过度优化
   
3. **模型选择** 🏆
   - 保存验证loss最低的checkpoint
   - 而不是最后一个epoch的模型
   
4. **超参数调优** ⚙️
   - 学习率、冻结策略等
   - 都需要在验证集上评估

#### 验证集划分建议

```python
# 数据充足 (>500样本)
train: 80%, validation: 20%

# 数据较少 (100-500样本)
train: 85%, validation: 15%
# 建议使用k-fold交叉验证

# 数据很少 (<100样本)
使用5-fold或10-fold交叉验证
```

---

## 🔍 验证修复是否生效

### 测试命令

```bash
# 运行小规模测试
python train.py --test --finetune \
  --pretrained_checkpoint ./checkpoints/best_model.pt \
  --freeze_strategy rhofold_backbone
```

### 检查日志关键信息

✅ **成功标志**：
```
✅ "🎯 使用微调模式（X个参数组）"
✅ "📊 参数组学习率配置:"
✅ "  rhofold_backbone: LR=1.00e-05, 参数=XXX"
✅ "  diffusion_module: LR=1.00e-04, 参数=XXX"
✅ RhoFold和扩散模块的学习率不同！
```

❌ **失败标志**：
```
❌ 所有模块使用相同学习率
❌ 没有显示"参数组学习率配置"
❌ RhoFold学习率过大（>1e-4）
```

---

## 📝 配置文件完整示例

```yaml
# config.yaml - 微调专用配置
data:
  data_dir: "./processed_data"
  batch_size: 4  # 微调时可能需要更小的batch
  max_sequence_length: 256
  num_workers: 4
  fold: 0
  use_all_folds: false  # 保留验证集
  
model:
  rhofold_checkpoint: "./pretrained/model_20221010_params.pt"

training:
  num_epochs: 30  # 微调通常不需要太多轮
  learning_rate: 1e-4  # 比预训练小
  weight_decay: 1e-5
  grad_clip_norm: 1.0
  warmup_steps: 500  # 微调也需要warmup
  scheduler_type: "warmup_cosine"
  
  # ⚠️ 微调关键配置
  validate_every: 1  # 每轮都验证
  early_stopping_patience: 8  # 更严格
  
output:
  output_dir: "./output_finetune"
  checkpoint_dir: "./checkpoints_finetune"
  save_every: 1
  keep_last_n_checkpoints: 3

device:
  device: "auto"
  mixed_precision: true
  use_torch_compile: false

# 🔥 增强功能配置
enhanced_features:
  enable_enhanced_training: true
  monitoring:
    enable_performance_monitoring: true
    enable_health_checking: true
  optimizer:
    use_advanced_optimizer: true  # ✅ 支持微调！
    optimizer_name: "adamw"
    gradient_accumulation_steps: 1
    scheduler_type: "warmup_cosine"
  dataloader:
    enable_prefetch: true
    prefetch_factor: 2
  evaluation:
    compute_structure_metrics: true
    compute_confidence_metrics: true

# 🎯 微调配置
finetune:
  enable_finetuning: true
  pretrained_checkpoint: "./checkpoints/best_model.pt"
  
  # 冻结策略选择
  # - "none": 不冻结（全模型微调）
  # - "rhofold_only": 只冻结RhoFold（推荐小数据集）
  # - "rhofold_backbone": 冻结RhoFold骨干（推荐中等数据集）
  # - "rhofold_heads": 冻结输出头
  # - "diffusion_only": 只训练扩散模块
  # - "confidence_only": 只训练置信度
  freeze_strategy: "rhofold_backbone"
  
  # 全部解冻
  unfreeze_after_epochs: 0  # 0表示不自动解冻
  
  # 渐进式解冻
  gradual_unfreeze:
    enable: false  # 可选功能
    unfreeze_every: 5
    unfreeze_order: "top_down"
  
  # 分层学习率（重要！）
  learning_rate_scaling:
    enable: true
    backbone_lr_ratio: 0.1  # RhoFold用1/10学习率
    head_lr_ratio: 1.0      # 头部用完整学习率
    layer_wise_decay: 0.9
  
  # 🎯 细粒度控制（高级功能）
  layer_control:
    enable_layer_control: false
    e2eformer:
      freeze_blocks: []
      trainable_blocks: []
      block_lr_ratios: {}

logging:
  log_level: "INFO"
```

---

## ⚠️ 常见问题

### Q1: 训练loss不下降？

**可能原因和解决方案**:
1. **学习率过大** → 减小10倍: `learning_rate: 1e-5`
2. **学习率过小** → 增大2-5倍: `learning_rate: 5e-4`
3. **冻结太多** → 改用更激进的策略: `freeze_strategy: none`
4. **数据太少** → 使用更保守的策略: `freeze_strategy: rhofold_only`

### Q2: Validation loss上升（过拟合）？

**解决方案**:
```yaml
training:
  early_stopping_patience: 3  # 更严格
  
finetune:
  freeze_strategy: "rhofold_only"  # 冻结更多
  learning_rate_scaling:
    backbone_lr_ratio: 0.05  # 更小的学习率
```

### Q3: 如何选择freeze_strategy？

| 数据量 | 推荐策略 | 说明 |
|--------|---------|------|
| <50样本 | `rhofold_only` | 只训练扩散模块 |
| 50-200样本 | `rhofold_backbone` | 冻结骨干，训练其他部分 |
| 200-500样本 | `rhofold_backbone` + gradual_unfreeze | 渐进式解冻 |
| >500样本 | `none` | 全模型微调 |

### Q4: 增强优化器不工作？

**检查**:
```yaml
enhanced_features:
  enable_enhanced_training: true  # ✅ 必须启用
  optimizer:
    use_advanced_optimizer: true  # ✅ 必须启用
```

**日志检查**:
```bash
grep "使用微调模式" output/training.log
grep "参数组学习率配置" output/training.log
```

### Q5: 多GPU如何微调？

```bash
# 使用torchrun
torchrun --nproc_per_node=4 train.py \
  --config config.yaml \
  --finetune \
  --pretrained_checkpoint ./checkpoints/best_model.pt
```

---

## 🎯 成功标准

微调成功的标志：

1. ✅ **日志正确**
   ```
   🎯 使用微调模式（X个参数组）
   📊 参数组学习率配置: (各组LR不同)
   ```

2. ✅ **验证loss稳定下降**
   - 不震荡
   - 与训练loss趋势一致

3. ✅ **性能提升**
   - 目标数据集上提升10%+
   - RMSD下降、TM-score上升

4. ✅ **没有严重过拟合**
   - train/val gap < 20%

5. ✅ **训练稳定**
   - 没有NaN
   - 没有梯度爆炸

---

## 📚 相关文件

- `diffold/advanced_optimizers.py` - 修改后的优化器（支持参数组）
- `train.py` - 修改后的训练脚本
- `FINE_GRAIN_FINETUNE_EXAMPLES.md` - 细粒度控制示例
- `config.yaml` - 配置文件示例

---

## 🔄 从旧版本迁移

如果你之前使用过fine-tuning功能，现在不需要改动配置文件，只需要：

1. ✅ 更新代码（已完成）
2. ✅ 保持配置不变
3. ✅ 重新训练

**自动兼容性**：
- 标准训练：自动使用 `model` 参数
- 微调训练：自动使用 `param_groups` 参数
- 无需手动切换

---

## 💡 最佳实践总结

### ✅ DO (推荐)

1. **Always使用验证集** - 即使数据少，也要留验证集
2. **从小学习率开始** - 微调LR = 预训练LR / 10~100
3. **使用分层学习率** - RhoFold用小LR，新增层用大LR
4. **频繁验证** - 每个epoch都验证
5. **严格early stopping** - patience=5-10
6. **启用warmup** - 微调前2-3个epoch用更小LR
7. **监控所有指标** - Loss、RMSD、TM-score等

### ❌ DON'T (避免)

1. **学习率过大** - 会破坏预训练权重
2. **冻结太多层** - 可能导致欠拟合
3. **训练太久** - 小数据集容易过拟合
4. **忽略验证集** - 无法判断过拟合
5. **在测试集上调参** - 会过度优化测试集

---

**最后更新**: 2025-10-14  
**状态**: ✅ 已修复并测试  
**版本**: v2.0 - Fine-tuning兼容版

