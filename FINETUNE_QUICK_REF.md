# 🚀 Fine-tuning 快速参考卡

## ⚡ 3分钟开始使用

### 1️⃣ 准备配置文件

```yaml
# config_finetune.yaml
finetune:
  enable_finetuning: true
  pretrained_checkpoint: "./checkpoints/best_model.pt"
  freeze_strategy: "rhofold_backbone"  # 根据数据量选择
  learning_rate_scaling:
    enable: true
    backbone_lr_ratio: 0.1

training:
  learning_rate: 1e-4  # 比预训练小
  validate_every: 1    # 每轮验证
  early_stopping_patience: 8

enhanced_features:
  enable_enhanced_training: true
  optimizer:
    use_advanced_optimizer: true  # ✅ 支持fine-tuning
```

### 2️⃣ 运行训练

```bash
python train.py --config config_finetune.yaml
```

### 3️⃣ 检查日志

✅ **成功标志**:
```
🎯 使用微调模式（6个参数组）
📊 参数组学习率配置:
  rhofold_backbone: LR=1.00e-05  # 小学习率
  diffusion_module: LR=1.00e-04  # 大学习率
```

---

## 📋 Freeze Strategy 快速选择

| 数据量 | 策略 | 命令行参数 |
|--------|------|-----------|
| <50样本 | `rhofold_only` | `--freeze_strategy rhofold_only` |
| 50-200样本 | `rhofold_backbone` | `--freeze_strategy rhofold_backbone` |
| 200-500样本 | `rhofold_backbone` + 渐进解冻 | `--freeze_strategy rhofold_backbone --gradual_unfreeze` |
| >500样本 | `none` | `--freeze_strategy none` |

---

## 🎯 关键参数对比

| 参数 | 预训练 | Fine-tuning |
|------|--------|-------------|
| `learning_rate` | 1e-3 ~ 1e-4 | **1e-5 ~ 1e-4** |
| `num_epochs` | 100+ | **10-30** |
| `validate_every` | 5 | **1** |
| `early_stopping_patience` | 20 | **5-10** |
| `warmup_steps` | 1000 | **500** |

---

## ✅ 验证集必须有！

```yaml
data:
  use_all_folds: false  # ❌ 不要用所有fold
  fold: 0               # ✅ 保留验证集
```

**原因**:
- 🔍 监控过拟合
- ⏹️ Early stopping
- 🏆 模型选择
- ⚙️ 超参数调优

---

## 🔥 常用命令

### 小数据集微调
```bash
python train.py --finetune \
  --pretrained_checkpoint ./checkpoints/best_model.pt \
  --freeze_strategy rhofold_only \
  --learning_rate 5e-5 \
  --epochs 20
```

### 中等数据集 + 渐进解冻
```bash
python train.py --finetune \
  --pretrained_checkpoint ./checkpoints/best_model.pt \
  --freeze_strategy rhofold_backbone \
  --gradual_unfreeze \
  --unfreeze_every 5 \
  --learning_rate 1e-4 \
  --epochs 30
```

### E2Eformer细粒度控制
```bash
python train.py --finetune \
  --enable_layer_control \
  --freeze_e2eformer_blocks 0 1 2 \
  --trainable_e2eformer_blocks 3 4 5 6 7 \
  --e2eformer_block_lr_ratios "3:0.05,4:0.08,5:0.1"
```

---

## ⚠️ 常见错误

### ❌ 学习率太大
```yaml
training:
  learning_rate: 1e-3  # ❌ 太大，会破坏权重
```
✅ **修复**: `learning_rate: 1e-4` 或更小

### ❌ 没有验证集
```yaml
data:
  use_all_folds: true  # ❌ 用了所有数据
```
✅ **修复**: `use_all_folds: false`

### ❌ 验证太少
```yaml
training:
  validate_every: 5  # ❌ 验证太少
```
✅ **修复**: `validate_every: 1`

### ❌ Early stopping太宽松
```yaml
training:
  early_stopping_patience: 20  # ❌ 太宽松
```
✅ **修复**: `early_stopping_patience: 5-8`

---

## 🐛 故障排查

### 问题1: Loss不下降

**症状**: 训练loss一直很高

**可能原因和修复**:
```bash
# 1. 学习率过大 → 减小
--learning_rate 1e-5

# 2. 学习率过小 → 增大
--learning_rate 5e-4

# 3. 冻结太多 → 解冻更多
--freeze_strategy none

# 4. 检查预训练模型是否正确加载
grep "加载预训练权重" output/training.log
```

### 问题2: Validation loss上升

**症状**: 训练loss下降，验证loss上升

**诊断**: 过拟合！

**修复**:
```yaml
finetune:
  freeze_strategy: "rhofold_only"  # 冻结更多
training:
  early_stopping_patience: 3        # 更严格
  learning_rate: 5e-5               # 更小学习率
```

### 问题3: 参数组不生效

**检查日志**:
```bash
grep "参数组学习率配置" output/training.log
```

**如果没有这行输出**:
```yaml
enhanced_features:
  enable_enhanced_training: true  # ✅ 必须true
  optimizer:
    use_advanced_optimizer: true  # ✅ 必须true

finetune:
  enable_finetuning: true         # ✅ 必须true
  learning_rate_scaling:
    enable: true                  # ✅ 必须true
```

---

## 📊 成功标准

训练成功的标志：

1. ✅ 日志显示参数组配置
2. ✅ 各模块学习率不同
3. ✅ 验证loss稳定下降
4. ✅ 性能提升10%+
5. ✅ train/val gap < 20%

---

## 📚 完整文档

- **快速入门**: `FINETUNE_USAGE.md`
- **修复说明**: `FINETUNE_FIX_SUMMARY.md`
- **细粒度控制**: `FINE_GRAIN_FINETUNE_EXAMPLES.md`

---

## 💡 最佳实践速记

✅ **Always DO**:
- 保留验证集
- 每轮验证
- 小学习率
- 使用分层LR
- 严格early stopping

❌ **Never DO**:
- 用所有数据训练
- LR太大（>1e-3）
- 在测试集上调参
- 训练太久（>50 epochs）
- 忽略过拟合信号

---

**🎯 记住这个公式**:

```
Fine-tuning学习率 = 预训练学习率 / 10~100
```

```
RhoFold学习率 = 扩散模块学习率 / 10
```

```
验证频率 = 每1 epoch（不是5！）
```

---

**版本**: v2.0  
**更新**: 2025-10-14  
**状态**: ✅ 可用

