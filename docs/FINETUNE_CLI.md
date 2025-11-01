# 🎯 Diffold 微调命令行指南

## 简介

`finetune.py` 是专门的微调脚本，提供简洁的命令行接口，让微调更容易上手。

## 与 train.py 的区别

| 特性 | train.py | finetune.py |
|------|----------|-------------|
| 用途 | 从头训练 | 在预训练模型基础上微调 |
| 预训练模型 | ❌ | ✅ 必需 |
| 微调模式 | 需手动配置 | ✅ 默认启用 |
| LoRA支持 | 需配置文件 | ✅ 命令行开关 |
| 接口 | 复杂（通用） | 简洁（专用） |

## 快速开始

### 场景1: 小数据集微调（<100样本）+ LoRA

```bash
python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora \
  --lora-r 4 \
  --freeze-strategy rhofold_only \
  --learning-rate 5e-5 \
  --epochs 20
```

**推荐配置**:
- LoRA rank=4（参数最少）
- 冻结RhoFold（只训练扩散模块）
- 小学习率（5e-5）
- 较少epochs（20）

### 场景2: 中等数据集（100-500样本）+ LoRA

```bash
python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora \
  --lora-r 8 \
  --lora-strategy diffusion_confidence \
  --freeze-strategy rhofold_backbone \
  --learning-rate 1e-4 \
  --epochs 30
```

**推荐配置**:
- LoRA rank=8（平衡性能和效率）
- 策略: diffusion_confidence（扩散+置信度）
- 冻结RhoFold骨干
- 中等学习率（1e-4）

### 场景3: 较大数据集（500-1000样本）+ LoRA

```bash
python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora \
  --lora-r 12 \
  --lora-strategy diffusion_all_heads \
  --freeze-strategy rhofold_backbone \
  --learning-rate 2e-4 \
  --epochs 40
```

### 场景4: 大数据集（>1000样本）标准微调

```bash
python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --freeze-strategy none \
  --learning-rate 1e-4 \
  --epochs 50
```

**不使用LoRA**，全模型微调。

## 命令行参数详解

### 必需参数

```bash
--pretrained PATH           # 预训练模型路径
```

### 微调策略

```bash
--freeze-strategy STRATEGY  # 冻结策略
  # 选项:
  #   - none: 不冻结（全模型微调）
  #   - rhofold_only: 仅冻结RhoFold（推荐<100样本）
  #   - rhofold_backbone: 冻结RhoFold骨干（推荐100-500样本）
  #   - rhofold_heads: 冻结RhoFold输出头

--backbone-lr-ratio FLOAT   # 骨干网络学习率比例
  # 默认: 0.1 (即骨干网络用1/10学习率)
```

### LoRA配置

```bash
--use-lora                  # 启用LoRA微调

--lora-r INT                # LoRA rank
  # 推荐:
  #   - 4: <100样本
  #   - 8: 100-500样本
  #   - 12: 500-1000样本
  #   - 16+: >1000样本

--lora-alpha INT            # LoRA alpha
  # 默认: 等于rank
  # 控制LoRA权重的贡献度

--lora-dropout FLOAT        # LoRA dropout
  # 默认: 0.05
  # 防止过拟合

--lora-strategy STRATEGY    # LoRA应用策略
  # 选项:
  #   - diffusion_only: 仅扩散模块
  #   - diffusion_confidence: 扩散+置信度（默认）
  #   - diffusion_all_heads: 扩散+所有头部
  #   - full_model: 全模型LoRA
```

### 训练参数

```bash
--learning-rate FLOAT       # 学习率
  # 推荐:
  #   - 5e-5: 小数据集 + 激进微调
  #   - 1e-4: 中等数据集（最常用）
  #   - 2e-4: 较大数据集

--epochs INT                # 训练轮数
  # 推荐:
  #   - 20: 小数据集
  #   - 30: 中等数据集
  #   - 40-50: 较大数据集

--batch-size INT            # Batch大小
  # 根据显存调整，默认为配置文件中的值
```

### 其他

```bash
--config PATH               # 配置文件路径 (默认: config.yaml)
--output-dir PATH           # 输出目录
--dry-run                   # 仅显示配置，不执行训练
```

## 使用配置文件

如果有复杂配置，可以创建专门的微调配置文件：

**config_finetune.yaml:**
```yaml
# 数据配置
data:
  data_dir: "./processed_data"
  batch_size: 1
  fold: 0  # 确保有验证集！

# 训练配置
training:
  learning_rate: 1e-4
  num_epochs: 30
  validate_every: 1
  early_stopping_patience: 10

# 微调配置
finetune:
  enable_finetuning: true
  pretrained_checkpoint: "./checkpoints/best_model.pt"
  freeze_strategy: "rhofold_backbone"
  learning_rate_scaling:
    enable: true
    backbone_lr_ratio: 0.1

# LoRA配置
lora:
  enable: true
  r: 8
  alpha: 8
  dropout: 0.05
  strategy: "diffusion_confidence"
```

然后运行：
```bash
python finetune.py --config config_finetune.yaml
```

## Dry Run 模式

在实际训练前，可以先检查配置：

```bash
python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora \
  --lora-r 8 \
  --dry-run
```

会显示完整配置但不执行训练。

## 输出示例

运行微调时会看到：

```
============================================================
🎯 Diffold 微调配置
============================================================

📦 预训练模型:
  checkpoints/best_model.pt

🔒 冻结策略: rhofold_backbone

🎯 LoRA配置:
  启用: ✅
  Rank: 8
  Alpha: 8
  Dropout: 0.05
  策略: diffusion_confidence

📊 学习率配置:
  基础学习率: 0.0001
  分层学习率: ✅
    骨干网络: 0.1x
    头部网络: 1.0x

🏃 训练参数:
  Epochs: 30
  Batch Size: 1
  验证频率: 每 1 epoch
  Early Stopping: 10 epochs

============================================================

🚀 开始微调...
```

## 常见问题

### Q: 我应该用LoRA还是标准微调？

**A**: 根据数据量决定：
- **<1000样本**: 强烈推荐LoRA
- **>1000样本**: 可以考虑标准微调

### Q: freeze-strategy和LoRA strategy的区别？

**A**:
- **freeze-strategy**: 决定**哪些模块**参与训练
- **lora-strategy**: 决定**如何训练**（在哪些模块应用LoRA）

两者配合使用：
```bash
--freeze-strategy rhofold_only    # RhoFold冻结
--lora-strategy diffusion_confidence  # 在扩散+置信度用LoRA
```

### Q: 如何选择learning-rate？

**A**:
- LoRA微调: 1e-4 ~ 2e-4（可以用较大学习率）
- 标准微调: 5e-5 ~ 1e-4（需要较小学习率）
- 从推荐值开始，观察loss调整

### Q: 训练loss不下降怎么办？

**A**: 尝试：
1. 增大学习率: `--learning-rate 2e-4`
2. 减小LoRA rank: `--lora-r 12`（增加表达能力）
3. 减少冻结: `--freeze-strategy none`

### Q: Validation loss上升（过拟合）？

**A**: 尝试：
1. 减小LoRA rank: `--lora-r 4`
2. 增大dropout: `--lora-dropout 0.1`
3. 增加冻结: `--freeze-strategy rhofold_only`
4. 减小学习率: `--learning-rate 5e-5`

## 完整示例

### 针对你的情况（<1000样本）

```bash
# 1. 检查配置
python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora \
  --lora-r 8 \
  --lora-strategy diffusion_confidence \
  --freeze-strategy rhofold_backbone \
  --learning-rate 1e-4 \
  --epochs 30 \
  --dry-run

# 2. 开始训练
python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora \
  --lora-r 8 \
  --lora-strategy diffusion_confidence \
  --freeze-strategy rhofold_backbone \
  --learning-rate 1e-4 \
  --epochs 30
```

## 参考

- 详细LoRA文档: `docs/LORA_USAGE.md`
- 快速开始: `LORA_QUICKSTART.md`
- 微调策略: `docs/FINETUNE_USAGE.md`

