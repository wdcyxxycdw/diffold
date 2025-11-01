# 🎯 Diffold 微调快速指南

## 新增专用微调脚本

现在微调功能已经独立出来，使用更简单！

### 📁 文件说明

- **`train.py`**: 从头训练（预训练）
- **`finetune.py`**: 微调专用脚本 ⭐ 新增
- **`docs/FINETUNE_CLI.md`**: 详细命令行文档

## 🚀 快速开始

### 针对你的情况（<1000样本）

**推荐配置：LoRA微调**

```bash
# 使用LoRA微调（最简单）
uv run python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora \
  --lora-r 8 \
  --freeze-strategy rhofold_backbone \
  --learning-rate 1e-4 \
  --epochs 30
```

**这一条命令就够了！** 会自动：
- ✅ 启用微调模式
- ✅ 加载预训练模型
- ✅ 应用LoRA（rank=8）
- ✅ 冻结RhoFold骨干
- ✅ 使用分层学习率
- ✅ 设置合适的训练参数

### 先检查配置（推荐）

```bash
# 添加 --dry-run 先看看配置
uv run python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora \
  --lora-r 8 \
  --freeze-strategy rhofold_backbone \
  --learning-rate 1e-4 \
  --epochs 30 \
  --dry-run
```

会显示：
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

...
```

确认无误后去掉 `--dry-run` 开始训练。

## 📊 根据数据量选择配置

### < 100 样本

```bash
uv run python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora --lora-r 4 \
  --freeze-strategy rhofold_only \
  --learning-rate 5e-5 \
  --epochs 20
```

**最保守配置**：
- rank=4（最少参数）
- 只训练扩散模块
- 小学习率
- 较少epochs

### 100-500 样本

```bash
uv run python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora --lora-r 8 \
  --freeze-strategy rhofold_backbone \
  --learning-rate 1e-4 \
  --epochs 30
```

**平衡配置** ⭐ 推荐

### 500-1000 样本

```bash
uv run python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora --lora-r 12 \
  --lora-strategy diffusion_all_heads \
  --freeze-strategy rhofold_backbone \
  --learning-rate 2e-4 \
  --epochs 40
```

**更激进的配置**

### > 1000 样本

```bash
uv run python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --freeze-strategy none \
  --learning-rate 1e-4 \
  --epochs 50
```

**标准微调**（不使用LoRA）

## 🎛️ 主要参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--pretrained` | 预训练模型路径 | 必需 |
| `--use-lora` | 启用LoRA | <1000样本建议启用 |
| `--lora-r` | LoRA rank | 4-12，数据越多越大 |
| `--freeze-strategy` | 冻结策略 | rhofold_backbone |
| `--learning-rate` | 学习率 | 1e-4（LoRA）, 5e-5（标准） |
| `--epochs` | 训练轮数 | 20-40 |

## 💡 常用技巧

### 1. 从配置文件启动

创建 `config_finetune.yaml`:
```yaml
finetune:
  enable_finetuning: true
  pretrained_checkpoint: "./checkpoints/best_model.pt"
  freeze_strategy: "rhofold_backbone"

lora:
  enable: true
  r: 8
  strategy: "diffusion_confidence"

training:
  learning_rate: 1e-4
  num_epochs: 30
```

然后：
```bash
uv run python finetune.py --config config_finetune.yaml
```

### 2. 自定义输出目录

```bash
uv run python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora \
  --output-dir ./finetune_results
```

### 3. 调整batch size（显存不够时）

```bash
uv run python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora \
  --batch-size 1  # 减小batch size
```

## 🆚 vs train.py

| 特性 | train.py | finetune.py |
|------|----------|-------------|
| 用途 | 从头训练 | 微调 |
| 预训练模型 | ❌ | ✅ 必需 |
| 命令行 | 通用但复杂 | 专用且简洁 |
| LoRA | 需配置文件 | `--use-lora`开关 |
| 推荐场景 | 大规模预训练 | 在预训练基础上适配 |

## 📚 更多文档

- **命令行详解**: `docs/FINETUNE_CLI.md`
- **LoRA使用指南**: `docs/LORA_USAGE.md`
- **LoRA快速开始**: `LORA_QUICKSTART.md`
- **微调策略**: `docs/FINETUNE_USAGE.md`

## ❓ 常见问题

### Q: 必须使用LoRA吗？

**A**: 不是，但强烈推荐：
- <1000样本：**建议使用LoRA**（防止过拟合）
- >1000样本：可以不用LoRA

### Q: 如何知道配置是否合适？

**A**: 使用 `--dry-run` 先检查：
```bash
uv run python finetune.py [你的参数] --dry-run
```

### Q: 训练中断了怎么办？

**A**: `finetune.py` 会自动保存checkpoint，下次继续训练即可。

### Q: 如何查看所有参数？

**A**: 
```bash
uv run python finetune.py --help
```

## 🎉 总结

使用新的 `finetune.py` 脚本：

1. ✅ **更简单**：一条命令搞定
2. ✅ **更安全**：自动启用微调模式
3. ✅ **更清晰**：配置摘要一目了然
4. ✅ **更专业**：针对微调优化

**立即开始：**
```bash
uv run python finetune.py \
  --pretrained checkpoints/best_model.pt \
  --use-lora \
  --lora-r 8 \
  --learning-rate 1e-4 \
  --epochs 30
```

祝微调顺利！🚀

