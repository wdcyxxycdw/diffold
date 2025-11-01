# 🎯 Diffold LoRA微调使用指南

## 📋 简介

LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法，可以在保持模型性能的同时大幅减少可训练参数数量和显存占用。

### 为什么使用LoRA？

| 对比项 | 标准微调 | LoRA微调 |
|--------|---------|---------|
| 可训练参数 | ~37M | ~0.78M (rank=8) |
| 显存占用 | 高 | 低 (~98%节省) |
| 训练速度 | 慢 | 快 |
| 过拟合风险 | 高（小数据集） | 低 |
| 适用数据量 | >1000样本 | <1000样本 |

**推荐场景**: 您的数据集不到1000条，非常适合使用LoRA！

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用uv安装PEFT库
uv pip install peft>=0.7.0
```

### 2. 配置LoRA

编辑 `config.yaml`:

```yaml
# 启用LoRA微调
lora:
  enable: true  # 开启LoRA
  r: 8  # rank=8适合中小数据集
  alpha: 8
  dropout: 0.05
  strategy: "diffusion_confidence"  # 扩散模块+置信度头
  save_adapter_only: true  # 仅保存LoRA权重，节省空间

# 同时配置微调
finetune:
  enable_finetuning: true
  pretrained_checkpoint: "./checkpoints/best_model.pt"  # 预训练模型
  freeze_strategy: "rhofold_only"  # 冻结RhoFold，训练扩散模块
  learning_rate_scaling:
    enable: true
    backbone_lr_ratio: 0.1

# 训练配置
training:
  learning_rate: 1e-4  # LoRA可以使用较大学习率
  num_epochs: 30
  validate_every: 1
  early_stopping_patience: 10
```

### 3. 开始训练

```bash
# 激活虚拟环境
source .venv/bin/activate

# 或使用uv run
uv run python train.py --config config.yaml
```

## 📊 LoRA策略选择

根据您的数据集大小选择合适的策略：

### 策略对比

| 策略 | 目标模块 | 数据量 | rank推荐 | 说明 |
|------|---------|--------|---------|------|
| `diffusion_only` | 仅扩散模块 | <100样本 | 4-8 | 最保守，防止过拟合 |
| `diffusion_confidence` | 扩散+置信度 | 100-500样本 | 8-12 | **推荐您使用** |
| `diffusion_all_heads` | 扩散+所有头 | 500-1000样本 | 12-16 | 更全面的微调 |
| `full_model` | 包括RhoFold | >1000样本 | 16-32 | 全模型LoRA |

### 配置示例

#### 方案A: 保守微调（<100样本）

```yaml
lora:
  enable: true
  r: 4
  alpha: 4
  dropout: 0.1
  strategy: "diffusion_only"

training:
  learning_rate: 5e-5
  early_stopping_patience: 5
```

#### 方案B: 推荐配置（<1000样本）✨

```yaml
lora:
  enable: true
  r: 8
  alpha: 8
  dropout: 0.05
  strategy: "diffusion_confidence"

training:
  learning_rate: 1e-4
  early_stopping_patience: 10
```

#### 方案C: 激进微调（接近1000样本）

```yaml
lora:
  enable: true
  r: 12
  alpha: 16
  dropout: 0.05
  strategy: "diffusion_all_heads"

training:
  learning_rate: 2e-4
  early_stopping_patience: 15
```

## 💾 模型保存和加载

### 保存格式

LoRA模式下会保存两种文件：

```
checkpoints/
├── lora_epoch_001/        # LoRA适配器权重
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_state.pt   # 训练状态
├── lora_epoch_002/
├── ...
└── best_lora/             # 最佳LoRA模型
    ├── adapter_config.json
    ├── adapter_model.bin
    └── training_state.pt
```

**优势**: LoRA适配器文件通常只有几MB，相比完整模型的几GB大大节省空间！

### 推理时使用LoRA模型

有两种方式：

#### 方式1: 加载LoRA适配器（推荐）

```python
from diffold.diffold import Diffold
from diffold.lora_utils import LoRAManager

# 1. 加载基础模型
base_model = Diffold(config, load_rhofold_weights=True)

# 2. 加载LoRA适配器
peft_model = LoRAManager.load_lora_weights(
    base_model, 
    "./checkpoints/best_lora"
)

# 3. 推理
peft_model.eval()
outputs = peft_model(tokens, rna_fm_tokens, seq, ...)
```

#### 方式2: 合并权重（用于部署）

```python
from diffold.lora_utils import LoRAManager

# 加载PEFT模型
peft_model = LoRAManager.load_lora_weights(base_model, "./checkpoints/best_lora")

# 合并LoRA权重到基础模型
merged_model = LoRAManager.merge_and_unload(peft_model)

# 保存完整模型
torch.save(merged_model.state_dict(), "merged_model.pt")
```

## 🔧 高级配置

### 自定义目标模块

如果默认策略不满足需求，可以自定义：

```yaml
lora:
  enable: true
  r: 8
  alpha: 8
  strategy: "custom"
  custom_target_modules:
    - "diffusion.token_transformer"  # DiffusionTransformer
    - "confidence_head.pairformer"   # ConfidenceHead
    - "single_dim_adapter"           # 维度适配器
```

### LoRA参数说明

- **r (rank)**: LoRA秩，控制低秩矩阵维度
  - 更大的r = 更多参数 = 更强表达能力 = 更容易过拟合
  - 推荐: 数据少用小r (4-8)，数据多用大r (12-32)

- **alpha**: 缩放因子，控制LoRA权重的贡献度
  - 通常设置为 r 的 1-2 倍
  - 实际学习率 = alpha / r
  - alpha = r: 标准设置
  - alpha > r: LoRA贡献更大

- **dropout**: LoRA层的dropout率
  - 用于防止过拟合
  - 数据少时增大 (0.1-0.2)
  - 数据多时减小 (0.0-0.05)

## 📈 训练监控

训练时会看到类似输出：

```
============================================================
🎯 应用LoRA微调
============================================================
📋 LoRA配置:
  Rank (r): 8
  Alpha: 8
  Dropout: 0.05
  Bias: none
  策略: diffusion_confidence
✅ LoRA应用成功!
📊 参数统计:
  可训练参数: 786,432
  总参数: 42,531,840
  可训练比例: 1.85%
============================================================
```

关注 "可训练比例" - 应该在 1-5% 之间，说明LoRA正常工作。

## ⚠️ 常见问题

### Q1: LoRA和freeze_strategy的关系？

- **freeze_strategy**: 决定哪些模块参与训练
- **LoRA**: 决定参与训练的模块如何高效训练（低秩分解）

**推荐组合**:
```yaml
finetune:
  freeze_strategy: "rhofold_only"  # 只训练扩散部分
lora:
  enable: true
  strategy: "diffusion_confidence"  # 在扩散部分用LoRA
```

### Q2: 训练loss不下降？

可能原因和解决方案：

1. **rank太小**: 增大 r 到 12-16
2. **学习率太小**: 增大到 2e-4
3. **需要更多epoch**: LoRA收敛较快，但仍需要足够epoch

### Q3: 过拟合怎么办？

```yaml
lora:
  r: 4  # 减小rank
  dropout: 0.1  # 增大dropout

training:
  learning_rate: 5e-5  # 减小学习率
  early_stopping_patience: 5  # 更严格的early stopping
```

### Q4: 显存还是不够？

```yaml
training:
  batch_size: 1  # 减小batch size

enhanced_features:
  optimizer:
    gradient_accumulation_steps: 4  # 梯度累积
```

### Q5: 如何查看模型是否正确应用了LoRA？

训练开始时会打印LoRA模块列表：

```
📝 LoRA模块列表:
  1. base_model.diffusion.token_transformer.0.attn.qkv.lora_A.default
  2. base_model.diffusion.token_transformer.0.attn.qkv.lora_B.default
  ...
```

如果显示 "未找到LoRA模块"，说明 target_modules 配置不正确。

## 📚 参考资料

- LoRA论文: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- PEFT库文档: https://huggingface.co/docs/peft
- Diffold微调文档: `docs/FINETUNE_USAGE.md`
- LoRA架构分析: `LORA_ANALYSIS.md`

## 🎉 总结

使用LoRA微调Diffold的完整流程：

1. ✅ 安装 `peft` 库
2. ✅ 在 `config.yaml` 中启用LoRA
3. ✅ 选择合适的策略和rank
4. ✅ 运行训练: `python train.py`
5. ✅ 监控可训练参数比例（1-5%）
6. ✅ 使用最佳LoRA模型进行推理

**针对您的情况（<1000样本）**:
- 使用 `strategy: "diffusion_confidence"`
- 设置 `r: 8`, `alpha: 8`
- 学习率 `1e-4`
- 预计可训练参数仅占 ~2%，训练速度快，显存占用低！

祝训练顺利！🚀

