# RhoFold

RhoFold 项目 - RNA 结构预测与分析工具

## 简介

基于深度学习的 RNA 三维结构预测项目，支持训练、推理和微调功能。

## 主要功能

- **训练**: 使用 `train.py` 训练模型
- **推理**: 使用 `inference_diffold.py` 或 `inference_rf.py` 进行结构预测
- **微调**: 使用 `finetune.py` 对预训练模型进行微调
- **批量推理**: 使用 `batch_inference_rhofold.py` 进行批量预测

## 环境要求

- Python >= 3.10
- PyTorch >= 1.12.0
- OpenMM >= 7.7.0
- 其他依赖见 `pyproject.toml`

## 安装

```bash
# 使用 uv 安装依赖
uv sync
```

## 使用

```bash
# 训练
uv run train.py

# 推理
uv run inference_diffold.py

# 微调
uv run finetune.py
```

## 许可证

见 LICENSE 文件

