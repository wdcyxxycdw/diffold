# 🚀 Diffold增强训练 - 快速开始

## 📦 安装依赖

```bash
# 基础依赖（增强功能需要）
pip install psutil matplotlib

# 可选：高级优化器（如果需要Lion优化器）
# pip install lion-pytorch
```

## 🎯 快速开始

### 1. 基础训练（自动启用增强功能）
```bash
# 默认配置 - 自动启用所有增强功能
python train.py

# 指定数据目录和基础参数
python train.py --data_dir ./processed_data --batch_size 4 --epochs 50
```

### 2. 预设配置模式
```bash
# 🚀 性能优先模式（最大化训练速度）
python train.py --enhanced_preset performance

# 🛡️ 安全优先模式（最大化训练稳定性）
python train.py --enhanced_preset safety

# 💾 内存高效模式（内存受限环境）
python train.py --enhanced_preset memory

# 🐛 调试模式（详细监控和日志）
python train.py --enhanced_preset debug
```

### 3. 选择性禁用功能
```bash
# 禁用所有增强功能（使用原版训练）
python train.py --disable_enhanced

# 禁用特定功能
python train.py --disable_monitoring          # 禁用性能监控
python train.py --disable_prefetch           # 禁用数据预取
python train.py --disable_advanced_optimizer # 禁用高级优化器

# 组合使用
python train.py --enhanced_preset safety --disable_prefetch
```

### 4. 小规模测试
```bash
# 运行测试模式（自动启用调试功能）
python train.py --test
```

## 📊 **配置参数说明**

### 基础参数（与原版兼容）
```bash
python train.py \
    --data_dir ./processed_data \
    --batch_size 4 \
    --max_length 256 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --device auto \
    --output_dir ./output
```

### 🔥 增强功能参数
```bash
# 预设配置
--enhanced_preset {performance,safety,memory,debug}

# 功能开关
--disable_enhanced              # 禁用所有增强功能
--disable_monitoring           # 禁用性能监控
--disable_prefetch            # 禁用数据预取
--disable_advanced_optimizer  # 禁用高级优化器
```

### 设备和性能参数
```bash
--device auto                 # 自动选择设备
--no_mixed_precision         # 禁用混合精度
--no_data_parallel          # 禁用多GPU
--gpu_ids 0 1               # 指定GPU ID
```

## 🎛️ **推荐使用场景**

### 🔰 初次使用
```bash
# 建议从小规模测试开始
python train.py --test

# 然后使用平衡配置
python train.py --enhanced_preset safety --epochs 10
```

### 🚀 追求最高性能
```bash
python train.py --enhanced_preset performance \
    --batch_size 8 \
    --epochs 100
```

### 🛡️ 稳定性优先
```bash
python train.py --enhanced_preset safety \
    --batch_size 2 \
    --epochs 100
```

### 💾 内存受限环境
```bash
python train.py --enhanced_preset memory \
    --batch_size 1 \
    --max_length 200
```

### 🐛 调试和分析
```bash
python train.py --enhanced_preset debug \
    --epochs 5 \
    --batch_size 2
```

## 📈 **输出和监控**

### 训练输出目录结构
```
output/
├── training.log              # 详细训练日志
├── tensorboard/             # TensorBoard日志
├── plots/                   # 训练曲线图
├── training_metrics.json    # 训练指标
├── training_monitor_report.json  # 监控报告（如果启用）
└── performance_metrics.png  # 性能图表（如果启用）

checkpoints/
├── best_model.pt           # 最佳模型
├── checkpoint_epoch_xxx.pt # 定期检查点
└── ...
```

### 实时监控
训练过程中会显示：
- 📊 实时损失和学习率
- ⏱️ 训练时间和预计完成时间
- 🔍 内存使用情况（如果启用监控）
- 🚨 异常检测和警告（如果启用健康检查）

## 🔧 **故障排除**

### 1. 增强功能不可用
```bash
# 如果看到 "增强功能不可用" 提示
pip install psutil matplotlib

# 或者使用原版功能
python train.py --disable_enhanced
```

### 2. 内存不足 (OOM)
```bash
# 自动处理（如果启用error_recovery）
python train.py --enhanced_preset memory

# 手动调整
python train.py --batch_size 1 --max_length 200
```

### 3. 训练速度慢
```bash
# 启用性能模式
python train.py --enhanced_preset performance

# 检查是否启用数据预取
python train.py  # 默认启用预取
```

### 4. 训练不稳定
```bash
# 使用安全模式
python train.py --enhanced_preset safety

# 或者使用更保守的设置
python train.py --batch_size 2 --learning_rate 5e-5
```

## 📋 **完整示例**

### 生产环境训练
```bash
python train.py \
    --enhanced_preset performance \
    --data_dir ./processed_data \
    --batch_size 6 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --output_dir ./results \
    --checkpoint_dir ./checkpoints \
    --save_every 10
```

### 研究实验
```bash
python train.py \
    --enhanced_preset debug \
    --data_dir ./processed_data \
    --batch_size 4 \
    --epochs 50 \
    --fold 0 \
    --output_dir ./experiment_fold0
```

### 资源受限环境
```bash
python train.py \
    --enhanced_preset memory \
    --batch_size 1 \
    --max_length 200 \
    --no_mixed_precision \
    --disable_prefetch
```

## 🎉 **就这么简单！**

增强训练系统现在完全集成到单个 `train.py` 文件中，你可以：

1. **🔄 无缝升级**：现有训练命令完全兼容
2. **🎛️ 灵活配置**：根据需求选择不同的优化组合
3. **📊 实时监控**：获得详细的训练过程洞察
4. **🛡️ 自动保护**：避免常见的训练问题
5. **⚡ 性能提升**：预计节省20-40%的训练时间

从 `python train.py --test` 开始，享受更高效、更稳定的训练体验！ 