#!/bin/bash

# 批量推理和指标计算脚本
# 使用方法: ./run_batch_inference.sh

echo "开始批量推理和指标计算..."

# 设置参数
CHECKPOINT_PATH="./checkpoints/best_model.pt"  # 修改为您的模型检查点路径
DATA_DIR="./processed_data"
OUTPUT_DIR="./batch_inference_output"
FOLD=3  # 验证集折数
MAX_SAMPLES=10  # 测试时限制样本数量，正式运行时设为None

# 运行批量推理
python batch_inference_metrics.py \
    --checkpoint_path ./checkpoints/checkpoint_epoch_019.pt \
    --data_dir ./processed_data \
    --output_dir ./batch_inference_output \
    --fold 3 \
    --max_samples 10 \
    --device auto \
    --log_level INFO

echo "批量推理完成！"
echo "结果文件保存在: $OUTPUT_DIR" 