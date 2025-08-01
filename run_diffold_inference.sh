#!/bin/bash

# Diffold推理运行脚本
# 使用方法: ./run_diffold_inference.sh [输入FASTA文件] [输出目录] [检查点路径]

# 设置默认参数
INPUT_FAS=${1:-"./example/input/3owzA/3owzA.fasta"}
OUTPUT_DIR=${2:-"./example/output/diffold_3owzA"}
CHECKPOINT_PATH=${3:-"./checkpoints/diffold_best.pt"}

# 检查输入文件是否存在
if [ ! -f "$INPUT_FAS" ]; then
    echo "错误: 输入FASTA文件不存在: $INPUT_FAS"
    exit 1
fi

# 检查检查点文件是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "警告: 检查点文件不存在: $CHECKPOINT_PATH"
    echo "将使用预训练模型进行推理"
    CHECKPOINT_PATH=""
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "开始Diffold推理..."
echo "输入文件: $INPUT_FAS"
echo "输出目录: $OUTPUT_DIR"
echo "检查点: $CHECKPOINT_PATH"

# 运行推理
python inference_diffold.py \
    --input_fas "$INPUT_FAS" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --relax_steps 1000 \
    --device cuda:0

echo "推理完成！"
echo "输出文件位于: $OUTPUT_DIR" 