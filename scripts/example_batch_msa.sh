#!/bin/bash
# 批量构建MSA的示例脚本

echo "======================================================================"
echo "批量MSA构建示例"
echo "======================================================================"
echo ""

# 示例1: 本地BLAST批量处理
echo "示例1: 使用本地BLAST批量处理"
echo "----------------------------------------------------------------------"
echo "uv run python scripts/query_msa.py \\"
echo "  --input_dir ./input_fasta_folder \\"
echo "  --output_a3m ./output_msa_folder \\"
echo "  --database_dpath ./database \\"
echo "  --binary_dpath ./rhofold/data/bin \\"
echo "  --n_cpu 16 \\"
echo "  --skip_existing"
echo ""

# 示例2: 在线BLAST批量处理（小批量）
echo "示例2: 使用在线BLAST批量处理（适合小批量）"
echo "----------------------------------------------------------------------"
echo "uv run python scripts/query_msa.py \\"
echo "  --input_dir ./input_fasta_folder \\"
echo "  --output_a3m ./output_msa_folder \\"
echo "  --online \\"
echo "  --email your@email.com \\"
echo "  --skip_existing"
echo ""

# 示例3: 处理RNAdata文件夹（如果已转换为FASTA）
echo "示例3: 处理实际数据"
echo "----------------------------------------------------------------------"
echo "# 假设你已经有一个包含FASTA文件的文件夹"
echo "uv run python scripts/query_msa.py \\"
echo "  --input_dir ./fasta_sequences \\"
echo "  --output_a3m ./processed_data/rMSA \\"
echo "  --database_dpath ./database \\"
echo "  --binary_dpath ./rhofold/data/bin \\"
echo "  --n_cpu 16 \\"
echo "  --skip_existing"
echo ""

# 测试示例
echo "======================================================================"
echo "测试示例（使用测试数据）"
echo "======================================================================"
echo ""
echo "我们已经创建了测试数据在 test_batch_msa/input/ 中"
echo "包含3个测试FASTA文件"
echo ""
ls -lh test_batch_msa/input/
echo ""
echo "要测试批量处理，你可以运行:"
echo ""
echo "uv run python scripts/query_msa.py \\"
echo "  --input_dir ./test_batch_msa/input \\"
echo "  --output_a3m ./test_batch_msa/output \\"
echo "  --online \\"
echo "  --email your@email.com"
echo ""
echo "注意: 在线模式需要提供有效的email地址"
echo ""


