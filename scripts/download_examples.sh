#!/bin/bash
# CASP和RNA-Puzzles数据下载示例脚本

echo "================================================================================"
echo "               CASP和RNA-Puzzles数据集下载示例"
echo "================================================================================"
echo ""

# 设置输出目录
OUTPUT_DIR="./benchmark_data"

# 示例1: 下载所有RNA-Puzzles
echo "示例1: 下载所有RNA-Puzzles数据"
echo "--------------------------------------------------------------------------------"
echo "uv run python scripts/download_benchmark_data.py \\"
echo "  --rna-puzzles \\"
echo "  --output $OUTPUT_DIR"
echo ""

# 示例2: 只下载部分puzzles
echo "示例2: 只下载特定的RNA-Puzzles"
echo "--------------------------------------------------------------------------------"
echo "uv run python scripts/download_benchmark_data.py \\"
echo "  --rna-puzzles \\"
echo "  --puzzles puzzle1 puzzle5 puzzle14 puzzle16 \\"
echo "  --output $OUTPUT_DIR"
echo ""

# 示例3: 从预制列表下载
echo "示例3: 从预制的PDB列表下载 (RNA-Puzzles子集)"
echo "--------------------------------------------------------------------------------"
echo "uv run python scripts/download_benchmark_data.py \\"
echo "  --pdb-list scripts/example_pdb_lists/rna_puzzles_subset.txt \\"
echo "  --output $OUTPUT_DIR"
echo ""

# 示例4: 下载CASP相关RNA结构
echo "示例4: 下载CASP相关的RNA结构示例"
echo "--------------------------------------------------------------------------------"
echo "uv run python scripts/download_benchmark_data.py \\"
echo "  --pdb-list scripts/example_pdb_lists/casp_rna_example.txt \\"
echo "  --output $OUTPUT_DIR"
echo ""

# 示例5: 快速测试（3个结构）
echo "示例5: 快速测试（下载3个短RNA结构）"
echo "--------------------------------------------------------------------------------"
echo "uv run python scripts/download_benchmark_data.py \\"
echo "  --pdb-list scripts/example_pdb_lists/short_rna_test.txt \\"
echo "  --output ./test_download"
echo ""

# 示例6: 只下载PDB，不要FASTA
echo "示例6: 只下载PDB结构文件"
echo "--------------------------------------------------------------------------------"
echo "uv run python scripts/download_benchmark_data.py \\"
echo "  --rna-puzzles \\"
echo "  --no-fasta \\"
echo "  --output $OUTPUT_DIR"
echo ""

# 示例7: 下载CIF格式
echo "示例7: 下载mmCIF格式而不是PDB格式"
echo "--------------------------------------------------------------------------------"
echo "uv run python scripts/download_benchmark_data.py \\"
echo "  --rna-puzzles \\"
echo "  --format cif \\"
echo "  --output $OUTPUT_DIR"
echo ""

echo "================================================================================"
echo "                            快速测试"
echo "================================================================================"
echo ""
echo "运行快速测试下载（3个短RNA结构）："
echo ""
echo "  bash scripts/download_examples.sh test"
echo ""
echo "================================================================================"
echo "                         可用的示例PDB列表"
echo "================================================================================"
echo ""
ls -1 scripts/example_pdb_lists/*.txt 2>/dev/null | while read file; do
    echo "  - $(basename $file)"
    head -1 "$file" | grep "^#" | sed 's/^# /    /'
done
echo ""
echo "================================================================================"

# 如果传入参数 "test"，则运行快速测试
if [ "$1" == "test" ]; then
    echo ""
    echo "运行快速测试..."
    echo ""
    uv run python scripts/download_benchmark_data.py \
      --pdb-list scripts/example_pdb_lists/short_rna_test.txt \
      --output ./test_download
fi

