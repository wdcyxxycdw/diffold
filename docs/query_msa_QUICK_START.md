# query_msa.py 快速开始指南

## 🚀 快速使用

### 批量处理（推荐）

```bash
# 最简单的用法
uv run python scripts/query_msa.py \
  --input_dir ./your_fasta_folder \
  --output_a3m ./your_output_folder

# 推荐用法（支持断点续传）
uv run python scripts/query_msa.py \
  --input_dir ./your_fasta_folder \
  --output_a3m ./your_output_folder \
  --skip_existing \
  --n_cpu 16
```

### 单文件处理

```bash
# 从FASTA文件
uv run python scripts/query_msa.py \
  --input_fasta input.fasta \
  --output_a3m output.a3m

# 从序列字符串
uv run python scripts/query_msa.py \
  --sequence "GGCGCGUUAACGCGUA" \
  --output_a3m output.a3m
```

## 📋 常用参数

| 参数 | 说明 |
|------|------|
| `--input_dir` | 批量模式：输入文件夹 |
| `--input_fasta` | 单文件模式：FASTA文件 |
| `--sequence` | 单文件模式：序列字符串 |
| `--output_a3m` | 输出路径（文件或文件夹） |
| `--skip_existing` | 跳过已存在的文件 ⭐ |
| `--n_cpu` | CPU核心数（默认4） |
| `--online` | 使用在线BLAST |
| `--email` | 在线模式必需的Email |

## 💡 实用技巧

1. **大批量处理**：使用 `--skip_existing` 支持断点续传
2. **多核加速**：设置 `--n_cpu` 为你的CPU核心数
3. **在线模式**：小批量（<10个）可以用 `--online`，无需配置数据库

## 📁 目录结构

```
your_project/
├── input_fasta/          # 你的FASTA文件
│   ├── seq1.fasta
│   ├── seq2.fasta
│   └── seq3.fa
└── output_msa/           # 输出MSA文件（自动创建）
    ├── seq1.a3m
    ├── seq2.a3m
    └── seq3.a3m
```

## ⚡ 示例

查看完整示例：
```bash
bash scripts/example_batch_msa.sh
```

查看详细文档：
```bash
cat BATCH_MSA_README.md
```

