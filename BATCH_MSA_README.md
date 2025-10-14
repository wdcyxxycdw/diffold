# 批量MSA构建功能

## ✅ 已完成的更新

已成功为 `scripts/query_msa.py` 添加批量处理功能！

## 🎯 新增功能

### 1. 批量处理模式
- 可以一次处理整个文件夹中的所有FASTA文件
- 支持 `.fasta`, `.fa`, `.fna` 三种扩展名
- 自动创建输出目录
- 显示实时进度

### 2. 智能跳过功能
- 使用 `--skip_existing` 可以跳过已处理的文件
- 支持断点续传，处理大批量数据时非常有用

### 3. 灵活的输出配置
- 可自定义输出文件后缀（默认 `.a3m`）
- 保持原文件名，只改变扩展名

## 📝 使用方法

### 基本用法（本地BLAST）

```bash
uv run python scripts/query_msa.py \
  --input_dir ./input_fasta_folder \
  --output_a3m ./output_msa_folder \
  --database_dpath ./database \
  --binary_dpath ./rhofold/data/bin \
  --n_cpu 16
```

### 批量处理 + 跳过已存在

```bash
uv run python scripts/query_msa.py \
  --input_dir ./input_fasta_folder \
  --output_a3m ./output_msa_folder \
  --skip_existing \
  --n_cpu 16
```

### 在线BLAST模式（小批量）

```bash
uv run python scripts/query_msa.py \
  --input_dir ./input_fasta_folder \
  --output_a3m ./output_msa_folder \
  --online \
  --email your@email.com \
  --skip_existing
```

## 📊 输出示例

```
================================================================================
批量MSA构建
================================================================================
输入目录: ./input_fasta_folder
输出目录: ./output_msa_folder
找到 150 个FASTA文件
模式: 本地BLAST
跳过已存在的文件: 是
================================================================================

[1/150] 处理中: sequence1.fasta ... ✓ 完成 -> sequence1.a3m
[2/150] 处理中: sequence2.fasta ... ✓ 完成 -> sequence2.a3m
[3/150] ⊘ 跳过 sequence3.fasta (已存在)
[4/150] 处理中: sequence4.fasta ... ✓ 完成 -> sequence4.a3m
...

================================================================================
批量处理完成
================================================================================
总计: 150 个文件
成功: 147 个
跳过: 3 个
失败: 0 个
================================================================================
```

## 🧪 测试

我们已经创建了测试数据：

```bash
# 查看测试文件
ls -lh test_batch_msa/input/

# 输出:
# seq1.fasta
# seq2.fasta
# seq3.fa
```

运行测试（需要email）：

```bash
uv run python scripts/query_msa.py \
  --input_dir ./test_batch_msa/input \
  --output_a3m ./test_batch_msa/output \
  --online \
  --email your@email.com
```

## 📚 新增参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--input_dir` | 字符串 | 输入FASTA文件夹（批量模式） | - |
| `--output_suffix` | 字符串 | 批量模式输出后缀 | `.a3m` |
| `--skip_existing` | 标志 | 跳过已存在的输出文件 | False |

## 🔄 工作流程

```
输入文件夹
├── seq1.fasta
├── seq2.fasta
└── seq3.fa
    ↓
  批量处理
    ↓
输出文件夹
├── seq1.a3m
├── seq2.a3m
└── seq3.a3m
```

## ⚠️ 注意事项

1. **数据库准备**：本地模式需要先运行 `./database/bin/builddb.sh` 构建数据库
2. **在线模式限制**：在线BLAST速度较慢，适合小批量（<20个文件）
3. **单序列要求**：每个FASTA文件应只包含一条序列
4. **断点续传**：使用 `--skip_existing` 可以安全地重新运行命令

## 📦 相关文件

- `scripts/query_msa.py` - 主程序（已更新）
- `scripts/query_msa_batch_usage.md` - 详细使用文档
- `scripts/example_batch_msa.sh` - 使用示例脚本
- `test_batch_msa/` - 测试数据目录

## 🚀 典型使用场景

### 场景1: 处理大量RNA序列

```bash
# 第一次运行
uv run python scripts/query_msa.py \
  --input_dir ./rna_sequences \
  --output_a3m ./msa_output \
  --n_cpu 32

# 如果中断了，可以继续
uv run python scripts/query_msa.py \
  --input_dir ./rna_sequences \
  --output_a3m ./msa_output \
  --n_cpu 32 \
  --skip_existing
```

### 场景2: 处理RNAdata文件夹

```bash
# 假设你已经从PDB文件提取了FASTA序列
uv run python scripts/query_msa.py \
  --input_dir ./RNAdata_fasta \
  --output_a3m ./processed_data/rMSA \
  --database_dpath ./database \
  --n_cpu 16 \
  --skip_existing
```

## ✨ 原有功能仍然保留

单文件模式依然可用：

```bash
# 使用序列字符串
uv run python scripts/query_msa.py \
  --sequence "GGCGCGUUAACGCGUA" \
  --output_a3m output.a3m

# 使用单个FASTA文件
uv run python scripts/query_msa.py \
  --input_fasta input.fasta \
  --output_a3m output.a3m
```

## 🎉 总结

现在你可以高效地批量处理大量RNA序列来构建MSA文件了！使用 `--skip_existing` 参数可以实现断点续传，非常适合处理大型数据集。


