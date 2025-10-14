# 基准数据集下载指南

## 🎯 快速开始

### 1. 下载所有RNA-Puzzles数据

```bash
# 下载所有RNA-Puzzles (24个puzzles)
uv run python scripts/download_benchmark_data.py \
  --rna-puzzles \
  --output ./benchmark_data

# 只下载特定的puzzles
uv run python scripts/download_benchmark_data.py \
  --rna-puzzles \
  --puzzles puzzle1 puzzle2 puzzle5 puzzle10 \
  --output ./benchmark_data
```

### 2. 下载CASP数据

```bash
# 下载CASP15 RNA targets
uv run python scripts/download_benchmark_data.py \
  --casp \
  --casp-version CASP15 \
  --output ./casp_data
```

**注意**: CASP数据需要从官网手动下载或注册API访问
- 官网: https://predictioncenter.org/

### 3. 从自定义PDB列表下载

```bash
# 创建PDB列表文件
cat > my_pdb_list.txt << 'EOF'
1y26
2l8f
3d2v
4p5j
5di1
EOF

# 下载列表中的所有PDB
uv run python scripts/download_benchmark_data.py \
  --pdb-list my_pdb_list.txt \
  --output ./my_benchmark_data
```

## 📊 RNA-Puzzles 数据集

脚本包含24个RNA-Puzzles的PDB映射：

| Puzzle | PDB ID | 描述 |
|--------|--------|------|
| puzzle1 | 2l8f | GlmS riboswitch |
| puzzle2 | 2lc8 | SAH riboswitch |
| puzzle3 | 2lhp | Lysine riboswitch |
| puzzle4 | 2m8k | SAM-I riboswitch |
| puzzle5 | 2n3r | FMN riboswitch |
| puzzle14 | 4k31 | Twister ribozyme |
| puzzle15 | 4nio | Hatchet ribozyme |
| puzzle16 | 4p5j | Pistol ribozyme |
| ... | ... | ... |

## 🔧 高级选项

### 只下载PDB文件（不要FASTA）

```bash
uv run python scripts/download_benchmark_data.py \
  --rna-puzzles \
  --no-fasta \
  --output ./benchmark_data
```

### 只下载FASTA序列（不要PDB）

```bash
uv run python scripts/download_benchmark_data.py \
  --rna-puzzles \
  --no-pdb \
  --output ./benchmark_data
```

### 下载CIF格式而不是PDB格式

```bash
uv run python scripts/download_benchmark_data.py \
  --rna-puzzles \
  --format cif \
  --output ./benchmark_data
```

### 组合多个数据集

```bash
# 同时下载RNA-Puzzles和自定义列表
uv run python scripts/download_benchmark_data.py \
  --rna-puzzles \
  --pdb-list my_list.txt \
  --output ./benchmark_data
```

## 📁 输出目录结构

```
benchmark_data/
├── RNA-Puzzles/
│   ├── pdb/
│   │   ├── 2l8f.pdb
│   │   ├── 2lc8.pdb
│   │   └── ...
│   └── fasta/
│       ├── 2l8f.fasta
│       ├── 2lc8.fasta
│       └── ...
├── CASP15/
│   ├── pdb/
│   └── fasta/
└── custom/
    ├── pdb/
    └── fasta/
```

## 📝 创建自己的PDB列表

```bash
# 方法1: 手动创建
cat > rna_structures.txt << 'EOF'
# RNA结构列表
# 每行一个PDB ID
1y26
2l8f
3d2v
4p5j
5di1
6d90
EOF

# 方法2: 从现有PDB文件夹提取ID
ls RNAdata/*.pdb | sed 's/.*\///;s/_.*//' | sort -u > rna_pdb_list.txt

# 方法3: 使用Python脚本生成
python -c "
pdb_ids = ['1y26', '2l8f', '3d2v', '4p5j', '5di1']
with open('my_list.txt', 'w') as f:
    for pdb_id in pdb_ids:
        f.write(f'{pdb_id}\\n')
"
```

## 🌐 数据来源

### RNA-Puzzles
- 官网: http://rnapuzzles.org/
- 论文: RNA-Puzzles: A CASP-like evaluation of RNA 3D structure prediction

### CASP (Critical Assessment of Structure Prediction)
- 官网: https://predictioncenter.org/
- RNA专项始于CASP15

### RCSB PDB
- 官网: https://www.rcsb.org/
- 所有PDB和FASTA文件从这里下载

## ⚡ 性能提示

1. **并发下载**: 脚本按顺序下载，如需加速可以修改为并发
2. **断点续传**: 重新运行会自动跳过已下载的文件
3. **网络问题**: 脚本自动重试（最多3次）
4. **批量下载**: 对于大量PDB ID，建议使用 `--pdb-list` 方式

## 🔍 验证下载

```bash
# 检查下载的文件数量
echo "PDB文件数: $(find benchmark_data -name '*.pdb' | wc -l)"
echo "FASTA文件数: $(find benchmark_data -name '*.fasta' | wc -l)"

# 检查文件大小
du -sh benchmark_data/*

# 列出所有下载的PDB ID
find benchmark_data -name '*.pdb' -exec basename {} .pdb \; | sort
```

## 🐛 故障排除

### 404 Not Found 错误
某些PDB ID可能不存在或已被废弃：
```bash
# 在RCSB PDB网站上搜索替代结构
# https://www.rcsb.org/
```

### 网络超时
增加重试次数或手动下载：
```bash
# 手动下载单个PDB
wget https://files.rcsb.org/download/2l8f.pdb
wget https://www.rcsb.org/fasta/entry/2l8f -O 2l8f.fasta
```

### 权限问题
确保输出目录有写入权限：
```bash
mkdir -p benchmark_data
chmod 755 benchmark_data
```

## 📚 下一步

下载完数据后，你可以：

1. **提取RNA链序列**:
   ```bash
   # 使用现有工具处理PDB文件
   # 提取特定链的序列
   ```

2. **构建MSA**:
   ```bash
   uv run python scripts/query_msa.py \
     --input_dir benchmark_data/RNA-Puzzles/fasta \
     --output_a3m benchmark_data/RNA-Puzzles/msa
   ```

3. **运行结构预测**:
   ```bash
   # 使用RhoFold或其他工具进行预测
   ```

