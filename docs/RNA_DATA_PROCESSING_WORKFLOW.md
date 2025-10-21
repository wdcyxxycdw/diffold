# RNA数据处理工作流程

本文档介绍如何从RCSB PDB下载RNA结构并处理成训练格式的完整流程。

## 🔄 完整工作流程

### 步骤1：下载PDB文件

使用 `download_pdb.py` 从RCSB下载原始PDB文件。

#### 方法A：命令行直接指定

```bash
# 下载单个PDB
uv run python scripts/download_pdb.py 1ehz --output ./downloads

# 下载多个PDB
uv run python scripts/download_pdb.py 1ehz 2j01 4v9f --output ./downloads
```

#### 方法B：从文件批量下载

1. 创建PDB列表文件（参考 `scripts/example_pdb_list.txt`）:

```text
# my_pdb_list.txt
1ehz  # tRNA
2j01  # rRNA
4v9f  # ribosome
```

2. 批量下载：

```bash
uv run python scripts/download_pdb.py --from-file my_pdb_list.txt --output ./downloads
```

#### 方法C：交互式输入

```bash
uv run python scripts/download_pdb.py --output ./downloads
# 然后逐个输入PDB ID，输入 q 退出
```

### 步骤2：处理PDB文件为训练格式

使用 `process_pdb_for_training.py` 将下载的PDB文件转换为训练格式。

#### 处理单个文件

```bash
uv run python scripts/process_pdb_for_training.py \
    downloads/pdb/1ehz.pdb \
    --output processed_data/pdb
```

#### 批量处理目录

```bash
uv run python scripts/process_pdb_for_training.py \
    downloads/pdb \
    --output processed_data/pdb
```

#### 递归处理多级目录

```bash
uv run python scripts/process_pdb_for_training.py \
    downloads \
    --output processed_data/pdb \
    --recursive
```

## 📋 完整示例：从零开始

假设你要下载并处理一批RNA结构：

### 1. 准备PDB列表

创建 `my_targets.txt`:
```text
1ehz
2j01
4v9f
1asy
6tna
```

### 2. 下载PDB文件

```bash
uv run python scripts/download_pdb.py \
    --from-file my_targets.txt \
    --output ./raw_pdb_data
```

输出结构：
```
raw_pdb_data/
└── pdb/
    ├── 1ehz.pdb
    ├── 2j01.pdb
    ├── 4v9f.pdb
    ├── 1asy.pdb
    └── 6tna.pdb
```

### 3. 处理为训练格式

```bash
uv run python scripts/process_pdb_for_training.py \
    ./raw_pdb_data/pdb \
    --output ./processed_data/pdb
```

输出结构：
```
processed_data/
└── pdb/
    ├── 1ehz_A.pdb  # 每条链单独保存
    ├── 2j01_A.pdb
    ├── 2j01_B.pdb  # 如果有多条链
    ├── 4v9f_0.pdb
    ├── 4v9f_1.pdb
    ├── ...
```

### 4. 验证处理结果

```bash
# 检查文件数量
ls -1 processed_data/pdb/*.pdb | wc -l

# 查看某个文件的内容
head -n 20 processed_data/pdb/1ehz_A.pdb

# 检查文件格式（应该只包含ATOM记录）
grep -v "^ATOM" processed_data/pdb/1ehz_A.pdb | head
```

## 🔧 高级选项

### 下载选项

```bash
# 下载CIF格式而不是PDB格式
uv run python scripts/download_pdb.py 1ehz --format cif --output ./downloads

# 同时下载FASTA序列文件
uv run python scripts/download_pdb.py 1ehz --with-fasta --output ./downloads

# 覆盖已存在的文件
uv run python scripts/download_pdb.py 1ehz --force --output ./downloads

# 静默模式
uv run python scripts/download_pdb.py 1ehz --quiet --output ./downloads
```

### 处理选项

```bash
# 保留原始的原子和残基编号（不重新编号）
uv run python scripts/process_pdb_for_training.py \
    input.pdb \
    --output ./processed \
    --no-renumber

# 设置最小原子数阈值（跳过太小的链）
uv run python scripts/process_pdb_for_training.py \
    input.pdb \
    --output ./processed \
    --min-atoms 50

# 调试模式（显示详细信息）
uv run python scripts/process_pdb_for_training.py \
    input.pdb \
    --output ./processed \
    --debug
```

## 📊 数据格式说明

### 原始PDB格式（从RCSB下载）

```pdb
HEADER    RNA                                     23-FEB-00   1EHZ
TITLE     THE CRYSTAL STRUCTURE OF YEAST PHENYLALANINE TRNA
COMPND    MOL_ID: 1;
... [大量元数据] ...
ATOM      1  O5'   G A   1      -9.332  10.937   7.364  1.00 0.00           O
ATOM      2  C5'   G A   1      -7.952  10.664   7.173  1.00 0.00           C
... [所有链的原子坐标] ...
```

**特点**：
- 包含完整的文件头和元数据
- 包含所有链（RNA + 蛋白质 + 配体等）
- 适合阅读和文献查阅

### 训练格式（处理后）

```pdb
ATOM      1  O5'   G 0   1      -9.332  10.937   7.364  1.00  0.00           O
ATOM      2  C5'   G 0   1      -7.952  10.664   7.173  1.00  0.00           C
ATOM      3  C4'   G 0   1      -7.547   9.375   7.839  1.00  0.00           C
... [只有RNA原子坐标] ...
```

**特点**：
- 只包含ATOM/HETATM记录
- 只包含RNA残基
- 每个文件对应一条链
- 原子和残基序号重新编号（从1开始）
- 适合机器学习训练和推理

## 🧬 RNA残基识别

脚本自动识别以下RNA残基：

### 标准碱基
- A, C, G, U

### 常见修饰残基
- PSU (假尿嘧啶)
- I (次黄嘌呤)
- M2G, 1MA, 7MG (甲基化)
- 以及100+种其他修饰残基

### 自动排除
- 蛋白质残基 (ALA, GLY, SER等)
- DNA残基 (DA, DC, DG, DT)
- 配体和溶剂分子

## ⚠️ 注意事项

### 1. 网络问题
- 下载大批量PDB时可能遇到网络超时
- 脚本有自动重试机制（最多3次）
- 如果下载失败，可以重新运行（会跳过已存在的文件）

### 2. 多链结构
- 一些PDB文件包含多条RNA链
- 处理脚本会自动分离每条链
- 输出文件命名为 `pdbID_chainID.pdb`

### 3. 混合结构
- 包含RNA和蛋白质的复合物结构
- 处理脚本会**自动过滤蛋白质**，只保留RNA
- 例如核糖体结构（RNA + r-蛋白）

### 4. 最小原子数
- 默认跳过少于10个原子的链（通常是小分子或离子）
- 可以通过 `--min-atoms` 调整阈值

## 🚀 常见使用场景

### 场景1：扩充训练数据集

```bash
# 1. 准备新的PDB列表
echo "7qr4" > new_structures.txt
echo "7qr3" >> new_structures.txt

# 2. 下载
uv run python scripts/download_pdb.py \
    --from-file new_structures.txt \
    --output ./downloads

# 3. 处理并添加到现有数据集
uv run python scripts/process_pdb_for_training.py \
    ./downloads/pdb \
    --output ./processed_data/pdb

# 4. 验证
ls -lh processed_data/pdb/7qr*
```

### 场景2：下载基准测试数据

```bash
# 下载CASP15的RNA targets
cat > casp15_targets.txt << EOF
7qr3
7qr4
# ... 其他targets
EOF

# 下载并处理
uv run python scripts/download_pdb.py \
    --from-file casp15_targets.txt \
    --output ./benchmark_raw

uv run python scripts/process_pdb_for_training.py \
    ./benchmark_raw/pdb \
    --output ./benchmark_data/casp15/pdb
```

### 场景3：快速获取单个结构

```bash
# 下载并处理一步到位
uv run python scripts/download_pdb.py 1ehz --output ./temp_download
uv run python scripts/process_pdb_for_training.py \
    ./temp_download/pdb/1ehz.pdb \
    --output ./processed_data/pdb
rm -rf ./temp_download
```

## 📚 相关资源

- **RCSB PDB**: https://www.rcsb.org/
- **PDB文件格式**: https://www.rcsb.org/docs/general-help/file-format
- **RNA修饰数据库**: http://mods.rna.albany.edu/

## 🐛 故障排除

### 问题1：下载失败
```bash
# 检查网络连接
curl https://files.rcsb.org/download/1ehz.pdb

# 使用--force重新下载
uv run python scripts/download_pdb.py 1ehz --force --output ./downloads
```

### 问题2：没有提取到RNA
```bash
# 使用--debug查看详细信息
uv run python scripts/process_pdb_for_training.py \
    input.pdb \
    --output ./test \
    --debug

# 检查原始PDB文件是否包含RNA
grep "^ATOM.*  A " input.pdb | head
grep "^ATOM.*  G " input.pdb | head
grep "^ATOM.*  C " input.pdb | head
grep "^ATOM.*  U " input.pdb | head
```

### 问题3：输出文件格式不对
```bash
# 检查是否只包含ATOM记录
head -n 50 output.pdb

# 检查是否有非RNA残基
grep "^ATOM" output.pdb | awk '{print $4}' | sort -u
```

## ✅ 检查清单

在使用处理后的数据之前，请确认：

- [ ] 文件只包含ATOM/HETATM记录
- [ ] 文件只包含RNA残基
- [ ] 每个文件对应一条链
- [ ] 文件命名格式为 `pdbID_chainID.pdb`
- [ ] 原子序号从1开始连续编号
- [ ] 残基序号从1开始连续编号
- [ ] 没有HEADER、REMARK等元数据行

## 💬 反馈与支持

如有问题或建议，请：
1. 检查本文档的故障排除部分
2. 使用 `--debug` 模式运行查看详细日志
3. 检查输入PDB文件的格式和内容

