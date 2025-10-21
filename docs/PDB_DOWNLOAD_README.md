# PDB文件下载工具

## 简介

`download_pdb.py` 是一个简单易用的PDB结构文件下载工具，支持从RCSB PDB数据库下载指定的蛋白质/RNA结构文件。

## 功能特点

- ✅ 支持命令行参数输入PDB ID
- ✅ 支持从文件批量读取PDB ID列表
- ✅ 支持交互式输入模式
- ✅ 支持PDB和CIF两种格式
- ✅ 可选下载FASTA序列文件
- ✅ 自动重试机制（最多3次）
- ✅ 跳过已存在的文件（可覆盖）
- ✅ 详细的日志输出

## 使用方法

### 基本用法

#### 1. 下载单个PDB文件

```bash
python scripts/download_pdb.py 1abc
```

#### 2. 下载多个PDB文件

```bash
python scripts/download_pdb.py 1abc 2def 3ghi
```

#### 3. 从文件批量下载

```bash
python scripts/download_pdb.py --from-file scripts/example_pdb_list.txt
```

#### 4. 交互式模式

不提供任何PDB ID时，自动进入交互式模式：

```bash
python scripts/download_pdb.py
```

然后输入PDB ID（支持空格或逗号分隔）：
```
PDB ID: 1abc 2def, 3ghi
```

输入 `q` 或 `quit` 退出。

### 高级选项

#### 指定输出目录

```bash
python scripts/download_pdb.py 1abc --output ./my_structures
```

#### 下载CIF格式

```bash
python scripts/download_pdb.py 1abc --format cif
```

#### 同时下载序列文件

```bash
python scripts/download_pdb.py 1abc --with-fasta
```

#### 覆盖已存在的文件

```bash
python scripts/download_pdb.py 1abc --force
```

#### 静默模式（只显示错误）

```bash
python scripts/download_pdb.py 1abc --quiet
```

### 组合使用

```bash
# 批量下载CIF格式，同时下载序列，保存到指定目录
python scripts/download_pdb.py --from-file pdb_list.txt \
    --format cif \
    --with-fasta \
    --output ./my_data
```

## PDB列表文件格式

创建一个文本文件，每行一个PDB ID，支持注释：

```text
# 这是注释行
1abc
2def  # 这是行内注释
3ghi

# 空行会被忽略
4jkl
```

参考示例文件：`scripts/example_pdb_list.txt`

## 输出目录结构

默认输出到 `./pdb_downloads/` 目录：

```
pdb_downloads/
├── pdb/          # PDB或CIF结构文件
│   ├── 1abc.pdb
│   ├── 2def.pdb
│   └── 3ghi.pdb
└── fasta/        # FASTA序列文件（如果使用 --with-fasta）
    ├── 1abc.fasta
    ├── 2def.fasta
    └── 3ghi.fasta
```

## 错误处理

- 自动重试机制（网络错误时最多重试3次）
- 404错误时跳过该文件并继续
- 已存在的文件默认跳过（使用 `--force` 覆盖）
- 详细的错误日志输出

## 命令行参数完整列表

```
positional arguments:
  pdb_ids               PDB ID（如 1abc 2def）。如果不提供，则进入交互式模式

optional arguments:
  -h, --help            显示帮助信息
  --from-file FILE, -f FILE
                        从文件读取PDB ID列表
  --output DIR, -o DIR  输出目录（默认: ./pdb_downloads）
  --format {pdb,cif}    文件格式（默认: pdb）
  --with-fasta          同时下载FASTA序列文件
  --force               覆盖已存在的文件
  --quiet, -q           静默模式（只显示错误）
```

## 常见使用场景

### 场景1：快速下载几个结构

```bash
python scripts/download_pdb.py 1ehz 2j01 4v9f
```

### 场景2：批量下载课题组需要的结构

1. 创建列表文件 `my_structures.txt`：
```text
1ehz  # 16S rRNA
2j01  # 23S rRNA
4v9f  # 70S ribosome
```

2. 执行下载：
```bash
python scripts/download_pdb.py --from-file my_structures.txt
```

### 场景3：为分析准备数据

```bash
# 下载结构和序列到特定目录
python scripts/download_pdb.py --from-file targets.txt \
    --with-fasta \
    --output ./analysis_data
```

### 场景4：下载实验中需要的参考结构

```bash
# 交互式添加
python scripts/download_pdb.py

# 然后逐个或批量输入PDB ID
PDB ID: 1abc
PDB ID: 2def 3ghi
PDB ID: quit
```

## 注意事项

1. PDB ID通常是4个字符（如 `1abc`, `7d4f`）
2. 下载的文件会自动转为小写文件名
3. 网络连接问题会自动重试，请耐心等待
4. 大批量下载时请注意RCSB服务器的访问限制
5. 推荐使用 `--with-fasta` 同时下载序列信息，方便后续分析

## 技术支持

如有问题，请检查：
1. 网络连接是否正常
2. PDB ID是否正确
3. RCSB PDB网站是否可访问：https://www.rcsb.org/
4. 输出目录是否有写入权限

## 相关资源

- RCSB PDB官网：https://www.rcsb.org/
- PDB文件格式说明：https://www.rcsb.org/docs/general-help/file-format
- CIF文件格式说明：https://mmcif.wwpdb.org/

