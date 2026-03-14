# 数据泄露分析日志

## 2026-03-14

### 分析目标

判断当前项目中“预测性能非常好”的说法，究竟来自真实泛化能力，还是受到训练集-测试集数据泄露或近重复样本污染。

### 当前使用的证据文件

- [benchmark_data/RNA-benchmark/tm_scores_analysis_default.tsv](../benchmark_data/RNA-benchmark/tm_scores_analysis_default.tsv)
- [benchmark_data/RNA-benchmark/train_test_similarity_blast.tsv](../benchmark_data/RNA-benchmark/train_test_similarity_blast.tsv)
- [benchmark_data/RNA-benchmark/train_test_similarity_needle.tsv](../benchmark_data/RNA-benchmark/train_test_similarity_needle.tsv)
- [benchmark_data/RNA-benchmark/train_test_similarity_mmseq2.tsv](../benchmark_data/RNA-benchmark/train_test_similarity_mmseq2.tsv)
- [benchmark_data/casp16/tm_scores_analysis_default.tsv](../benchmark_data/casp16/tm_scores_analysis_default.tsv)
- [benchmark_data/casp16/train_test_similarity_blast.tsv](../benchmark_data/casp16/train_test_similarity_blast.tsv)
- [benchmark_data/casp16/train_test_similarity_needle.tsv](../benchmark_data/casp16/train_test_similarity_needle.tsv)
- [benchmark_data/casp16/train_test_similarity_mmseq2.tsv](../benchmark_data/casp16/train_test_similarity_mmseq2.tsv)
- [benchmark_data/casp15/tm_scores_analysis_default.tsv](../benchmark_data/casp15/tm_scores_analysis_default.tsv)
- [diffold/docs/CV_SPLITS_SUMMARY.md](../diffold/docs/CV_SPLITS_SUMMARY.md)
- [scripts/compute_structure_similarity_tmscore.py](../scripts/compute_structure_similarity_tmscore.py)
- [scripts/check_data_leakage.py](../scripts/check_data_leakage.py)

补充说明：

- 本文档优先引用仓库内可提交文件。
- 对本地未纳入版本管理的训练数据或评测产物，仅保留相对项目根目录的路径说明，不写死机器绝对路径。

### 当前事实

1. 项目文档说明内部交叉验证划分使用了 CD-HIT `80%` 序列相似度阈值，目标是避免相似序列落入不同折。[引文 1]
2. 这只能说明“训练/验证折划分试图防止序列级泄露”，不能直接证明外部 benchmark 干净。
3. 结构相似度脚本中明确写明：
   - `TM-score >= 0.5` 通常可视为相同折叠类型 [引文 2]
   - `TM-score >= 0.3` 可能存在拓扑相似 [引文 2]
4. 本分析中出现的 `effective_identity` 指标采用项目脚本中的内部启发式定义：
   - `effective_identity = pident * qcov / 100` [引文 3]
   - 它用于排序和风险分层，不应误写成通行标准文献指标。
5. 真实训练数据目录存在于：
   - `processed_data/`（本地目录，未纳入版本管理）
   - 其中包含真实的 `list/`、`pdb/`、`sequences/` 等目录。
6. 当前仓库默认训练配置使用：
   - `data_dir = ./processed_data`
   - `fold = 3`
   - 见 [config.yaml](../config.yaml#L4)

### 关于“这些是不是训练样本”的证据边界

目前可以区分三件事：

1. **确定属于 processed_data 总库**
   - 例如 `7uga_A`、`4xw7_A`、`7jjd_A`、`7da7_C` 都真实存在于外部 `processed_data/pdb` 与 `processed_data/sequences` 中。
2. **确定属于 fold-3 训练列表**
   - `7uga_A`
   - `4xw7_A`
   - `4rzd_A`
   - 以及若干中风险邻居如 `3w3s_B`、`6xzb_g2`、`1asy_S`、`6zto_AX`、`6jxm_B`、`6wzr_A`、`4znp_A`
3. **不属于 fold-3 训练列表，但属于 processed_data 其他折的训练成员**
   - `7jjd_A`
   - `7da7_C`
   - 这两个在 `fold-3` 中是 `valid_fold-3`，但在其余多个折里会出现在训练集

因此：

- 对 `fold = 3` 这个默认训练配置，可以说 `7uga_A` 和 `4xw7_A` 确实是训练样本。
- 但不能把所有“最相似训练样本”都一概说成 `fold-3` 训练成员。
- 更准确的说法应是：
  - 它们至少属于 `processed_data` 总训练参考库中的成员
  - 其中一部分已确认属于 `fold-3` 的实际训练列表
  - 另一部分只确认属于其他折的训练集合或 `fold-3` 的验证集合

### 当前统计结论

#### RNA-benchmark

- 样本数: 43
- 平均最大训练集结构相似度: 0.4132
- `TM >= 0.5`: 9/43 (20.9%)
- `TM >= 0.7`: 4/43 (9.3%)
- `TM >= 0.9`: 3/43 (7.0%)

#### CASP16

- 样本数: 14
- 平均最大训练集结构相似度: 0.3954
- `TM >= 0.5`: 4/14 (28.6%)
- `TM >= 0.7`: 1/14 (7.1%)
- `TM >= 0.9`: 1/14 (7.1%)

#### CASP15

- 样本数: 6
- 平均最大训练集结构相似度: 0.4363
- `TM >= 0.5`: 2/6 (33.3%)
- `TM >= 0.7`: 0/6

### 已确认的高风险样本

#### 1. RNA-benchmark: `7UGA`

- 训练集最大结构相似度 `TM-score = 1.0000`
- 最相似训练样本为 `7uga_A`
- 证据文件:
  - [benchmark_data/RNA-benchmark/tm_scores_analysis_default.tsv](../benchmark_data/RNA-benchmark/tm_scores_analysis_default.tsv#L7)
  - [benchmark_data/RNA-benchmark/train_test_similarity_blast.tsv](../benchmark_data/RNA-benchmark/train_test_similarity_blast.tsv#L6)
- 备注:
  - BLAST 结果显示全长 `43/43` 命中，`100%` identity
  - `7uga_A` 已确认存在于真实 `processed_data/list/fold-3_train_ids`
  - 这已经非常接近“同一序列直接进入训练集”的证据

#### 2. CASP16: `R1263`

- 最大结构相似度 `TM-score = 0.9299`
- 最相似训练样本为 `4xw7_A`
- Needle 结果为全长 `100%` identity
- MMseq2 结果 `effective_identity = 1.00` [引文 3]
- 证据文件:
  - [benchmark_data/casp16/tm_scores_analysis_default.tsv](../benchmark_data/casp16/tm_scores_analysis_default.tsv#L10)
  - [benchmark_data/casp16/train_test_similarity_needle.tsv](../benchmark_data/casp16/train_test_similarity_needle.tsv#L10)
  - [benchmark_data/casp16/train_test_similarity_mmseq2.tsv](../benchmark_data/casp16/train_test_similarity_mmseq2.tsv#L10)
- 备注:
  - `4xw7_A` 已确认存在于真实 `processed_data/list/fold-3_train_ids`
  - 这属于当前最强的数据泄露证据之一

### 中等风险样本

#### CASP16

- `R1203`
  - Needle identity `65.6%`
  - 结构最大 TM-score `0.5410`
- `R1211`
  - MMseq2 `effective_identity = 0.44` [引文 3]
  - 结构最大 TM-score `0.4429`
- `R1261`
  - 结构最大 TM-score `0.5950`
- `R1262`
  - 结构最大 TM-score `0.5899`

这些样本不一定是“直接泄露”，但已经足以说明测试集并不完全远离训练集分布。

### 方法学问题

#### 1. RNA-benchmark 的序列相似度表互相矛盾

- `BLAST` 表中有大量非零命中，且存在全长 `100%` 命中
- `Needle` 表全部为 `0`
- `MMseq2` 表也全部为 `0`

这说明当前 RNA-benchmark 的序列泄露分析流程存在不一致，暂时不能拿任何单一表格直接当作最终事实。

#### 2. 样本数量不一致

- `RNA-benchmark/tm_scores_analysis_default.tsv`: 43 条
- `RNA-benchmark/train_test_similarity_*.tsv`: 40 条

这说明至少有 3 个样本没有进入当前序列相似度表，后续需要补齐映射关系。

### 风险分层规则（当前工作版）

为便于后续做 clean benchmark，本轮先使用保守且可执行的 4 级分层：

- `high`
  - `TM-score >= 0.9`
  - 或全长精确序列命中
  - 或 `effective_identity >= 0.8` [引文 3]
- `medium`
  - `0.5 <= TM-score < 0.9`
  - 或存在明显的中高序列相似性信号
- `low`
  - `0.3 <= TM-score < 0.5`
- `minimal`
  - `TM-score < 0.3` 且没有明显序列重合信号

说明：

- 这是一套“泄露排查规则”，不是论文标准分类法。
- 目的不是给模型打分，而是生成后续 clean benchmark 过滤名单。

完整清单见：

- [docs/DATA_LEAKAGE_RISK_TABLE.tsv](./DATA_LEAKAGE_RISK_TABLE.tsv)
- [docs/DATA_LEAKAGE_FOLD_MEMBERSHIP.tsv](./DATA_LEAKAGE_FOLD_MEMBERSHIP.tsv)

### 10 折归属矩阵结论

基于本地 `processed_data/list` 的真实交叉验证划分，我把所有 `high + medium` 风险样本的最近邻训练样本逐个映射到了 `fold-0` 到 `fold-9`。

矩阵文件见：

- [docs/DATA_LEAKAGE_FOLD_MEMBERSHIP.tsv](./DATA_LEAKAGE_FOLD_MEMBERSHIP.tsv)

从这个矩阵里可以得到更具体的判断：

1. 当前 `high + medium` 风险样本共 `19` 个。
2. 这 `19` 个样本的最近邻都不是“只在总库里存在、但不属于交叉验证划分”的孤立条目。
3. 它们全部都已经进入标准 10 折划分，并且呈现出非常整齐的模式：
   - `9` 个折里属于 `train`
   - `1` 个折里属于 `valid`
   - `0` 个折里属于 `absent`
4. 这说明这些最近邻不是分析脚本偶然扫到的边缘样本，而是项目正式训练/验证池中的常规成员。

对默认配置 `fold = 3`，结论进一步细化为：

- RNA-benchmark 的 `high + medium` 风险样本共 `9` 个
  - 其中 `7` 个最近邻在 `fold-3` 里是 `train`
  - `2` 个最近邻在 `fold-3` 里是 `valid`
  - 这两个例外是 `7UCR -> 7jjd_A` 和 `8FCS -> 7da7_C`
- CASP16 的 `high + medium` 风险样本共 `8` 个
  - `8/8` 的最近邻都在 `fold-3` 里是 `train`
- CASP15 的 `high + medium` 风险样本共 `2` 个
  - `2/2` 的最近邻都在 `fold-3` 里是 `train`

这意味着：

- 对 `7UGA -> 7uga_A`、`R1263 -> 4xw7_A` 这类关键案例，现在已经不是“像训练集”，而是“其最近邻确实位于默认 `fold-3` 训练列表”。
- 对 `7UCR -> 7jjd_A`、`8FCS -> 7da7_C` 这类样本，最近邻虽然不在 `fold-3` 训练里，但它们仍然是项目 10 折数据池中的正式成员，只是落在 `fold-3` 的验证集合。
- 因此，当前最准确的表述不应再是笼统的“可能有泄露”，而应拆成两类：
  - `fold-3` 直接训练泄露嫌疑
  - 非 `fold-3` 直接训练泄露，但属于同一交叉验证成员池的高近邻污染

### 风险分层摘要

#### RNA-benchmark

- `high`: 3
- `medium`: 6
- `low`: 22
- `minimal`: 12

#### CASP16

- `high`: 1
- `medium`: 7
- `minimal`: 6

#### CASP15

- `medium`: 2
- `low`: 4

### Clean Benchmark 方案

#### 方案 A: Conservative clean

只剔除 `high-risk` 样本，适合先做一轮温和复核。

- RNA-benchmark
  - 保留 `40/43` (93.0%)
  - 剔除: `7UCR`, `7UGA`, `8VPV`
- CASP16
  - 保留 `13/14` (92.9%)
  - 剔除: `R1263`
- CASP15
  - 保留 `6/6` (100%)
  - 不剔除

#### 方案 B: Strict clean

剔除 `high-risk + medium-risk` 样本，适合做更严格的无泄露复核。

- RNA-benchmark
  - 保留 `34/43` (79.1%)
  - 剔除: `7UCR`, `7UGA`, `7UME`, `8FCS`, `8SP9`, `8UPT`, `8V1I`, `8VPV`, `8VT5`
- CASP16
  - 保留 `6/14` (42.9%)
  - 剔除: `R1203`, `R1205`, `R1209`, `R1211`, `R1212`, `R1261`, `R1262`, `R1263`
- CASP15
  - 保留 `4/6` (66.7%)
  - 剔除: `R1108`, `R1116`

解释：

- `Conservative clean` 用来回答“去掉最明显的泄露后，结果还站不站得住”。
- `Strict clean` 用来回答“在明显远离训练集的样本上，模型还剩多少表现”。

### 阶段性判断

当前更合理的结论是：

- 项目里不是所有高分都来自泄露，模型应当存在一定真实泛化能力。
- 但现有 benchmark 中已经发现足以污染整体结论的高风险样本。
- 因此，“性能非常好”这个说法目前不能直接成立，至少应当改写为：
  - 当前结果可能被训练集近重复或直接重复样本抬高
  - 需要在 leak-clean 子集上重新评估

### Clean benchmark 重算结果

为了直接回答“去掉可疑样本之后，性能是否还成立”，我使用了真实逐样本评测结果：

- `results/diffold_casp16+bench/evaluation_results/evaluation_results.csv`
- `results/rhofold_casp16+bench/evaluation_results/evaluation_results.csv`

说明：

- 这两份 `evaluation_results.csv` 是本地评测产物，当前未纳入版本管理，因此这里只保留路径说明，不做仓库内链接。

这两份结果与论文稿中的总体均值一致：

- Diffold: `TM-score = 0.5853`, `RMSD = 2.2932`
- RhoFold+: `TM-score = 0.3091`, `RMSD = 3.8511`

但为了做**公平对比**，重算时我改用两模型都成功输出的共同样本：

- 共 `55` 个样本
- 其中 RNA-benchmark `43` 个
- CASP16 `12` 个
- RhoFold+ 缺失 `R1261` 和 `R1286`

汇总表见：

- [docs/CLEAN_BENCHMARK_RECALC.tsv](./CLEAN_BENCHMARK_RECALC.tsv)
- [docs/CLEAN_BENCHMARK_RECALC_BY_DATASET.tsv](./CLEAN_BENCHMARK_RECALC_BY_DATASET.tsv)

#### 共同样本基线

在共同 `55` 样本上：

- Diffold
  - `TM-score = 0.5704`
  - `RMSD = 2.3649`
  - `GDT-TS = 0.5202`
  - `lDDT = 0.8397`
  - `Clash score = 0.00374`
- RhoFold+
  - `TM-score = 0.3091`
  - `RMSD = 3.8511`
  - `GDT-TS = 0.3132`
  - `lDDT = 0.8062`
  - `Clash score = 0.00787`

#### Conservative clean

只去掉 `high-risk` 样本：

- 去除样本：`7UCR`, `7UGA`, `8VPV`, `R1263`
- 剩余 `51` 个共同样本

重算后：

- Diffold: `TM-score = 0.5845`, `RMSD = 2.3490`
- RhoFold+: `TM-score = 0.2870`, `RMSD = 3.9990`

#### 去掉 fold-3 直接训练泄露嫌疑

去掉所有 `high + medium` 且 `fold_3_status = train` 的样本：

- 去除样本：
  - `7UGA`, `7UME`, `8SP9`, `8UPT`, `8V1I`, `8VPV`, `8VT5`
  - `R1203`, `R1205`, `R1209`, `R1211`, `R1212`, `R1262`, `R1263`
- 剩余 `41` 个共同样本

重算后：

- Diffold: `TM-score = 0.5841`, `RMSD = 2.2873`
- RhoFold+: `TM-score = 0.2557`, `RMSD = 4.1456`

#### Strict clean

去掉全部 `high + medium` 风险样本：

- 额外再去掉 `7UCR`, `8FCS`
- 最终剩余 `39` 个共同样本

重算后：

- Diffold: `TM-score = 0.5889`, `RMSD = 2.3115`
- RhoFold+: `TM-score = 0.2327`, `RMSD = 4.2926`

#### 直接训练泄露样本是否抬高了 Diffold 总分

从共同样本的分组均值看，答案是**没有明显抬高 Diffold，反而更像明显抬高了 RhoFold+**：

- `fold-3` 直接训练泄露嫌疑组（14 个样本）
  - Diffold 平均 `TM-score = 0.5304`
  - RhoFold+ 平均 `TM-score = 0.4657`
- clean 组（39 个样本）
  - Diffold 平均 `TM-score = 0.5889`
  - RhoFold+ 平均 `TM-score = 0.2327`

这表示：

- 对 Diffold 来说，可疑训练近邻样本并没有把整体均值向上“撑高”。
- 对 RhoFold+ 来说，这些可疑近邻样本显著更容易，去掉之后分数下降明显。

#### dataset 维度的重算结论

RNA-benchmark 上，Strict clean 后：

- Diffold: `TM-score = 0.5416`
- RhoFold+: `TM-score = 0.2485`

CASP16 上，去掉 `fold-3` 直接训练泄露嫌疑后只剩 `5` 个共同样本：

- Diffold: `TM-score = 0.9103`
- RhoFold+: `TM-score = 0.1248`

但这里必须注明：

- CASP16 的 clean 子集样本数非常小，统计波动会很大。
- 因此它能支持“Diffold 优势没有因为去泄露而消失”，但还不足以支持非常强的统计宣称。

#### 现阶段更精确的结论

到这一步，结论需要修正为：

- benchmark 中确实存在真实的数据泄露/近邻污染，尤其是 `7UGA` 与 `R1263`
- 但在当前可用结果上，去掉这些可疑样本后，Diffold 的优势并没有塌陷
- 真正对训练近邻更敏感、分数被可疑样本明显抬高的，是 RhoFold+
- 所以“Diffold 全靠数据泄露才显得强”这个说法，目前**不被数据支持**
- 更准确的说法应是：
  - benchmark 本身有污染，原始分数不能直接当作完全无泄露结论
  - 但 Diffold 的主要优势看起来并不依赖这些已识别的泄露样本

### 公式与阈值引用说明

#### 引文 1

内部交叉验证切分文档使用了 `80%` 相似度阈值，并将其作为“避免相似序列分散到不同折”的依据：

- [diffold/docs/CV_SPLITS_SUMMARY.md](../diffold/docs/CV_SPLITS_SUMMARY.md#L75)

说明：

- 这是项目内部切分策略说明，不是外部 benchmark 无泄露的证明。

#### 引文 2

项目的结构相似度分析脚本把 `TM-score >= 0.5` 和 `TM-score >= 0.3` 作为经验阈值：

- [scripts/compute_structure_similarity_tmscore.py](../scripts/compute_structure_similarity_tmscore.py#L335)

指标背景参考：

- [diffold/docs/RNA_METRICS_SUMMARY.md](../diffold/docs/RNA_METRICS_SUMMARY.md#L191)
- [diffold/docs/RNA_METRICS_SUMMARY.md](../diffold/docs/RNA_METRICS_SUMMARY.md#L197)

说明：

- `TM-score >= 0.5` 这里用作“高结构接近性/潜在同折叠”的工程判别阈值。
- 具体任务中的严格判定仍应结合 RNA 场景和 benchmark 定义解释。

#### 引文 3

`effective_identity` 是本项目脚本中的内部组合指标，按覆盖度修正 identity：

- [scripts/compute_sequence_similarity_blast.py](../scripts/compute_sequence_similarity_blast.py#L433)

说明：

- 这个公式是项目内部为了避免“短局部高 identity 命中”而构造的排序指标。
- 它适合作为泄露风险排查的辅助量，不应直接等同于标准数据库报告中的单一 identity 指标。

### 下一步待办

1. 已完成：生成 `high / medium / low / minimal` 风险名单，见 [docs/DATA_LEAKAGE_RISK_TABLE.tsv](./DATA_LEAKAGE_RISK_TABLE.tsv)。
2. 为 RNA-benchmark 补齐样本 ID 映射，解释为什么 TM 表有 43 条而序列表只有 40 条。
3. 用统一口径重新计算 RNA-benchmark 的序列相似度，优先采用全局比对或 coverage 修正后的指标。
4. 如果能找到模型的真实 `evaluation_results.csv`，在 clean 子集上重算最终性能。
5. 同时汇报：
   - 原始 benchmark 分数
   - 去除可疑样本后的 clean benchmark 分数
   - 两者差值

### 工作结论状态

- 当前状态: `存在明确泄露风险，尚不能证明“性能非常好”代表真实泛化`
- 结论置信度: 中等偏高
- 仍需补充:
  - RNA-benchmark 的统一序列相似度复核
  - 模型真实评估结果表
