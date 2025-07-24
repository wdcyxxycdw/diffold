# RNA结构评估指标系统 - 纯RNA专用版本

## 概述

本系统专门为RNA结构预测任务设计，删除了所有蛋白质相关代码，提供简洁高效的RNA专用评估指标。

## 主要特性

### ✅ 纯RNA专用设计
- **RNAEvaluationMetrics类**: 专门为RNA设计，无任何蛋白质相关代码
- **简洁的API**: 无需指定结构类型，默认就是RNA
- **高效实现**: 删除了条件判断，直接使用RNA优化参数

### ✅ 删除的内容
- ❌ **GDT-TS/GDT-HA指标** - 不适合RNA结构
- ❌ **蛋白质参数选项** - 所有分支逻辑
- ❌ **结构类型判断** - 简化为纯RNA

### ✅ RNA专用指标

#### 1. RNA TM-score
**基于张雄组RNA-align方法**
```python
# RNA专用d0公式
if L > 30:
    d0 = 1.24 * ((L - 15) ** (1/3)) - 1.8 + 0.3  # +0.3Å偏移
else:
    d0 = 0.5 + 0.3 * (L / 30.0)  # 短序列平滑过渡
```

**质量阈值**:
- ≥ 0.45: 同一Rfam家族相似性
- ≥ 0.6: 优秀预测质量

#### 2. RNA lDDT
**RNA专用参数**:
```python
cutoff_distances = [1.0, 2.0, 4.0, 8.0]  # Å
inclusion_radius = 20.0  # Å
```

**质量阈值**:
- ≥ 70: 高质量预测
- ≥ 50: 良好质量预测

#### 3. RNA Clash Score
**RNA专用参数**:
```python
clash_threshold = 2.5  # Å
```

**质量阈值**:
- ≤ 5%: 低冲突，物理合理

## 使用方法

### 基本使用
```python
from diffold.advanced_optimizers import RNAEvaluationMetrics

# 创建RNA评估器（无需参数）
evaluator = RNAEvaluationMetrics()

# 更新指标
evaluator.update(
    loss=loss_value,
    batch_size=batch_size,
    predicted_coords=predicted_coords,  # [batch_size, n_atoms, 3]
    target_coords=target_coords,        # [batch_size, n_atoms, 3]
    confidence_scores=confidence_scores # 可选
)

# 获取指标
metrics = evaluator.compute_metrics()
```

### 训练中使用
```python
# 训练脚本会自动使用RNAEvaluationMetrics
from diffold.advanced_optimizers import RNAEvaluationMetrics
self.enhanced_metrics = {
    'train': RNAEvaluationMetrics(),
    'val': RNAEvaluationMetrics()
}
```

## 训练集成

### 自动日志输出
```
🧬 RNA结构评估: RMSD=2.341Å TM-score=0.678(85.2%≥0.45) lDDT=73.2(78.5%≥70) Clash=3.2%
```

### TensorBoard监控
- `RNA_Metrics/RMSD`
- `RNA_Metrics/TM_Score`
- `RNA_Metrics/TM_Score_Good_Ratio`
- `RNA_Metrics/lDDT`
- `RNA_Metrics/lDDT_High_Quality_Ratio`
- `RNA_Metrics/Clash_Score`

## 指标详解

### 1. RMSD (Root Mean Square Deviation)
- **单位**: Å
- **范围**: [0, +∞)
- **最优**: 越低越好
- **用途**: 基础几何偏差测量

### 2. RNA TM-score
- **单位**: 无量纲
- **范围**: (0, 1]
- **最优**: 越高越好
- **用途**: RNA结构全局相似性
- **特色**: RNA专用d0归一化公式

### 3. RNA lDDT
- **单位**: 0-100分
- **范围**: [0, 100]
- **最优**: 越高越好
- **用途**: 局部结构质量评估
- **特色**: 大inclusion radius (20Å)

### 4. RNA Clash Score
- **单位**: 百分比
- **范围**: [0, 100]
- **最优**: 越低越好
- **用途**: 物理合理性检查
- **特色**: RNA优化阈值 (2.5Å)

## 性能优势

### 计算效率
- **无条件判断**: 直接使用RNA参数，提高执行效率
- **简化逻辑**: 删除所有蛋白质分支，降低复杂度
- **内存优化**: 适合RNA序列长度（通常50-200核苷酸）

### 代码维护性
- **单一职责**: 专门服务RNA结构预测
- **清晰简洁**: 无多余代码和参数
- **易于扩展**: 可方便添加新的RNA特异性指标

## 向后兼容

```python
# 保持向后兼容性
EvaluationMetrics = RNAEvaluationMetrics

# 旧代码仍然有效
from diffold.advanced_optimizers import EvaluationMetrics
evaluator = EvaluationMetrics()  # 实际使用RNAEvaluationMetrics
```

## 文件变更

### 核心文件
1. `diffold/advanced_optimizers.py`
   - 新增: `RNAEvaluationMetrics`类
   - 删除: 所有蛋白质相关条件判断
   - 优化: 直接使用RNA专用参数

2. `train.py`
   - 更新: 直接使用`RNAEvaluationMetrics()`
   - 优化: 更清晰的RNA专用日志

3. `diffold/example_metrics_usage.py`
   - 重构: 纯RNA示例代码
   - 新增: RNA专用特性展示

## 质量保证

### 科学依据
- **TM-score**: 基于Zhang Lab RNA-align论文
- **lDDT**: 针对RNA结构特性调整
- **参数选择**: 基于RNA结构数据库统计

### 测试覆盖
- **边界情况**: 短序列、长序列处理
- **异常处理**: 坐标不匹配、计算失败
- **性能测试**: 典型RNA序列长度

## 使用建议

### 适用场景
✅ **RNA二级结构预测验证**  
✅ **RNA三级结构建模评估**  
✅ **RNA-蛋白质复合物中的RNA部分**  
✅ **核酶活性位点结构质量**  

### 质量标准
- **优秀预测**: TM-score≥0.6, lDDT≥70, Clash≤5%
- **良好预测**: TM-score≥0.45, lDDT≥50, Clash≤10%
- **可接受预测**: TM-score≥0.3, lDDT≥30, Clash≤15%

## 参考文献

1. **RNA-align**: Sha Gong, Chengxin Zhang, Yang Zhang. "RNA-align: quick and accurate alignment of RNA 3D structures based on size-independent TM-scoreRNA." Bioinformatics, 35: 4459-4461 (2019)

2. **TM-score原理**: Zhang Y, Skolnick J. "Scoring function for automated assessment of protein structure template quality." Proteins, 57: 702-710 (2004)

3. **lDDT方法**: Mariani V, Biasini M, Barbato A, Schwede T. "lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests." Bioinformatics, 29: 2722-2728 (2013)

## 总结

本系统提供了专门为RNA结构预测设计的评估指标，删除了所有蛋白质相关代码，实现了：

- 🎯 **专业性**: 100%专门为RNA设计
- ⚡ **高效性**: 无条件判断，执行更快
- 🔬 **科学性**: 基于权威RNA结构评估方法
- 🛠️ **易用性**: 简单API，无需配置参数
- 📈 **完整性**: 从训练到可视化的全流程支持

适合所有RNA结构预测和评估任务！ 