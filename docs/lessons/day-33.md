# Day 33：随机森林：抽样、特征与集成

<!-- course-navigation:start -->
**先修导航**：[Day 23：决策树：熵、分裂与 CART](day-23.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-33.md)
<!-- course-navigation:end -->

## 学习目标

解释随机性为何有助于降低树间相关性。

## 核心说明

每棵树通过有放回抽样构造训练集；每个节点分裂时再抽取候选特征，并从候选中找较好划分。若所有树完全相同，增加树数不会获得集成收益。

sklearn 分类森林平均各树的类别概率后取最大值，不严格等同每棵树硬投票。OOB 使用某样本未参与训练的树预测该样本，是训练阶段的辅助评价；不能取代所有外部验证。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
rng = np.random.default_rng(0)
indices = rng.integers(0, 100, size=100)
oob = np.setdiff1d(np.arange(100), indices)
print("unique sampled", len(np.unique(indices)), "out of bag", len(oob))
```

## 结果解读

样本数等于抽样次数，但有重复，因此会有未抽到的样本。大样本时未抽到比例约 exp(-1)。

## 练习

重复抽样估计 OOB 比例；解释相关特征为何影响重要性解读。实现见 Day 34。

[返回完整课程目录](../curriculum.md)
