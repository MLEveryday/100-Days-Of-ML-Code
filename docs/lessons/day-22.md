# Day 22：Hoeffding 不等式与样本量

<!-- course-navigation:start -->
**先修导航**：[Day 24：经验风险与泛化风险](day-24.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-22.md)
<!-- course-navigation:end -->

## 学习目标

理解固定假设下偏差界的前提。

## 核心说明

对独立同分布且有界于 [0,1] 的变量，P(|样本均值−期望|≥ε)≤2exp(−2nε²)。样本数增加时，上界下降；界往往保守，不等于实际错误概率。

机器学习中在同一数据上挑很多假设会引入选择效应，不能把固定假设的界直接套到任意选出的模型。有限 M 个候选可结合 union bound 得到 2Mexp(−2nε²)，更复杂假设集需要进一步理论。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
rng = np.random.default_rng(0)
n, epsilon = 200, 0.1
means = rng.binomial(n, 0.5, size=5000)/n
print("empirical tail", np.mean(np.abs(means-0.5) >= epsilon))
print("bound", min(1., 2*np.exp(-2*n*epsilon**2)))
```

## 结果解读

模拟值通常低于上界；有限重复会产生抽样波动，不构成证明。

## 练习

解出让上界≤0.05 所需的 n；讨论相关样本为何不满足简单独立假设。

[返回完整课程目录](../curriculum.md)
