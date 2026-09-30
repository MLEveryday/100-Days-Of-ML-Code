# Day 24：经验风险与泛化风险

<!-- course-navigation:start -->
**先修导航**：无其他课程硬性先修；需能读写基本 Python 表达式。

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-24.md)
<!-- course-navigation:end -->

## 学习目标

区分训练损失、目标分布风险和假设空间。

## 核心说明

输入空间 X 是可能输入的集合，输出空间 Y 是目标集合，假设空间 H 是允许选择的函数。经验风险是训练样本损失平均值；泛化风险是目标分布上的期望损失，通常不可直接精确计算。

最小化经验风险可能过拟合。正则化、验证和限制复杂度用于平衡拟合与泛化。有限假设集的界依赖候选数；VC 维等概念描述更一般的容量，不应把参数数量机械等同 VC 维。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
y = np.array([0, 0, 1, 1])
predictions = {"constant": np.zeros(4, dtype=int), "memorized": y.copy()}
for name, pred in predictions.items():
    print(name, "training 0-1 risk", np.mean(pred != y))
```

## 结果解读

记忆训练答案得到零经验风险，却没有定义新样本预测行为，因此不能推出泛化优秀。

## 练习

以 Day 25 为例写出输入空间、假设空间和损失。延伸阅读 [Bloomberg ML 课程](https://bloomberg.github.io/foml/#home)。

[返回完整课程目录](../curriculum.md)
