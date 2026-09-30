# Day 12：SVM：软间隔与核函数

<!-- course-navigation:start -->
**先修导航**：[Day 9：SVM：线性分隔与支持向量](day-09.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-12.md)
<!-- course-navigation:end -->

## 学习目标

解释 C、gamma 和核方法，准备 Day 13/16。

## 核心说明

软间隔目标可写为 ½||w||²+CΣξi，并满足 yi(wᵀxi+b)≥1−ξi、ξi≥0（标签 yi∈{−1,+1}）。ξi 允许间隔违例，包括分对但落入间隔的样本。C 大时更重视违例代价，C 小时更容许训练误差以换取较宽间隔。

核函数计算隐式特征映射的内积。RBF 核 K(x,x′)=exp(−gamma·||x−x′||²)；gamma 大意味着相似度随距离衰减更快。线性核没有这个 RBF 参数作用。手工升维可帮助理解，但一般无需显式构造高维特征。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
distance_squared = np.array([0., 1., 4.])
for gamma in [0.1, 1., 10.]:
    print(gamma, np.exp(-gamma * distance_squared))
```

## 结果解读

距离为零时核值为 1；gamma 越大，远点影响越小。

## 练习

解释为什么缩放输入会改变 RBF 核值，并讨论 C/gamma 应联合选择。

[返回完整课程目录](../curriculum.md)
