# Day 8：逻辑回归：对数几率与正则化

<!-- course-navigation:start -->
**先修导航**：[Day 4：逻辑回归：从得分到概率](day-04.md)、[Day 5：逻辑回归：交叉熵与梯度](day-05.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-08.md)
<!-- course-navigation:end -->

## 学习目标

补充概率模型解释，与 Day 5 的优化步骤衔接。

## 核心说明

几率 odds=p/(1−p)，对数几率 log(odds)=wᵀx+b。其他变量不变时，某特征增加一单位，会将几率乘以 exp(wj)，并不是让概率固定增加 wj。

L2 正则化惩罚大系数，降低对训练噪声的适应。sklearn 的 C 是正则化强度的倒数，C 较小通常约束更强。系数解释依赖特征单位与共线性，不能直接作因果结论。原 README 的第三方链接跳转到其他主题，改以本地说明和官方文档为准。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
p = np.array([0.2, 0.5, 0.8])
print(np.log(p/(1-p)))
w = np.log(2.)
print("odds multiplier", np.exp(w))
```

## 结果解读

p=0.5 的对数几率为 0；系数 log(2) 表示条件几率翻倍。

## 练习

选两个不同初始概率，令几率翻倍，比较概率增量。参考 [LogisticRegression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)。

[返回完整课程目录](../curriculum.md)
