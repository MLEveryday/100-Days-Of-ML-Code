# Day 27：线性代数：行列式、逆与子空间

<!-- course-navigation:start -->
**先修导航**：[Day 26：线性代数：向量与线性变换](day-26.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-27.md)
<!-- course-navigation:end -->

## 学习目标

理解可逆性及求解线性系统。

## 核心说明

列空间由矩阵列的所有线性组合构成，零空间包含满足 Ax=0 的向量。方阵行列式为零时不可逆；绝对值很小还需结合尺度和条件数判断数值稳定性。

解 Ax=b 优先用 np.linalg.solve，不显式求逆后相乘；非方阵或噪声系统可考虑最小二乘。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
A = np.array([[2., 1.], [1., 3.]])
b = np.array([1., 2.])
x = np.linalg.solve(A, b)
print(x, "residual", np.linalg.norm(A@x-b))
print("determinant", np.linalg.det(A), "condition", np.linalg.cond(A))
```

## 结果解读

残差接近零是数值检验，但病态系统即使残差小也可能对输入扰动敏感。

## 练习

把一行改为另一行的倍数，解释为何 solve 报错；找一个非零零空间向量。

[返回完整课程目录](../curriculum.md)
