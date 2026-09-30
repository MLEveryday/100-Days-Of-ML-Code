# Day 26：线性代数：向量与线性变换

<!-- course-navigation:start -->
**先修导航**：[Day 45：NumPy：dtype、shape 与 ufunc](day-45.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-26.md)
<!-- course-navigation:end -->

## 学习目标

连接矩阵乘法、基向量和特征变换。

## 核心说明

向量可以表示坐标或特征；矩阵的列是基向量变换后的结果。列向量约定下 Av 表示变换，批量样本按行存储时通常写 X@A.T。线性变换保持加法与数乘，不包含独立平移项。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
A = np.array([[2., 1.], [0., 1.]])
v = np.array([1., 2.])
print(A @ v)
X = np.array([[1., 0.], [0., 1.], [1., 2.]])
print(X @ A.T)
```

## 结果解读

矩阵同时缩放和剪切；行存储与列存储约定不同会改变转置的位置。

## 练习

验证 A(u+v)=Au+Av。参考 [3Blue1Brown 线代系列](https://www.3blue1brown.com/topics/linear-algebra)，按向量、线性组合、矩阵乘法标题学习。

[返回完整课程目录](../curriculum.md)
