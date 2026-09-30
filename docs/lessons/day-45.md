# Day 45：NumPy：dtype、shape 与 ufunc

<!-- course-navigation:start -->
**先修导航**：无其他课程硬性先修；需能读写基本 Python 表达式。

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-45.md)
<!-- course-navigation:end -->

## 学习目标

建立数组与向量化的基础。

## 核心说明

ndarray 的元素共享 dtype，shape 描述各轴长度，axis 指操作沿哪个维度进行。ufunc 对数组逐元素运算，通常比 Python 循环简洁。向量化仍需考虑中间数组内存。

显式整数类型可能溢出；float32/float64 精度与内存不同。新例子使用 default_rng 管理随机状态，不影响其他模块的全局随机数。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
rng = np.random.default_rng(0)
a = rng.normal(size=(3, 2)).astype(np.float32)
print(a.shape, a.dtype, a.nbytes)
print(np.square(a))
print(a.mean(axis=0))
```

## 结果解读

axis=0 沿样本轴聚合，输出每列一个均值，形状为 (2,)；默认 axis=None 则聚合全部元素。

## 练习

比较 astype(float64) 的内存；给出整数平方溢出的例子并选择合理类型。参见[阅读说明](../readings.md)。

[返回完整课程目录](../curriculum.md)
