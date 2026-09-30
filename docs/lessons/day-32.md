# Day 32：微积分：Taylor 近似与曲率

<!-- course-navigation:start -->
**先修导航**：[Day 30：微积分：导数与链式法则](day-30.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-32.md)
<!-- course-navigation:end -->

## 学习目标

把一阶、二阶信息联系到优化。

## 核心说明

在 x0 附近，f(x0+δ)≈f(x0)+f′(x0)δ+½f″(x0)δ²。二阶项描述曲率；局部近似不应被当作远处的精确模型。多变量中梯度对应一阶信息，Hessian 对应二阶信息。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
for delta in [0.1, 0.5, 1.]:
    exact = np.exp(delta)
    approximation = 1+delta+delta**2/2
    print(delta, exact, approximation, abs(exact-approximation))
```

## 结果解读

展开点越远，截断误差通常越明显。

## 练习

给出梯度为零但不是极小值的函数例子。总结导数、链式法则和曲率在训练网络中的不同作用。

[返回完整课程目录](../curriculum.md)
