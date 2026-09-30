# Day 30：微积分：导数与链式法则

<!-- course-navigation:start -->
**先修导航**：无其他课程硬性先修；需能读写基本 Python 表达式。

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-30.md)
<!-- course-navigation:end -->

## 学习目标

把局部变化率联系到损失优化。

## 核心说明

导数是局部变化率；复合函数 f(g(x)) 的导数为 f′(g(x))g′(x)。神经网络通过复合运算形成计算图，链式法则将局部导数连接起来。

有限差分是数值检查而非精确推导。中心差分通常比同等步长的单边差分更准确，但步长过小会有浮点消减误差。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
f = lambda x: np.sin(x*x)
x, h = 0.7, 1e-5
numerical = (f(x+h)-f(x-h))/(2*h)
analytic = np.cos(x*x)*2*x
print(numerical, analytic)
```

## 结果解读

结果应在小容差内一致；这可以帮助排查反向传播公式。

## 练习

将 h 从 1e-1 扫到 1e-12，画误差。参考 [微积分系列](https://www.3blue1brown.com/topics/calculus)。

[返回完整课程目录](../curriculum.md)
