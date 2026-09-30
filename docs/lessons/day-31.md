# Day 31：微积分：极限与隐函数

<!-- course-navigation:start -->
**先修导航**：[Day 30：微积分：导数与链式法则](day-30.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-31.md)
<!-- course-navigation:end -->

## 学习目标

理解局部近似的适用范围。

## 核心说明

导数的差商来自极限，但计算机只计算有限精度值。隐函数如 x²+y²=1，在 y≠0 时求导得到 dy/dx=−x/y；靠近 y=0 时斜率会变大，不能直接除以零。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
x = 0.5
y = np.sqrt(1-x*x)
analytic = -x/y
h = 1e-5
numerical = (np.sqrt(1-(x+h)**2)-np.sqrt(1-(x-h)**2))/(2*h)
print(analytic, numerical)
```

## 结果解读

只取上半圆局部分支；隐函数可能有多分支或不满足局部可微条件。

## 练习

令 x 接近 1，观察数值稳定性；解释一步跨过定义域为何不能继续差分。

[返回完整课程目录](../curriculum.md)
