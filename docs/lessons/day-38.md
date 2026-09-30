# Day 38：反向传播：梯度检查

<!-- course-navigation:start -->
**先修导航**：[Day 37：反向传播：计算图与局部导数](day-37.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-38.md)
<!-- course-navigation:end -->

## 学习目标

用数值差分验证解析梯度。

## 核心说明

中心差分 [L(θ+h)−L(θ−h)]/(2h) 可近似梯度；逐参数检查耗时高，适合小网络调试。检查时固定数据、关闭随机 dropout，扰动后恢复原参数。

绝对误差适合这里的小量级例子；量级跨度大时还应比较相对误差。ReLU 在零点不可微，差分穿过折点时不应简单判解析公式错误。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
x, y = np.array([1., 2.]), 1.
w = np.array([0.2, -0.1])
loss = lambda p: (x@p-y)**2
analytic = 2*(x@w-y)*x
numeric = np.zeros_like(w)
for i in range(len(w)):
    direction = np.zeros_like(w)
    direction[i] = 1e-5
    numeric[i] = (loss(w+direction)-loss(w-direction))/(2e-5)
print(analytic, numeric)
np.testing.assert_allclose(analytic, numeric, atol=1e-7)
```

## 结果解读

与解析梯度一致说明该小样本的实现合理，不证明整个训练系统无误。

## 练习

对照 Day 18 的全部参数检查，主动改错一个转置或平均因子，确认检查能捕获。

[返回完整课程目录](../curriculum.md)
