# Day 35：神经网络：层、形状与激活

<!-- course-navigation:start -->
**先修导航**：[Day 17：从逻辑回归到神经元](day-17.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-35.md)
<!-- course-navigation:end -->

## 学习目标

把单个神经元扩展为多层网络。

## 核心说明

全连接层将每个输入与各输出神经元连接：H=activation(XW+b)。输入 batch 维度不能混入特征维度。权重、偏置是可学习参数，激活函数一般不是每个样本独立学习的参数。

MNIST 的 28×28 输入可展开为 784 维，Dense(128) 的参数量为 784×128+128；卷积网络则通过局部连接和共享权重利用空间结构。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
X = np.ones((4, 784))
W = np.zeros((784, 128))
b = np.zeros(128)
print((X@W+b).shape)
print("parameters", W.size+b.size)
```

## 结果解读

本例零初始化仅用于形状演示，实际隐藏层不应这样初始化。

## 练习

计算再接 Dense(10) 的参数量；参考 [神经网络系列](https://www.3blue1brown.com/topics/neural-networks)，实现见 Day 39。

[返回完整课程目录](../curriculum.md)
