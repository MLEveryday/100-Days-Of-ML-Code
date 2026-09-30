# Day 36：梯度下降：batch、epoch 与学习率

<!-- course-navigation:start -->
**先修导航**：[Day 5：逻辑回归：交叉熵与梯度](day-05.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-36.md)
<!-- course-navigation:end -->

## 学习目标

理解训练循环中的计量单位。

## 核心说明

batch 是一次梯度估计使用的样本组，step 是一次参数更新，epoch 是遍历训练数据一遍。小批量梯度有噪声，但可以降低内存并加快每步计算。

学习率太大可能震荡甚至发散，太小可能收敛缓慢。优化器降低训练损失并不保证验证损失同步下降。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
w = 5.0
for step in range(8):
    loss = (w-2)**2
    gradient = 2*(w-2)
    print(step, w, loss)
    w -= 0.2*gradient
```

## 结果解读

二次函数的更新逐步接近 w=2；深度网络的损失通常不是这么简单的凸函数。

## 练习

换成学习率 1.1 观察变化。600 个样本、batch_size=32 时，一个 epoch 有多少次更新？说明最后一个小批次。

[返回完整课程目录](../curriculum.md)
