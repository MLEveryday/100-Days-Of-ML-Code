# Day 5：逻辑回归：交叉熵与梯度

<!-- course-navigation:start -->
**先修导航**：[Day 4：逻辑回归：从得分到概率](day-04.md)、[Day 30：微积分：导数与链式法则](day-30.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-05.md)
<!-- course-navigation:end -->

## 学习目标

推导二分类损失及一个更新步骤；前置：Day 4、导数。

## 核心说明

单样本交叉熵是 −[y log(p)+(1−y)log(1−p)]。将 p=sigmoid(z) 代入，可写成 log(1+exp(z))−yz。这个损失对“自信但错误”的预测惩罚更大。

对 z 的梯度为 p−y；n 个样本的平均梯度为 Xᵀ(p−y)/n，偏置梯度为 mean(p−y)。梯度下降做 w←w−η∇w，η 是学习率。下面用 logaddexp 避免直接 exp 大数溢出。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
X = np.array([[0.], [1.], [2.]])
y = np.array([0., 0., 1.])
w, b = np.zeros(1), 0.
z = X @ w + b
p = np.exp(-np.logaddexp(0, -z))
loss = np.mean(np.logaddexp(0, z) - y*z)
w -= 0.1 * (X.T @ (p-y) / len(y))
b -= 0.1 * np.mean(p-y)
print(loss, w, b)
```

## 结果解读

初始概率全为 0.5，损失约 0.693；一次更新只是起点，并不等于已经训练好。

## 练习

用中心差分验证 w 梯度；尝试不同学习率重复更新并绘制损失，不用测试集选择学习率。

[返回完整课程目录](../curriculum.md)
