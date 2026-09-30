# Day 37：反向传播：计算图与局部导数

<!-- course-navigation:start -->
**先修导航**：[Day 30：微积分：导数与链式法则](day-30.md)、[Day 35：神经网络：层、形状与激活](day-35.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-37.md)
<!-- course-navigation:end -->

## 学习目标

区分算梯度和更新参数。

## 核心说明

反向传播沿计算图从损失向输入传播导数，利用链式法则重用中间结果。优化器再根据梯度更新参数；反向传播本身不规定学习率。

例如 z=wx+b、L=z²，则 dL/dz=2z、dL/dw=2zx、dL/db=2z。多分支相遇时，来自各路径的梯度相加。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
w, x, b = 2., 3., 1.
z = w*x+b
loss = z*z
dz = 2*z
dw, db, dx = dz*x, dz, dz*w
print(loss, dw, db, dx)
```

## 结果解读

每个梯度的形状应与对应变量一致；矩阵情形尤其需要检查。

## 练习

将 L 改为 z²+w²，写出 dw，说明为何需要额外加上 2w。

[返回完整课程目录](../curriculum.md)
