# Day 28：线性代数：点积与叉积

<!-- course-navigation:start -->
**先修导航**：[Day 26：线性代数：向量与线性变换](day-26.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-28.md)
<!-- course-navigation:end -->

## 学习目标

用点积理解相似度和投影。

## 核心说明

点积 uᵀv=||u||||v||cosθ，将长度与方向联系起来。余弦相似度消除长度影响，但零向量没有定义。三维叉积给出垂直于两向量的方向；本课显式使用三维向量以免混淆二维接口。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
u = np.array([1., 2., 0.])
v = np.array([2., 1., 0.])
print("dot", u@v)
print("cosine", u@v/(np.linalg.norm(u)*np.linalg.norm(v)))
print("cross", np.cross(u,v))
```

## 结果解读

叉积方向与交换输入顺序有关；点积交换顺序不变。

## 练习

验证叉积与两个输入都正交；给出长度不同但余弦相似度为 1 的向量。

[返回完整课程目录](../curriculum.md)
