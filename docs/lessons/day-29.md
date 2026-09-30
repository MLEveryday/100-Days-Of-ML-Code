# Day 29：线性代数：特征值与 PCA 预备

<!-- course-navigation:start -->
**先修导航**：[Day 27：线性代数：行列式、逆与子空间](day-27.md)、[Day 28：线性代数：点积与叉积](day-28.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-29.md)
<!-- course-navigation:end -->

## 学习目标

理解方向保持及对称矩阵分解。

## 核心说明

Av=λv（v≠0）表示变换后仍在 v 张成的直线上；λ>0 保持方向，λ<0 反向，λ=0 将该向量映射到零。实对称矩阵可用 eigh 获得实特征值和正交特征向量。协方差矩阵的主特征向量对应最大方差方向，是 PCA 的基础。

特征向量正负号都合法；特征值相同的子空间内基也不唯一，不能要求输出逐字一致。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
A = np.array([[2., 1.], [1., 2.]])
values, vectors = np.linalg.eigh(A)
order = np.argsort(values)[::-1]
values, vectors = values[order], vectors[:, order]
print(values, vectors)
print(np.allclose(A@vectors, vectors*values))
```

## 结果解读

按降序排序后第一列是最大特征值方向；eigh 默认顺序与此不同。

## 练习

生成相关二维数据，先中心化，再求协方差的主方向；解释特征缩放为何影响 PCA。

[返回完整课程目录](../curriculum.md)
