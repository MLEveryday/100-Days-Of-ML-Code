# Day 43：K-means：目标、初始化与局限

<!-- course-navigation:start -->
**先修导航**：[Day 28：线性代数：点积与叉积](day-28.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-43.md)
<!-- course-navigation:end -->

## 学习目标

理解簇中心和局部最优。

## 核心说明

K-means 最小化每个点到其簇中心的平方欧氏距离之和。交替进行“分配到最近中心”和“重算每簇均值”，目标通常不增，但只能保证到达局部稳定解而非全局最优。

尺度不同的特征会主导距离；离群点会拉动均值。K 的选择可参考惯性变化、轮廓系数和任务解释，不能仅看惯性最小，因为 K 增大通常使惯性下降。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
X = np.array([[0., 0.], [0., 1.], [4., 4.], [5., 4.]])
centers = X[[0, 2]].copy()
labels = np.argmin(((X[:, None, :]-centers[None, :, :])**2).sum(axis=2), axis=1)
centers = np.array([X[labels==k].mean(axis=0) for k in range(2)])
print(labels, centers)
```

## 结果解读

簇编号 0/1 只是标识，不天然代表某种真实类别；完整实现还需处理空簇和停止条件。

## 练习

把一个离群点加入数据并重算中心。Day 44 比较球状簇和月牙数据。

[返回完整课程目录](../curriculum.md)
