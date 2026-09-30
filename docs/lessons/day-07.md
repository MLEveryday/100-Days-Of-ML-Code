# Day 7：KNN：距离、邻居和投票

<!-- course-navigation:start -->
**先修导航**：[Day 28：线性代数：点积与叉积](day-28.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-07.md)
<!-- course-navigation:end -->

## 学习目标

理解惰性学习以及 K 的作用；前置：欧氏距离。

## 核心说明

KNN 保存训练样本，预测时找最近 K 个邻居。分类用投票或距离加权，回归通常取平均。非参数指没有预设固定形式的分布模型，不表示没有 K、距离度量等超参数。

K 太小容易受噪声影响，太大容易平滑掉局部结构。特征量纲会改变距离；年龄与薪资不缩放就容易被薪资主导。高维下距离区分度下降，称为维度灾难的一部分。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
X = np.array([[0., 0.], [1., 0.], [2., 2.]])
y = np.array([0, 0, 1])
q = np.array([0.8, 0.2])
d = np.linalg.norm(X-q, axis=1)
nearest = np.argsort(d)[:2]
print(d, nearest, np.bincount(y[nearest]).argmax())
```

## 结果解读

最近的两点均属于 0。平票需要明确规则；选择奇数 K 也不能解决所有多分类平票。

## 练习

把一个坐标乘 100 后重算距离；说明训练内标准化为何会改变结果。Day 11 用交叉验证选 K。

[返回完整课程目录](../curriculum.md)
