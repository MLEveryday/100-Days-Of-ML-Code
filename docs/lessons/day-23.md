# Day 23：决策树：熵、分裂与 CART

<!-- course-navigation:start -->
**先修导航**：无其他课程硬性先修；需能读写基本 Python 表达式。

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-23.md)
<!-- course-navigation:end -->

## 学习目标

用数值例子理解分类纯度，衔接 Day 25。

## 核心说明

分类熵 H=−Σpk log₂pk；纯节点熵为零。信息增益为父节点熵减去子节点加权熵。连续特征可按阈值划成左右两支。

ID3 常用类别特征多路分裂，sklearn 的 DecisionTreeClassifier 是 CART 风格二叉树，criterion="entropy" 不会把它变成 ID3。深树可记住训练噪声，需要深度、最小叶样本或剪枝控制复杂度。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
def entropy(counts):
    p = np.asarray(counts, dtype=float)
    p = p[p > 0]/p.sum()
    return -np.sum(p*np.log2(p))
parent = entropy([4, 4])
after = 0.5*entropy([3, 1]) + 0.5*entropy([1, 3])
print("entropy", parent, "gain", parent-after)
```

## 结果解读

父节点完全混合熵为 1，比两子节点更不纯；此分裂有正信息增益。

## 练习

比较另一个分裂 [4,0]/[0,4]。讨论为什么训练信息增益最大并不保证泛化最好。

[返回完整课程目录](../curriculum.md)
