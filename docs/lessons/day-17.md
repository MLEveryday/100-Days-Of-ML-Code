# Day 17：从逻辑回归到神经元

<!-- course-navigation:start -->
**先修导航**：[Day 4：逻辑回归：从得分到概率](day-04.md)、[Day 26：线性代数：向量与线性变换](day-26.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-17.md)
<!-- course-navigation:end -->

## 学习目标

理解单神经元、非线性激活和批量向量化。

## 核心说明

一个二分类神经元与逻辑回归有相同的基本结构：加权和、偏置、sigmoid。多个神经元组成一层，用矩阵乘法一次处理多个样本。设 X 为 (n,d)，W 为 (d,h)，偏置 b 为 (1,h)，输出为 (n,h)。

隐藏层需要非线性激活；若连续多层都只是线性映射，整体仍可合并为一个线性映射。延伸学习可使用 [DeepLearning.AI 课程目录](https://www.deeplearning.ai/courses/)，以内容标题而非固定周次定位。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
X = np.array([[1., 2.], [3., 4.]])
W = np.array([[0.1, -0.2, 0.3], [0.2, 0.1, -0.1]])
b = np.zeros((1, 3))
Z = X @ W + b
print(Z.shape, np.maximum(Z, 0))
```

## 结果解读

两条样本通过三个 ReLU 神经元后仍有两行，列数变为三个隐藏特征。

## 练习

写出两层线性网络合并后的 W 和 b；加入 ReLU 后指出合并不再普遍成立。

[返回完整课程目录](../curriculum.md)
