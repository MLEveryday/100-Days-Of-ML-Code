# Day 46：NumPy：广播、聚合与布尔掩码

<!-- course-navigation:start -->
**先修导航**：[Day 45：NumPy：dtype、shape 与 ufunc](day-45.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-46.md)
<!-- course-navigation:end -->

## 学习目标

理解维度对齐和条件筛选。

## 核心说明

广播从尾部维度比较，两维相等或其中一个为 1 才兼容。避免依赖“看起来能算”的形状隐式行为；keepdims 可保留轴方便广播。

数组条件用 &、|、~，各比较式加括号，不用 Python 标量 and/or。NaN 聚合要按语义选择 nanmean 等函数，而非一律忽略。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
X = np.array([[1., 10.], [2., 20.], [3., 30.]])
centered = X-X.mean(axis=0, keepdims=True)
mask = (X[:, 0] >= 2) & (X[:, 1] < 30)
print(centered, X[mask])
```

## 结果解读

(3,2) 减 (1,2) 合法；若误得到 (3,) 均值向量，尾维可能不兼容或产生错误语义。

## 练习

预测 (3,1)+(1,4) 的形状，先手算再运行。旧外部教材的 seaborn-whitegrid 样式改用现代样式或不用样式。

[返回完整课程目录](../curriculum.md)
