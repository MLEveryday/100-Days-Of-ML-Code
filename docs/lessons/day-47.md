# Day 47：NumPy：索引、排序与结构化数据

<!-- course-navigation:start -->
**先修导航**：[Day 45：NumPy：dtype、shape 与 ufunc](day-45.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-47.md)
<!-- course-navigation:end -->

## 学习目标

区分视图与副本，避免意外改写。

## 核心说明

基本切片通常产生视图，花式索引通常产生副本。修改视图会影响原数组；需要独立数据时显式 copy。argsort 返回排序索引，argpartition 只保证分区，不保证区内完整顺序。

结构化数组允许字段有不同 dtype，适合部分底层存储任务；日常带标签表格通常用 pandas 更方便。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
a = np.arange(6)
view = a[1:4]
copy = a[[1, 2, 3]]
view[0] = 99
print(a, copy)
values = np.array([5., 2., 8., 1.])
print(np.argsort(values), np.argpartition(values, 1)[:2])
```

## 结果解读

原数组的第二项变为 99，但花式索引副本仍保留原值。

## 练习

创建含 name/score 字段的结构化数组。旧损坏链接已替换为[完整章节](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.09-Structured-Data-NumPy.ipynb)。

[返回完整课程目录](../curriculum.md)
