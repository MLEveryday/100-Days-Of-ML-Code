# Day 48：pandas：索引、缺失值与 concat

<!-- course-navigation:start -->
**先修导航**：[Day 45：NumPy：dtype、shape 与 ufunc](day-45.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-48.md)
<!-- course-navigation:end -->

## 学习目标

掌握按标签操作表格，并替换旧 append。

## 核心说明

loc 按标签选取，iloc 按位置选取。赋值使用 df.loc[mask, column]=value，避免链式赋值：在不同 pandas 版本、Copy-on-Write 设置下它可能不能更新原表。

DataFrame.append 已被移除；用 pd.concat 合并行。批量收集多个表后一次 concat，比循环不断扩展表更清晰。缺失值不是零；Int64 等可空类型可以保留缺失整数。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import pandas as pd
a = pd.DataFrame({"name": ["A", "B"], "score": pd.Series([80, None], dtype="Int64")})
b = pd.DataFrame({"name": ["C"], "score": [90]})
combined = pd.concat([a, b], ignore_index=True)
combined.loc[combined["score"].isna(), "score"] = 0
print(combined)
```

## 结果解读

本例把缺失填零只是展示赋值语法，真实分数是否能这样处理需业务依据。ML 填补规则仍只能在训练集拟合。

## 练习

比较 loc/iloc 在非默认索引上的差别；保留缺失指示列，讨论直接填零的风险。

[返回完整课程目录](../curriculum.md)
