# Day 50：pandas：字符串、时间与无泄漏特征

<!-- course-navigation:start -->
**先修导航**：[Day 48：pandas：索引、缺失值与 concat](day-48.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-50.md)
<!-- course-navigation:end -->

## 学习目标

理解时间索引和只使用过去信息的特征。

## 核心说明

字符串列使用 .str 方法批量处理；时间用 to_datetime 解析，跨地区数据明确时区。resample 将时间划入新频率区间，rolling 在窗口内聚合；二者含义不同。

预测未来时先按时间划分；目标的滚动特征一般需要 shift，避免把当前答案包含在自身特征中。不要随机打乱时间序列评估。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import pandas as pd
df = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=5, tz="UTC"), "sales": [3, 5, 4, 8, 9]})
df = df.set_index("time")
df["past_mean"] = df["sales"].shift(1).rolling(2).mean()
print(df)
print(df["sales"].resample("2D").sum())
```

## 结果解读

前两行 past_mean 缺失是因为没有足够过去样本；不能用未来值回填。旧 BA 别名在现代 pandas 中使用 BYE 等明确别名。

## 练习

不 shift 时写出第 3 天的特征，说明泄漏来源。比较 UTC 与本地时区转换后的日期边界。

[返回完整课程目录](../curriculum.md)
