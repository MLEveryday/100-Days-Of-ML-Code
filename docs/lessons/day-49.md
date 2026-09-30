# Day 49：pandas：连接、分组与透视表

<!-- course-navigation:start -->
**先修导航**：[Day 48：pandas：索引、缺失值与 concat](day-48.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-49.md)
<!-- course-navigation:end -->

## 学习目标

避免连接膨胀并解释分组统计。

## 核心说明

merge 按键组合表；重复键可能造成多对多乘积。用 validate="many_to_one" 等声明预期关系，indicator 查看匹配状态。

pivot 要求索引/列组合唯一，pivot_table 可通过聚合处理重复。groupby 的缺失键、分类变量未出现类别等默认行为可能随版本变化，重要时显式给出 dropna、observed。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import pandas as pd
orders = pd.DataFrame({"user": [1, 1, 2], "amount": [10, 20, 15]})
users = pd.DataFrame({"user": [1, 2], "region": ["east", "west"]})
joined = orders.merge(users, on="user", validate="many_to_one", indicator=True)
print(joined)
print(joined.groupby("region", dropna=False, observed=True)["amount"].sum())
```

## 结果解读

user=1 的两笔订单是合法多对一；若 users 出现重复 user，validate 会报错而不是静默膨胀。

## 练习

加入重复用户键确认错误，再决定去重规则；不要盲目 drop_duplicates 消除问题。

[返回完整课程目录](../curriculum.md)
