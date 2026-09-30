# Day 50：练习提示与参考答案

[返回本课](../../docs/lessons/day-50.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

第 3 天滚动两天的边界分别写出来。

</details>

<details>
<summary>第二层：参考解法</summary>

原 sales=[3,5,4,8,9]。不 shift 的第 3 天均值为 (5+4)/2=4.5，包含当天目标；shift 后为 (3+5)/2=4。UTC 2024-01-01 20:00 转 Asia/Shanghai 为 2024-01-02 04:00，分日汇总边界会变。

</details>

<details>
<summary>第三层：自检标准</summary>

只有预测时已知的过去真实值可用；多步预测不能直接用未来各日真实销量滚动。

</details>
