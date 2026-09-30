# Day 48：练习提示与参考答案

[返回本课](../../docs/lessons/day-48.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

把 index 设为 [10,20] 再选择。

</details>

<details>
<summary>第二层：参考解法</summary>

loc[10] 选择标签 10，iloc[0] 选择第一行；loc[0] 在此会 KeyError。填补前保存 score_missing=score.isna()，否则填零后无法区分真零与原缺失；填补方案应基于数据语义。

</details>

<details>
<summary>第三层：自检标准</summary>

缺失指示列先生成，不能填补后再 isna。

</details>
