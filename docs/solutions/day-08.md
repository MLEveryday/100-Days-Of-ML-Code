# Day 8：练习提示与参考答案

[返回本课](../../docs/lessons/day-08.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

把 odds 翻倍后，用 odds/(1+odds) 转回概率。

</details>

<details>
<summary>第二层：参考解法</summary>

新概率 p′=2p/(1+p)。p=0.2 时 p′=1/3，增量约 0.1333；p=0.5 时 p′=2/3，增量约 0.1667。相同几率倍数对应不同概率增量。

</details>

<details>
<summary>第三层：自检标准</summary>

结果仍在 (0,1)，不能把 log(2) 当作固定概率增量。

</details>
