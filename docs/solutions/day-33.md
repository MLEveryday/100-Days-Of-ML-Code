# Day 33：练习提示与参考答案

[返回本课](../../docs/lessons/day-33.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

单次未被抽中的概率是 (1−1/n)^n。

</details>

<details>
<summary>第二层：参考解法</summary>

n=100、抽样 100 次时，某样本未出现概率 (99/100)^100≈0.3660。重复多次平均 OOB 比例应接近它。相关特征可互相替代，置乱其中一个的影响可能较小，不能据此判其没有业务关联。

</details>

<details>
<summary>第三层：自检标准</summary>

OOB 比例不是固定每次恰好 36.6%；重要性不是因果效应。

</details>
