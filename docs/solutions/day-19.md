# Day 19：练习提示与参考答案

[返回本课](../../docs/lessons/day-19.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

同一输入的确定性分类器只能给出一个类别。

</details>

<details>
<summary>第二层：参考解法</summary>

添加相同 x 的 +1 和 −1 标签后，不存在把这两个样本都分对的线性边界，算法可持续来回更新。保留 max_epochs，并输出未收敛状态；可用 pocket 方法保存错误最少的权重，但不能消除标签矛盾。

</details>

<details>
<summary>第三层：自检标准</summary>

训练不收敛不是无限延长训练的理由；不可谎称不可分数据满足感知机收敛条件。

</details>
