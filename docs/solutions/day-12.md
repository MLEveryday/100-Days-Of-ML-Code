# Day 12：练习提示与参考答案

[返回本课](../../docs/lessons/day-12.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

把 x 同时乘 a 后，平方距离乘 a²。

</details>

<details>
<summary>第二层：参考解法</summary>

RBF 变为 exp(−gamma·a²||x−x′||²)，等效于把 gamma 乘 a²。保持核不变需 gamma 除 a²。C 控制间隔违例代价、gamma 控制局部性，应在含标准化的 Pipeline 内联合交叉验证。

</details>

<details>
<summary>第三层：自检标准</summary>

标准化只拟合训练折；线性核不使用 RBF 的 gamma。

</details>
