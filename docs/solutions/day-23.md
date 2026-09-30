# Day 23：练习提示与参考答案

[返回本课](../../docs/lessons/day-23.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

纯节点只包含一类，熵为 0。

</details>

<details>
<summary>第二层：参考解法</summary>

父节点 [4,4] 熵为 1 bit。原分裂 [3,1]/[1,3] 子节点加权熵约 0.811278，增益约 0.188722；[4,0]/[0,4] 子节点熵 0，增益 1。训练纯度提升可能在拟合噪声，仍需验证。

</details>

<details>
<summary>第三层：自检标准</summary>

信息增益需按子节点样本数加权；不能只把子节点熵相加。

</details>
