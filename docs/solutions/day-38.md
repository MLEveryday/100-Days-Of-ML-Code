# Day 38：练习提示与参考答案

[返回本课](../../docs/lessons/day-38.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

优先破坏可通过形状检查但数值错误的项。

</details>

<details>
<summary>第二层：参考解法</summary>

例如漏掉平均损失的 1/n，解析梯度约放大 n 倍，差分检查会失败。错误转置有时先导致形状异常，也算被捕获。恢复正确实现后检查通过；使用固定参数、小 batch 和中心差分。

</details>

<details>
<summary>第三层：自检标准</summary>

测试失败后恢复代码；不通过放大容差让错误“通过”。

</details>
