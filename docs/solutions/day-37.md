# Day 37：练习提示与参考答案

[返回本课](../../docs/lessons/day-37.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

w 同时经 z 和直接平方两条路径影响 L。

</details>

<details>
<summary>第二层：参考解法</summary>

dL/dw=2zx+2w。给定 w=2、x=3、b=1，则 z=7，梯度 42+4=46，L=49+4=53。偏置梯度仍为 2z=14。

</details>

<details>
<summary>第三层：自检标准</summary>

分支梯度相加，不是相乘；不能漏掉直接 w² 的路径。

</details>
