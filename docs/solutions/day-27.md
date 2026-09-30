# Day 27：练习提示与参考答案

[返回本课](../../docs/lessons/day-27.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

选择容易手算的秩 1 矩阵。

</details>

<details>
<summary>第二层：参考解法</summary>

A=[[2,1],[4,2]] 第二行是第一行两倍，det=0。v=[1,−2] 非零且 Av=0；列线性相关，没有唯一逆。b 若不在列空间则无解，在列空间则解不唯一，solve 要求唯一解因而报错。

</details>

<details>
<summary>第三层：自检标准</summary>

验证 Av 接近零；不要把奇异矩阵报错解释成浮点库故障。

</details>
