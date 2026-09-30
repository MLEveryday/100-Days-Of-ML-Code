# Day 31：练习提示与参考答案

[返回本课](../../docs/lessons/day-31.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

差分两端 x±h 都需落在 [−1,1]。

</details>

<details>
<summary>第二层：参考解法</summary>

上半圆导数 −x/√(1−x²)，x 接近 1 时绝对值很大。若 x+h>1，平方根无实数值，NumPy 返回 NaN/警告；先检查定义域再调整 h，而不是把 NaN 当导数。

</details>

<details>
<summary>第三层：自检标准</summary>

边界处此分支没有有限导数；减小 h 也不能把不存在的有限导数算出来。

</details>
