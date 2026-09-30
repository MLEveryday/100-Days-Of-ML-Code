# Day 46：练习提示与参考答案

[返回本课](../../docs/lessons/day-46.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

从右向左对齐形状，缺维视为 1。

</details>

<details>
<summary>第二层：参考解法</summary>

(3,1)+(1,4) 输出 (3,4)：每行的单值与长度 4 的行相加。比如 arange(3)[:,None]+arange(4)[None,:] 得 [[0,1,2,3],[1,2,3,4],[2,3,4,5]]。

</details>

<details>
<summary>第三层：自检标准</summary>

广播不是数据自动匹配含义；确认轴代表样本还是特征。

</details>
