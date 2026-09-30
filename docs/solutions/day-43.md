# Day 43：练习提示与参考答案

[返回本课](../../docs/lessons/day-43.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

均值对远距离样本敏感。

</details>

<details>
<summary>第二层：参考解法</summary>

若向簇 [[0,0],[0,1]] 添加 [100,100]，均值从 [0,0.5] 变为 [33.333,33.667]。完整 K-means 还会重新分配，所以此计算只展示固定簇均值的敏感性。

</details>

<details>
<summary>第三层：自检标准</summary>

区分固定分组重算中心与完整算法重新分配，不宣称只做一次均值就是最终解。

</details>
