# Day 54：练习提示与参考答案

[返回本课](../../Code/Day%2054_Hierarchical_Clustering.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

single 使用两簇间最小点距，容易被一串桥接点连接。

</details>

<details>
<summary>第二层：参考解法</summary>

固定原数据，另加一个离群点或桥接点序列，分别计算 single/complete 的树和轮廓分数。single 可链式连接，complete 会受最远点影响；单个离群点不一定产生链。按 distance_threshold 切树时设置 n_clusters=None，记录簇数。

</details>

<details>
<summary>第三层：自检标准</summary>

AgglomerativeClustering 无 predict；不能把新样本直接传给不存在的接口，重拟合会改变分组。

</details>
