# Day 29：练习提示与参考答案

[返回本课](../../docs/lessons/day-29.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

先中心化，再计算协方差矩阵的 eigh。

</details>

<details>
<summary>第二层：参考解法</summary>

生成 t 与 2t+小噪声作为两列，减训练列均值，np.cov(...,rowvar=False) 得到对称矩阵。最大特征值对应的向量近似 [1,2]/√5（正负均可）。把一列放大后其方差贡献增大，主方向会改变。

</details>

<details>
<summary>第三层：自检标准</summary>

用 C@v≈λv 和向量范数 1 检查；不要要求固定特征向量符号。

</details>
