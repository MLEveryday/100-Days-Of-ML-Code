# Day 52：练习提示与参考答案

[返回本课](../../docs/lessons/day-52.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

用 hist 的返回值 heights 和 edges。

</details>

<details>
<summary>第二层：参考解法</summary>

np.sum(heights*np.diff(edges)) 应约等于 1（density=True、计入的样本非空）。fig.savefig(...,dpi=150) 导出 PNG，另保存 .svg；前者是像素图，后者通常可无损缩放矢量元素，但大散点图 SVG 可能很大。

</details>

<details>
<summary>第三层：自检标准</summary>

柱高之和一般不为 1；不要混淆密度与概率质量。

</details>
