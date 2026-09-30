# Day 14：练习提示与参考答案

[返回本课](../../Code/Day%2014_SVM_Margin.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

SVC 的规范化边界为 wᵀx+b=±1。

</details>

<details>
<summary>第二层：参考解法</summary>

两条平行边界距离是 2/||w||，使用 `np.linalg.norm(model.coef_)`。同时记录 C、宽度、n_support_ 与验证分数。本例得到何种变化应从输出读取；数值容差、退化解和软间隔违例使图上“最近样本距离”不等于该宽度。

</details>

<details>
<summary>第三层：自检标准</summary>

分母是范数而不是平方范数；支持向量可能在软间隔内，不能用两类最近点直接代替。

</details>
