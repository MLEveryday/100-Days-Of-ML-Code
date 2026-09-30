# Day 34：练习提示与参考答案

[返回本课](../../Code/Day%2034_Random_Forests.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

同一折、同一种子与数据，比较树数。

</details>

<details>
<summary>第二层：参考解法</summary>

对 10/100 棵树用训练内相同 CV 折记录评分、标准差和训练耗时；更多树常降低随机波动但增加成本，不保证每次分数上升。Permutation importance 是固定模型下置乱特征带来的评分下降，相关特征会分摊/掩盖影响。

</details>

<details>
<summary>第三层：自检标准</summary>

测试集重要性仅解释最终模型，不能看过后继续按测试成绩选特征。

</details>
