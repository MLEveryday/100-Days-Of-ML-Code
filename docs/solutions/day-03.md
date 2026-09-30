# Day 3：练习提示与参考答案

[返回本课](../../Code/Day%203_Multiple_Linear_Regression.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

独热编码的基准类查看 categories_ 和 drop_idx_；比较预测与系数两件事。

</details>

<details>
<summary>第二层：参考解法</summary>

拟合后的 state 编码器 `categories_[0][drop_idx_[0]]` 给出基准州。州系数表示在其他特征不变时相对基准的预测差。保持拆分相同，将 drop 改为 None 重新拟合：设计矩阵秩变化，系数可能变化，预测可近似相同。Ridge 的 alpha 通过训练内 GridSearchCV 选择。

</details>

<details>
<summary>第三层：自检标准</summary>

记录特征名与系数对应关系；比较预测用 allclose 容差，不能用“删列必然提高准确率”作结论。

</details>
