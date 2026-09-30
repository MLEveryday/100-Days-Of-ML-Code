# Day 25：练习提示与参考答案

[返回本课](../../Code/Day%2025_Decision_Tree.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

复杂度高会更容易拟合训练集，不保证验证提升。

</details>

<details>
<summary>第二层：参考解法</summary>

在训练内固定交叉验证折，对 max_depth=None 与 2/3/5 比较训练 accuracy、验证 F1 和叶节点数。无限深度可能训练近乎完美但验证更差；选择最佳验证方案后再测试。

</details>

<details>
<summary>第三层：自检标准</summary>

使用相同折比较，不能把最终测试集当作调参验证集。

</details>
