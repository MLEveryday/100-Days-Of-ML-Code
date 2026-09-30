# Day 15：练习提示与参考答案

[返回本课](../../Code/Day%2015_Naive_Bayes.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

先算两个类别的未归一化分数，再除以分数之和。

</details>

<details>
<summary>第二层：参考解法</summary>

示例先验 P(A)=0.6、P(B)=0.4；两个观测特征的条件概率分别为 A:(0.5,0.2)、B:(0.1,0.8)。分数 0.06 与 0.032，后验约 0.6522 与 0.3478，预测 A。实际 GaussianNB 用概率密度而非离散概率。独立假设不完全满足仍可能保留正确的类别排序。

</details>

<details>
<summary>第三层：自检标准</summary>

两个后验之和为 1；高相关特征可能重复计算证据，影响概率校准。

</details>
