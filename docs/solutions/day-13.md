# Day 13：练习提示与参考答案

[返回本课](../../Code/Day%2013_SVM.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

在训练折上拟合每个 C，再观察支持向量数。

</details>

<details>
<summary>第二层：参考解法</summary>

可对包含 StandardScaler/SVC 的 Pipeline 用 GridSearchCV 搜索 classifier__C=[0.1,1,10]，保存每折验证分数，再用最佳模型查看 n_support_。每次 C 都从训练数据拟合，不能把同一个已缩放全量表传给所有折。

</details>

<details>
<summary>第三层：自检标准</summary>

报告选择指标及均值；支持向量多少本身不是越小越好。

</details>
