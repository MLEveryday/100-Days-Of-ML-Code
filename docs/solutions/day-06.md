# Day 6：练习提示与参考答案

[返回本课](../../Code/Day%206_Logistic_Regression.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

从原训练集再拆验证集，使用 predict_proba 的正类列。

</details>

<details>
<summary>第二层：参考解法</summary>

固定训练出的模型，在验证集扫描 0.2～0.8 等预先选定阈值，计算 precision/recall/F1。阈值升高，预测正类数不会增加，recall 不会上升；precision 不保证严格单调。按任务成本选阈值，之后只评价一次测试集。

</details>

<details>
<summary>第三层：自检标准</summary>

标明正类 Purchased=1；不能用测试标签选阈值。

</details>
