# Day 2：练习提示与参考答案

[返回本课](../../Code/Day%202_Simple_Linear_Regression.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

基线只能使用训练目标的均值；残差要保留原行索引。

</details>

<details>
<summary>第二层：参考解法</summary>

用 `DummyRegressor(strategy="mean")` 在 X_train/y_train 上拟合，计算测试 MAE，再与模型比较。按 `abs(y_test-y_pred)` 降序查看异常行，核实录入和来源，不按误差删除。对预先列出的多个划分种子报告全部 MAE 或均值/标准差。

</details>

<details>
<summary>第三层：自检标准</summary>

基线不能从 y_test 求均值；大小关系由实测决定，不强求线性回归在每次小样本划分都获胜。

</details>
