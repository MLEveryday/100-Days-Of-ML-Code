# Day 1：练习提示与参考答案

[返回本课](../../Code/Day%201_Data_Preprocessing.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

只用 X_train 的 Age 求均值；类别列和数值列分开检查。

</details>

<details>
<summary>第二层：参考解法</summary>

训练集非缺失均值可用 `X_train["Age"].mean()`；应与填补器 statistics_[0] 相同。复制 X_test，把 Country 设为未知值，再 transform：国家独热列全零，数值列仍按原参数处理。全量填补会让测试分布影响均值，造成泄漏。

</details>

<details>
<summary>第三层：自检标准</summary>

8 条训练、2 条测试；转换后均为有限数，列数相同；测试数据改变后填补器参数不变。

</details>
