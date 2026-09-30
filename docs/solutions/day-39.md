# Day 39：练习提示与参考答案

[返回本课](../../Code/Day%2039.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

固定划分，区分流程样本与完整学习效果。

</details>

<details>
<summary>第二层：参考解法</summary>

小样本模式只验证训练/保存/加载链路，不与完整训练成绩直接作公平性能比较。错分索引为 argmax(prob,axis=1)!=y，取一个查看其概率向量。softmax 和为 1 但并非校准保证，噪声、过拟合与分布变化都影响可信度。

</details>

<details>
<summary>第三层：自检标准</summary>

每次比较写入独立实验目录；不要选择测试集中好看的案例代表全体。

</details>
