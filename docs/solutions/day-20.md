# Day 20：练习提示与参考答案

[返回本课](../../docs/lessons/day-20.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

记录数据拆分、种子、参数、评分和预算。

</details>

<details>
<summary>第二层：参考解法</summary>

实验表每行只改变 alpha 或一个网络超参数，写入验证均值/标准差与耗时。标准 inverted dropout 在训练时随机置零并除以保留率，推理时不随机丢弃，以使用稳定的完整表示；MC dropout 属于另行定义的例外方法。

</details>

<details>
<summary>第三层：自检标准</summary>

测试成绩不参与行之间选择；正则化、dropout、早停不要同时改变后归因给单个因素。

</details>
