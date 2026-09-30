# Day 44：练习提示与参考答案

[返回本课](../../Code/Day%2044_KMeans.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

随机种子变化与缩放变化分成两组实验。

</details>

<details>
<summary>第二层：参考解法</summary>

固定 K，比较多个初始化种子的 inertia，再比较 n_init=1 与 10。将一维乘 100 时原始平方距离该维权重乘 10000；按训练/分析数据的列标准化后尺度主导减弱。轮廓系数依赖距离几何，未必符合业务定义。

</details>

<details>
<summary>第三层：自检标准</summary>

本例是无监督结构分析，不能把簇编号与真标签直接算分类 accuracy。

</details>
