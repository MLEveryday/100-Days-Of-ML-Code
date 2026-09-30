# Day 10：练习提示与参考答案

[返回本课](../../docs/lessons/day-10.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

用预先指定的噪声/样本数组合，而不是不断试到满意。

</details>

<details>
<summary>第二层：参考解法</summary>

例如固定噪声 [0.05,0.2,0.4]、样本数 [100,300]，每组合固定若干种子，按相同验证比例分别拟合两条 Pipeline，记录验证 accuracy 的均值和波动。月牙数据上局部方法常更适合，但高噪声会削弱优势。

</details>

<details>
<summary>第三层：自检标准</summary>

完整报告所有组合；不可据一次分数认定某算法普遍更优。

</details>
