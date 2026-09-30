# Day 30：练习提示与参考答案

[返回本课](../../docs/lessons/day-30.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

将每个 h 的中心差分与解析导数比较。

</details>

<details>
<summary>第二层：参考解法</summary>

f(x)=sin(x²) 的导数 2x cos(x²)。在 x=0.7 处对 h=10**(-k), k=1…12 计算绝对误差并画双对数图。开始减小 h 会降低截断误差，极小时相减损失有效位数，误差可能回升。

</details>

<details>
<summary>第三层：自检标准</summary>

不预设误差严格单调；同时记录 h 与误差，不能只报告最小值。

</details>
