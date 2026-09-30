# Day 17：练习提示与参考答案

[返回本课](../../docs/lessons/day-17.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

采用行向量批量约定 XW+b。

</details>

<details>
<summary>第二层：参考解法</summary>

两层线性/仿射映射为 (XW1+b1)W2+b2，合并 W=W1W2、b=b1W2+b2。ReLU 在输入跨过零点时改变有效斜率，一般不能由一个全局仿射映射表示。

</details>

<details>
<summary>第三层：自检标准</summary>

核对 W1:(d,h)、W2:(h,k)、合并 W:(d,k)，偏置广播结果为 (n,k)。

</details>
