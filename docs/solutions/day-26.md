# Day 26：练习提示与参考答案

[返回本课](../../docs/lessons/day-26.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

分别计算 A@(u+v) 和 A@u+A@v。

</details>

<details>
<summary>第二层：参考解法</summary>

任取同维 u,v，np.allclose 两种计算应为 True。按矩阵分配律逐分量展开即可证明。添加平移 f(x)=Ax+b 后，f(u+v) 与 f(u)+f(v) 相差 b，因此一般不是线性变换。

</details>

<details>
<summary>第三层：自检标准</summary>

证明同时需数乘性质；几个数值例子只能核对实现，不能替代一般证明。

</details>
