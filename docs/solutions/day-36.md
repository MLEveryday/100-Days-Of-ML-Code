# Day 36：练习提示与参考答案

[返回本课](../../docs/lessons/day-36.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

更新误差 e=w−2，推导 e 的倍率。

</details>

<details>
<summary>第二层：参考解法</summary>

η=1.1 时 e_next=(1−2η)e=−1.2e，符号交替且绝对值放大，出现振荡发散。600/32 向上取整为 19 次更新，18 个完整 batch 共 576 条，末批 24 条（未 drop_remainder）。

</details>

<details>
<summary>第三层：自检标准</summary>

不能向下取整成 18；epoch 计数据遍历，step 计参数更新。

</details>
