# Day 45：练习提示与参考答案

[返回本课](../../docs/lessons/day-45.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

内存为元素数×每元素字节数，溢出前先转类型。

</details>

<details>
<summary>第二层：参考解法</summary>

示例 3×2 数组 float32 占 24 字节，float64 占 48 字节。np.array([100],dtype=np.int8)**2 超出 int8 的 [−128,127]，应先 astype(np.int64) 再平方，结果 10000；运算后再转换无法恢复。

</details>

<details>
<summary>第三层：自检标准</summary>

nbytes 不包含 Python 对象开销；类型选择还需考虑目标取值范围。

</details>
