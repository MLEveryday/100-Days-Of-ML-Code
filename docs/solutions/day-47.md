# Day 47：练习提示与参考答案

[返回本课](../../docs/lessons/day-47.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

用字段名和 dtype 的列表构造结构化 dtype。

</details>

<details>
<summary>第二层：参考解法</summary>

`np.array([("A",80.0),("B",90.0)], dtype=[("name","U10"),("score","f8")])`，然后按 arr["score"] 取列。`U10` 的长度是固定上限，较长字符串可能被截断。

</details>

<details>
<summary>第三层：自检标准</summary>

score 取出应为 [80,90]；基本切片可影响原数组，花式索引是副本。

</details>
