# Day 40：练习提示与参考答案

[返回本课](../../Code/Day%2040.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

按 split 建哈希集合并计算交集。

</details>

<details>
<summary>第二层：参考解法</summary>

train/validation/test 两两交集应为空；每个 split 的标签集合应为 {0,1}。n=20 每类时数量分别为 12/4/4。坏图统计只覆盖已扫描文件，改 limit 时新建清单；多个清单必须用 COURSE_PET_MANIFEST 明确选择。

</details>

<details>
<summary>第三层：自检标准</summary>

旧清单文件字节应保持不变；确认日志打印的清单路径正是预期实验。

</details>
