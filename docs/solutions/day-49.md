# Day 49：练习提示与参考答案

[返回本课](../../docs/lessons/day-49.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

many_to_one 要求右表每个键最多一条。

</details>

<details>
<summary>第二层：参考解法</summary>

给 users 添加重复 user=1 后 merge(validate="many_to_one") 应报 MergeError。如果两条记录完全重复可去除完全重复行；若地区冲突，需要来源/生效时间规则或修正数据，不能任意保留第一条。

</details>

<details>
<summary>第三层：自检标准</summary>

去重后再次校验键唯一性和订单行数，不能让连接悄悄成倍增加记录。

</details>
