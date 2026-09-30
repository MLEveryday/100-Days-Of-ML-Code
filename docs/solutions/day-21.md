# Day 21：练习提示与参考答案

[返回本课](../../Code/Day%2021_HTML_Data_Collection.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

区分缺字段、非数字和完全无记录。

</details>

<details>
<summary>第二层：参考解法</summary>

缺 .hours/.score 会触发显式 ValueError；非数字 float 转换报 ValueError；重复行被 drop_duplicates 去除；选择器匹配不到任何行会报错。真实采集可按明确规则跳过坏行并统计原因，保留 source_url、时间戳、单位和原始文本。

</details>

<details>
<summary>第三层：自检标准</summary>

不要把错误值自动写成 0；演示 HTML 不能标成真实采集数据。

</details>
