# Day 42：练习提示与参考答案

[返回本课](../../Code/Day%2042.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

准确率只看越过阈值与否，交叉熵还看置信程度。

</details>

<details>
<summary>第二层：参考解法</summary>

真实 y=1 时，预测 0.49→0.01 都被判为 0，accuracy 不变，但 −log(p) 从约 0.713 上升到 4.605。仅改变一个因素并固定清单、预算和选模规则；根据验证损失选检查点，再报告最终测试。

</details>

<details>
<summary>第三层：自检标准</summary>

新一轮比较创建新目录，历史 config/history/checkpoint 不覆盖；无法据合成图片推断真实猫狗效果。

</details>
