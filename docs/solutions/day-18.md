# Day 18：练习提示与参考答案

[返回本课](../../Code/Day%2018_Numpy_Neural_Network.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

先检查小 batch 梯度，再开始长循环。

</details>

<details>
<summary>第二层：参考解法</summary>

把 dz 的除以 len(X) 暂时去掉，解析梯度会放大 n 倍，而数值梯度不变，assert 应失败；恢复后误差应 <1e−6。固定初始化比较多个学习率，记录训练和验证损失；过大可振荡，过小进展慢。

</details>

<details>
<summary>第三层：自检标准</summary>

y 必须是 (n,1)；恢复正确实现后检查通过，训练损失下降且有限。

</details>
