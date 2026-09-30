# Day 41：练习提示与参考答案

[返回本课](../../Code/Day%2041.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

Conv2D 参数=(kh·kw·cin+1)·cout。

</details>

<details>
<summary>第二层：参考解法</summary>

两卷积层参数分别 (3×3×3+1)×16=448、(3×3×16+1)×32=4640；Dense 为 (32+1)×16=528 和 (16+1)×1=17，总计 5633。训练损失下降而验证上升提示过拟合，可考虑更多训练数据、正则化或较早停止，但需在验证集比较。

下面是可独立运行的数值核对：

```python
parameters = (3*3*3+1)*16 + (3*3*16+1)*32 + (32+1)*16 + (16+1)*1
assert parameters == 5633
print(parameters)
```

</details>

<details>
<summary>第三层：自检标准</summary>

summary 应为 5633 可训练参数；早停看 val_loss，不能看测试损失挑轮数。

</details>
