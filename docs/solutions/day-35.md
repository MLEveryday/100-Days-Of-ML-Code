# Day 35：练习提示与参考答案

[返回本课](../../docs/lessons/day-35.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

每个 Dense 都有输入×输出权重和输出维偏置。

</details>

<details>
<summary>第二层：参考解法</summary>

784→128 层有 784×128+128=100480 参数；再接 128→10 层有 128×10+10=1290，总计 101770。Flatten 无可学习参数。

下面是可独立运行的数值核对：

```python
first = 784*128 + 128
second = 128*10 + 10
assert first + second == 101770
print(first, second, first+second)
```

</details>

<details>
<summary>第三层：自检标准</summary>

batch_size 不影响模型参数数目。

</details>
