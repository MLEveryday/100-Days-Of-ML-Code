# Day 22：练习提示与参考答案

[返回本课](../../docs/lessons/day-22.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

对 2exp(−2nε²)≤δ 取对数。

</details>

<details>
<summary>第二层：参考解法</summary>

n≥ln(2/δ)/(2ε²)。ε=0.1、δ=0.05 时约 184.444，向上取整为 185。相关样本不满足该独立样本推导，重复复制样本不会真的把有效样本量提高。

下面是可独立运行的数值核对：

```python
import math
n = math.ceil(math.log(2/0.05)/(2*0.1**2))
assert n == 185
assert 2*math.exp(-2*n*0.1**2) <= 0.05
print(n)
```

</details>

<details>
<summary>第三层：自检标准</summary>

代回 n=185 后上界≤0.05；向下取整不能保证该不等式。

</details>
