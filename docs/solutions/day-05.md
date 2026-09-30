# Day 5：练习提示与参考答案

[返回本课](../../docs/lessons/day-05.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

固定 b，只扰动 w；中心差分的分母是 2h。

</details>

<details>
<summary>第二层：参考解法</summary>

初始 p 全为 0.5，dL/dw=−1/6，dL/db=1/6。学习率 0.1 更新后 w≈0.016667、b≈−0.016667。分别对 w±1e−5 计算平均交叉熵，中心差分应接近 −1/6。循环训练时每步重算 z/p/梯度，记录各学习率的损失。

下面是可独立运行的数值核对：

```python
import numpy as np
X = np.array([[0.], [1.], [2.]])
y = np.array([0., 0., 1.])
w, b, h = np.zeros(1), 0., 1e-5

def loss(weight):
    z = X @ weight + b
    return np.mean(np.logaddexp(0, z) - y*z)

p = np.exp(-np.logaddexp(0, -(X @ w+b)))
analytic = X.T @ (p-y)/len(y)
numeric = (loss(w+h)-loss(w-h))/(2*h)
np.testing.assert_allclose(analytic[0], numeric, atol=1e-6)
np.testing.assert_allclose(analytic[0], -1/6, atol=1e-12)
print("Gradient:", analytic[0], "finite difference:", numeric)
```

</details>

<details>
<summary>第三层：自检标准</summary>

初始损失约 0.693147；差分与解析梯度误差应很小，例如 <1e−6。学习率效果从训练/验证曲线判断，不挑测试成绩。

</details>
