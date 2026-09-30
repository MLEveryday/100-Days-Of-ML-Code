# Day 19：感知机与学习问题

<!-- course-navigation:start -->
**先修导航**：[Day 9：SVM：线性分隔与支持向量](day-09.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-19.md)
<!-- course-navigation:end -->

## 学习目标

理解在线更新、线性可分假设和局限。

## 核心说明

感知机按 sign(wᵀx+b) 分类。对标签 y∈{−1,+1}，若 y(wᵀx+b)≤0，则更新 w←w+ηyx，b←b+ηy。数据线性可分时存在收敛保证；不可分时可能持续更新。

损失、模型、优化和评价是不同层次。经典课程 [Caltech Learning From Data](https://work.caltech.edu/telecourse.html) 可作延伸，课程年代不意味着数学失效。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
import numpy as np
X = np.array([[0., 0.], [1., 1.], [2., 1.]])
y = np.array([-1, 1, 1])
w, b = np.zeros(2), 0.
for epoch in range(20):
    errors = 0
    for x, label in zip(X, y):
        if label*(w@x+b) <= 0:
            w += label*x
            b += label
            errors += 1
    if errors == 0:
        break
print(w, b, epoch)
```

## 结果解读

最大迭代数避免不可分数据无限循环；训练收敛不等于验证表现良好。

## 练习

加入相同输入但相反标签，观察无法全部分对的情况；说明实际工程为何要有停止条件。

[返回完整课程目录](../curriculum.md)
