# Day 20：正则化、验证与学习率

<!-- course-navigation:start -->
**先修导航**：[Day 1：数据预处理](../../Code/Day%201_Data_Preprocessing.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-20.md)
<!-- course-navigation:end -->

## 学习目标

设计不依赖测试集调参的实验。

## 核心说明

训练集用于更新参数，验证集用于选超参数，测试集用于选定方案后的评价。多次根据测试成绩调参会让测试集变成验证集。交叉验证也应把填补和缩放包含在 Pipeline 内。

L2 惩罚大权重，dropout 在训练阶段随机屏蔽部分激活，早停根据验证表现限制训练时长；三者机制不同。下面用 Ridge 演示正则化选择，神经网络早停见 Day 39/42。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
X, y = make_regression(n_samples=100, n_features=10, noise=20, random_state=0)
Xt, Xs, yt, ys = train_test_split(X, y, random_state=0)
search = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.1, 1., 10., 100.]}, cv=5)
search.fit(Xt, yt)
print(search.best_params_, search.score(Xs, ys))
```

## 结果解读

最优参数来自训练折；最终测试 R² 只报告，不用于再次挑 alpha。

## 练习

写一张实验表，固定划分和预算，只改一个参数。解释 dropout 推理时为何不继续随机丢弃。

[返回完整课程目录](../curriculum.md)
