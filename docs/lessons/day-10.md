# Day 10：比较 KNN 与线性 SVM

<!-- course-navigation:start -->
**先修导航**：[Day 7：KNN：距离、邻居和投票](day-07.md)、[Day 9：SVM：线性分隔与支持向量](day-09.md)、[Day 20：正则化、验证与学习率](day-20.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-10.md)
<!-- course-navigation:end -->

## 学习目标

在相同验证集上观察两种模型的归纳偏好。

## 核心说明

KNN 根据局部相似样本投票，线性 SVM 寻找全局线性边界。两者均对尺度敏感，但对噪声、样本量和预测耗时的反应不同。比较时固定数据划分和指标，预处理应在各训练折内部完成。

本练习使用验证集探索模型，不声称它是最后一次测试。更系统的选择见 Day 11、13、16。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行。

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
X, y = make_moons(n_samples=150, noise=0.15, random_state=0)
Xt, Xv, yt, yv = train_test_split(X, y, stratify=y, random_state=0)
for estimator in [KNeighborsClassifier(5), SVC(kernel="linear")]:
    model = make_pipeline(StandardScaler(), estimator).fit(Xt, yt)
    print(type(estimator).__name__, model.score(Xv, yv))
```

## 结果解读

非线性数据通常不利于单一线性边界，但单次验证结果不能证明一种算法普遍更好。

## 练习

改变噪声和样本数量，记录观察；不要只报告最有利于某模型的一次结果。

[返回完整课程目录](../curriculum.md)
