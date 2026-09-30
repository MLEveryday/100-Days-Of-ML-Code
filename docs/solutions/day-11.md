# Day 11：练习提示与参考答案

[返回本课](../../Code/Day%2011_K-NN.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

单个混淆矩阵只描述一个阈值；ROC 使用连续或离散概率分数的多个阈值。

</details>

<details>
<summary>第二层：参考解法</summary>

先用 predict_proba 得正类分数，再调用 roc_curve(y_test, scores)，每个阈值得到 FPR=FP/(FP+TN)、TPR=TP/(TP+FN)。KNN 概率可能取值有限，所以曲线呈阶梯状。非参数描述模型形式，不代表没有 K/weights 等超参数。

</details>

<details>
<summary>第三层：自检标准</summary>

ROC 的点来自同一固定模型；不要把一个矩阵的四个数字直接当成整条曲线。

</details>
