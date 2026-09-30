# Day 16：练习提示与参考答案

[返回本课](../../Code/Day%2016_Kernel_SVM.md) · [学习路线](../learning-paths.md)

先完成本课练习，再按需展开。开放实验按方法和记录自检，不以“分数越高”作为唯一标准。

<details>
<summary>第一层：提示</summary>

噪声与 gamma 的实验必须在训练/验证阶段完成。

</details>

<details>
<summary>第二层：参考解法</summary>

固定外部测试集；用训练集划出的验证集比较 gamma=[0.1,1,10,100]，保持 C 和拆分不变。较大 gamma 可形成碎片边界，训练成绩提高但验证下降时是过拟合线索。增加噪声是单独一组实验，不与 gamma 同时改变后归因。

</details>

<details>
<summary>第三层：自检标准</summary>

展示训练与验证指标两列；最终测试评价不能反过来决定继续选参。

</details>
