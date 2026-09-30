# Day 51：Matplotlib：线图、散点与误差条

<!-- course-navigation:start -->
**先修导航**：[Day 45：NumPy：dtype、shape 与 ufunc](day-45.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-51.md)
<!-- course-navigation:end -->

## 学习目标

使用显式 fig/ax，正确表达轴和不确定性。

## 核心说明

线图强调有序关系，散点展示成对观测；未排序横坐标连线可能误导。误差条表示测量误差、标准差或置信区间时要明确是哪一种，不能笼统称为“异常处理”。

采用面向对象接口 fig,ax=plt.subplots，使多图状态清晰。样式只影响外观，旧 seaborn-whitegrid 可改为 seaborn-v0_8-whitegrid，或直接不设置。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行，图形保存到当前工作目录；在 Notebook 文件浏览器中打开打印出的 PNG。

```python
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(5)
y = x**2
fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=np.ones(5), fmt="o-", label="measurement ±1 unit")
ax.set(xlabel="Time (s)", ylabel="Distance (m)")
ax.legend()
fig.savefig("day51_example.png", dpi=150, bbox_inches="tight")
print("Saved: day51_example.png")
plt.close(fig)
```

## 结果解读

本例误差为演示常数，不是从样本推断出的置信区间。交互时可在关闭前 plt.show，导出用 fig.savefig。

## 练习

把横坐标打乱，比较 scatter 与 plot；为误差条写一段明确的图注。

[返回完整课程目录](../curriculum.md)
