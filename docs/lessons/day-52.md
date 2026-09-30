# Day 52：Matplotlib：分布、色条和子图

<!-- course-navigation:start -->
**先修导航**：[Day 51：Matplotlib：线图、散点与误差条](day-51.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-52.md)
<!-- course-navigation:end -->

## 学习目标

选择合适的视觉编码并避免误读。

## 核心说明

直方图受 bin 边界影响，density=True 表示密度，面积积分约为 1，不是每根柱子的高度都为概率。连续数值色条和离散类别图例表达不同信息。

子图比较应保持轴范围/单位可比；使用 constrained_layout 帮助布局。顺序色图适合从低到高，发散色图适合围绕有意义中心的正负偏差。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行，图形保存到当前工作目录；在 Notebook 文件浏览器中打开打印出的 PNG。

```python
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
x = rng.normal(size=200)
fig, axes = plt.subplots(1, 2, sharey=True, constrained_layout=True)
for ax, bins in zip(axes, [8, 25]):
    ax.hist(x, bins=bins, density=True)
    ax.set(xlabel="Value", ylabel="Density", title=f"bins={bins}")
fig.savefig("day52_example.png", dpi=150, bbox_inches="tight")
print("Saved: day52_example.png")
plt.close(fig)
```

## 结果解读

同一批数据只改分箱就会产生不同视觉细节，不应挑最支持结论的图。

## 练习

计算柱高×宽度之和；导出 150 dpi PNG 和 SVG，比较不同用途。

[返回完整课程目录](../curriculum.md)
