# Day 53：Matplotlib：三维视图与二维对照

<!-- course-navigation:start -->
**先修导航**：[Day 51：Matplotlib：线图、散点与误差条](day-51.md)、[Day 26：线性代数：向量与线性变换](day-26.md)

[选择学习路线](../learning-paths.md) · [练习提示、参考答案与自检](../solutions/day-53.md)
<!-- course-navigation:end -->

## 学习目标

理解投影遮挡和视角影响。

## 核心说明

三维图通过投影显示空间结构，会遮挡部分点并改变视觉距离；旋转视角能帮助探索，但静态报告通常还需二维投影或等高线对照。

用 projection="3d" 创建坐标轴，明确三个轴的单位。网格分辨率越大，渲染和存储开销越高，先从小网格开始。

## 最小示例

环境见[安装与运行](../setup.md)。下列代码块可作为独立单元运行，图形保存到当前工作目录；在 Notebook 文件浏览器中打开打印出的 PNG。

```python
import numpy as np
import matplotlib.pyplot as plt
x, y = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
z = x*x+y*y
fig = plt.figure(figsize=(9, 4))
ax3 = fig.add_subplot(121, projection="3d")
ax3.plot_surface(x, y, z, cmap="viridis")
ax3.set(xlabel="x", ylabel="y", zlabel="z")
ax2 = fig.add_subplot(122)
ax2.contour(x, y, z)
ax2.set(xlabel="x", ylabel="y", aspect="equal")
fig.savefig("day53_example.png", dpi=150, bbox_inches="tight")
print("Saved: day53_example.png")
plt.close(fig)
```

## 结果解读

抛物面的等高线为圆形，二维图往往更容易比较距离。

## 练习

改变三维视角但保持数据不变，记录结论是否受视觉影响；选择最适合解释曲率的视图。

[返回完整课程目录](../curriculum.md)
