"""Navigation generated from the prerequisite map; prose/solutions stay editable."""
import json
import os
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
MAP = json.loads((ROOT / "docs/course-map.json").read_text(encoding="utf-8"))
BY_DAY = {row["day"]: row for row in MAP}
BEGINNER_STAGES = [
    ("数组、表格与绘图", [45, 46, 47, 48, 49, 50, 51, 52]),
    ("线代与微积分预备", [26, 27, 28, 29, 53, 30, 31, 32]),
    ("预处理与回归", [1, 2, 3, 4, 5, 8]),
    ("分类与评价", [7, 9, 12, 20, 6, 10, 11, 13, 14, 15, 16, 19]),
    ("树模型、学习理论与数据提取", [23, 25, 33, 34, 24, 22, 21]),
    ("神经网络与实验", [17, 35, 36, 37, 38, 18, 39, 40, 41, 42]),
    ("聚类", [43, 44, 54]),
]


def link(day, current):
    row = BY_DAY[day]
    relative = os.path.relpath(ROOT / row["path"], current.parent).replace(os.sep, "/")
    return f"[Day {day}：{row['title']}]({quote(relative, safe='/.-_')})"


def navigation(day, current):
    deps = BY_DAY[day]["prerequisites"]
    prerequisites = "、".join(link(d, current) for d in deps) or "无其他课程硬性先修；需能读写基本 Python 表达式。"
    route = quote(os.path.relpath(ROOT / "docs/learning-paths.md", current.parent).replace(os.sep, "/"), safe="/.-_")
    answer = quote(os.path.relpath(ROOT / f"docs/solutions/day-{day:02d}.md", current.parent).replace(os.sep, "/"), safe="/.-_")
    return f"**先修导航**：{prerequisites}\n\n[选择学习路线]({route}) · [练习提示、参考答案与自检]({answer})\n"


def theory_outputs():
    outputs = {}
    for row in MAP:
        path = ROOT / row["path"]
        if path.parent.name != "lessons":
            continue
        text = path.read_text(encoding="utf-8")
        start, end = "<!-- course-navigation:start -->", "<!-- course-navigation:end -->"
        if start in text:
            before, remainder = text.split(start, 1)
            _, after = remainder.split(end, 1)
            text = before.rstrip() + "\n\n" + after.lstrip()
        first, rest = text.split("\n", 1)
        outputs[path] = first + "\n\n" + start + "\n" + navigation(row["day"], path) + end + "\n\n" + rest.lstrip()
    path = ROOT / "docs/learning-paths.md"
    text = '''# 选择适合你的学习路线

Day 编号保留原项目历史顺序，不等于严格先修顺序。每课的先修导航给出直接依赖，可继续沿链接补齐基础；[课程目录](curriculum.md)仍按原 Day 编号查找。

## 零基础路线

适合已会 Python 变量、列表、函数、循环，但尚不熟悉数组、矩阵和导数的读者。若这些 Python 语法也不熟悉，请先完成 [Python 官方入门教程](https://docs.python.org/zh-cn/3/tutorial/)。先按[安装说明](setup.md)建立环境。

下面覆盖 54 天，每天只列一次，先修章节都在使用它们之前。每阶段可按自己的进度分多次学习；不要求一天学完。

'''
    for title, days in BEGINNER_STAGES:
        text += f"### {title}\n\n" + " → ".join(link(d, path) for d in days) + "\n\n"
    text += '''## 已有基础路线

适合已掌握 NumPy/pandas、矩阵乘法、概率基础和链式法则的读者。可以按 [Day 1～54 原顺序](curriculum.md)学习，也可按任务选择实现课：

'''
    for title, days in [("回归", [1, 2, 3]), ("分类", [6, 11, 13, 14, 15, 16, 25, 34]),
                        ("神经网络", [18, 39, 40, 41, 42]), ("聚类", [44, 54])]:
        text += f"- {title}：" + " → ".join(link(d, path) for d in days) + "。\n"
    text += '''
跳过理论前先做一次自测：能否说明训练集与测试集的职责、矩阵乘法后的 shape、均值/方差与概率的区别、链式法则如何传播梯度？不能时按该课先修链接回补，而不是跳过缺失知识继续复制代码。

## 如何使用练习答案

先独立尝试，再打开本课“练习提示、参考答案与自检”链接。答案分为提示、参考解法和自检标准三个折叠区。数值题可核对数值与容差；开放实验以数据划分、控制变量、完整记录为标准，不保证某个参数总能得最高分。

Day 39～42 的实验记录使用独立目录，详见[实验保存说明](experiments.md)。真实猫狗训练不是本次材料更新的验证内容。
'''
    outputs[path] = text
    return outputs
