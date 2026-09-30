# 机器学习100天

[零基础／已有基础学习路线](docs/learning-paths.md) · [实验保存与复现](docs/experiments.md)

本项目基于 [Avik-Jain 的英文项目](https://github.com/Avik-Jain/100-Days-Of-ML-Code)，提供中文机器学习学习路线、讲解与练习。

**当前覆盖 Day 1～54，Day 55～100 尚未编写。** 现有课程已更新数据预处理、模型评价、现代 scikit-learn 和 Keras 3 用法；理论日提供本地说明，图片与外部视频作为辅助资料。

## 开始学习

- [完整的逐日课程目录](docs/curriculum.md)：每天的目标、代码或练习。
- [环境安装与运行](docs/setup.md)：Python 3.11/3.12、基础和深度学习依赖、Notebook 内核。
- [数据集说明](datasets/readme.md)：字段、来源限制、MNIST 和猫狗数据准备。
- [常见问题](FAQ.MD)：数据泄漏、路径、输出图形、模型格式。
- [外部教材兼容说明](docs/readings.md)：旧版书籍示例的替代 API。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python "Code/Day 1_Data_Preprocessing.py"
```

脚本默认把图形保存在 `outputs/`。打开 Notebook 时按顺序运行所有单元；完整说明见安装文档。深度学习依赖单独安装，基础课程不需要 GPU。

## 学习路线

| 阶段 | 内容 |
|---|---|
| Day 1～3 | 预处理、简单与多元线性回归 |
| Day 4～16 | 逻辑回归、KNN、SVM、朴素贝叶斯与核方法 |
| Day 17～25 | 神经网络基础、学习理论、HTML 数据提取与决策树 |
| Day 26～38 | 线性代数、微积分、随机森林与反向传播 |
| Day 39～42 | MNIST、猫狗 CNN、TensorBoard 与受控实验 |
| Day 43～54 | 聚类、NumPy、pandas、Matplotlib 与层次聚类 |

## 参与维护

实现课以 `Code/Day*.py` 中的分步代码与 Markdown 注释为源，运行 `python tools/sync_lessons.py` 生成配套 `.md` 和 `.ipynb`。理论日编辑 `docs/lessons/`。验证方式见[安装文档](docs/setup.md)。

请阅读[翻译与维护规范](Translation%20specification.MD)。[更新与验证记录](docs/modernization.md)说明本轮修正范围；[逐日审阅报告](docs/day-by-day-review.md)保留修正前的问题基线；[原始学习日志](docs/original-roadmap.md)保留原项目的历史路线。
