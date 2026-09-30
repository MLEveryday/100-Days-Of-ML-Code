# 常见问题

## 应先划分数据，还是先预处理？

先划分。填补均值、标准化参数、类别集合等都只能从训练集学习，测试集只调用 `transform`。交叉验证时用 Pipeline 让每一折的预处理也只学习该折训练数据。见 [Day 1](Code/Day%201_Data_Preprocessing.md)。

## 应安装哪个版本？

按[环境说明](docs/setup.md)在新虚拟环境安装版本文件。支持的教学基线为 Python 3.11/3.12；不再兼容 Python 2 或 sklearn 0.19。课程使用 `model_selection`、`SimpleImputer`、`ColumnTransformer` 和现代 `OneHotEncoder`，无需寻找旧包绕过错误。

## 为什么运行结果和历史截图不同？

数据范围、划分、随机种子、默认参数和依赖版本都会影响结果。现在的课程统一了脚本与 Notebook 的代码，旧截图不作为当前结果基准。比较实验应保持数据划分和评价方式不变。

## 为什么图没有弹出？

默认保存到 `outputs/` 并打印路径，适用于 Notebook、终端与无显示器服务器。要显示桌面窗口可设置 `COURSE_SHOW_PLOTS=1`；Notebook 可在文件浏览器打开输出 PNG。

## 为什么找不到数据或共享模块？

请保留完整仓库，从根目录或 Code 目录打开 Notebook。猫狗数据需自行准备，并先运行 Day 40；不要把 Notebook 单独复制到另一个目录。见[数据说明](datasets/readme.md)。

## 缺少 X.pickle 或旧模型无法加载？

现代课程不再使用 pickle 图像数组。Day 40 输出 JSON 清单，后续按批次读取图像。Keras 3 的完整模型保存为 `.keras`；旧 SavedModel 与新格式的加载规则不同，参见 [Day 39](Code/Day%2039.md)。

## 怎样修改课程？

实现课编辑带单元标记的 `.py`，再执行 `python tools/sync_lessons.py` 同步 md 与 Notebook。理论课编辑 docs/lessons 中的 Markdown。提交前按[运行说明](docs/setup.md)检查，避免只改三份材料中的一份。

## 理论视频和外部书籍是否必需？

每个学习日均有本地目标、摘要和练习。视频、旧书及速查表作为延伸阅读，其 API 不保证与当前环境一致。新代码以本仓库版本和官方文档为准；参考 [现代工具阅读说明](docs/readings.md)。

## 多个猫狗清单或保存时 FileExistsError 怎么办？

新版本每次生成独立实验；多个清单须指定 `COURSE_PET_MANIFEST`。指标和模型不覆盖已存在的记录，重跑训练单元会创建新目录。详见[实验保存](docs/experiments.md)。

## 不懂前置知识、练习没有思路怎么办？

每课都有先修导航和三层答案入口。先按[零基础路线](docs/learning-paths.md)补齐数组、矩阵和导数，练习先看提示，再看参考解法，最后用自检标准核对。
