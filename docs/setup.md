# 环境与运行

## 安装

推荐 Python 3.11 或 3.12。版本文件固定的是可验证的教学基线，不表示它们是最新发行版。基础学习无需安装 TensorFlow；GPU 不是本教程的必需条件。

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 激活方式为 `.venv\Scripts\Activate.ps1`。深度学习第 39～42 天另执行：

```bash
python -m pip install -r requirements-deep.txt
```

Linux CPU 环境也可先安装基础依赖，再运行 `python -m pip install tensorflow-cpu==2.18.1 keras==3.15.1`，替代上面的深度学习安装命令。安装包是否支持当前系统/架构以 TensorFlow 的发行支持为准。不要把旧环境中的包直接混装到新课程环境。

## Python 与 Notebook

```bash
python "Code/Day 1_Data_Preprocessing.py"
python "Code/Day 3_Multiple_Linear_Regression.py"
python -m notebook
```

在 Jupyter 中打开 `Code/` 下的 Notebook，选择当前环境的内核，从第一个单元按顺序运行。可用以下命令注册内核：

```bash
python -m ipykernel install --user --name ml-course --display-name "ML Course"
```

脚本按文件位置定位仓库；Notebook 从当前工作目录向上寻找仓库。可从仓库根目录或 `Code/` 启动。把单独 Notebook 移到仓库外会失去 `course_utils.py` 等依赖，应复制完整仓库。

图形保存在 `outputs/`，脚本打印保存路径。Notebook 可在文件浏览器打开 PNG；桌面交互窗口可设置 `COURSE_SHOW_PLOTS=1`。自动检查使用 `MPLBACKEND=Agg`，无需显示器。模型、图形和日志不会纳入 Git。

## 深度学习顺序

Day 39 使用仓库自带 MNIST，不需要下载、不修改 TensorFlow 安装文件。

Day 40 的真实猫狗图像需按[数据说明](../datasets/readme.md)准备；先生成清单，再运行 Day 41/42：

```bash
python "Code/Day 39.py"
python "Code/Day 40.py"
python "Code/Day 41.py"
python "Code/Day 42.py"
tensorboard --logdir outputs/experiments/day42
```

浏览器打开 `http://localhost:6006`。远程服务器须使用你自己的端口转发方式，本教程不自动开放服务端口。

可设置的环境变量：

| 变量 | 含义 |
|---|---|
| `COURSE_OUTPUT_DIR` | 输出目录，默认仓库的 `outputs/`；相对路径相对于仓库根目录；40～42 必须一致 |
| `COURSE_PET_IMAGES` | 包含 `Dog/`、`Cat/` 的目录，默认 `datasets/PetImages`；相对路径相对于仓库根目录 |
| `COURSE_PET_MANIFEST` | 指定 Day 40 清单；存在多份清单时必填，可选旧格式清单；相对路径以仓库根目录为准 |
| `COURSE_PET_LIMIT` | Day 40 每类最多使用多少张有效图像，默认 500，最少 10 |
| `COURSE_SMOKE=1` | MNIST 少量数据；猫狗每类 20 张；训练 1 epoch，用于验证流程 |
| `COURSE_SHOW_PLOTS=1` | 保存图像后显示窗口 |

Day 39～42 每次运行创建独立实验目录，保存配置、数据依据和结果，详见[实验保存与复现](experiments.md)。Day 40 生成的清单记录数据路径、哈希、类别、划分和坏图；Day 41/42 不重新划分，会先校验清单中所选文件的 SHA-256；数据改动后应创建并记录新实验。改变上限或数据目录后会自动生成新的清单目录；存在多份清单时，必须明确选择 COURSE_PET_MANIFEST。快速验证的精度不具备质量评价意义。

## 修改和验证课程

带 `# %%` 标记的 `Code/Day*.py` 是可执行教学源文件，Markdown 注释写解释，代码单元写实现。不要只改生成的 `.md` 或 `.ipynb`：

```bash
python tools/sync_lessons.py
python tools/sync_lessons.py --check
python tools/check_lessons.py
python tools/check_docs.py
python -m unittest discover -s tests -v
```

检查器依次执行全部基础脚本及对应 Notebook，Notebook 使用当前 Python 的内核，输出写临时目录，不污染教学文件。深度学习额外执行：

```bash
python tools/check_lessons.py --deep
```

深度学习检查生成少量合成图片验证解码、划分、训练、保存、加载、日志管线，同时用本地 MNIST 验证数字分类流程。**合成图片检查不验证真实猫狗分类精度**；完整数据上的学习效果需另外记录。

未标记的 Kafka 和个人实验文件不属于课程检查范围，也不会自动运行。

## 练习与先修导航维护

[学习路线](learning-paths.md)和各课先修导航由 `docs/course-map.json` 生成，修改后运行同步工具。`docs/solutions/day-XX.md` 是可独立编辑的三层练习答案。理论 Markdown 中标记的导航块由工具管理，正文与答案不会被重新生成。
