# 数据集说明

CSV 随原教程提供，用于教学演示；原始收集方法、货币单位及逐样本来源没有完整记录。不要据此作真实业务或因果结论。保留原始值，修订应记录依据。

| 文件 | 样本数 | 特征与目标 | 使用课程 |
|---|---:|---|---|
| Data.csv | 10 | Country 类别；Age 年龄；Salary 薪资；Purchased Yes/No | Day 1，Age/Salary 空值用于填补演示 |
| studentscores.csv | 28 | Hours 学习时长；Scores 分数 | Day 2，统一使用全部 28 行 |
| 50_Startups.csv | 50 | R&D Spend、Administration、Marketing Spend 三类支出；State 州；Profit 利润 | Day 3，零支出保留为合法值 |
| Social_Network_Ads.csv | 400 | User ID、Gender、Age、EstimatedSalary；Purchased 0/1 | 分类课程选 Age/EstimatedSalary；不使用 User ID |
| mnist.npz | 60000 训练、10000 测试 | 28×28 uint8 灰度图；数字标签 0～9 | Day 39 |

studentscores.csv 有偏离简单线性趋势的样本，例如 Hours=2.5、Scores=93；目前缺少原始记录来解释，不能仅为提高指标删除。薪资和支出的单位未明确，不擅自标为美元。

MNIST 文件包含 `x_train`、`y_train`、`x_test`、`y_test` 四个数组，可通过 `np.load(..., allow_pickle=False)` 直接加载；来源背景参见 [MNIST](https://www.tensorflow.org/datasets/catalog/mnist)。从原训练集再划验证集，原测试集保留到最终评价。

## 猫狗图像（Day 40～42）

图像不包含在 Git 仓库。请从 [Microsoft Cats vs. Dogs 数据说明](https://www.microsoft.com/en-us/download/details.aspx?id=54765) 获取当前下载入口，阅读其使用条款并解压。旧教程 ZIP 文件名可能已经变化，本课程不绑定旧下载 URL。

目录应为：

```text
datasets/PetImages/
├── Dog/
│   ├── 0.jpg
│   └── ...
└── Cat/
    ├── 0.jpg
    └── ...
```

也可设置 `COURSE_PET_IMAGES` 指向现有 PetImages。Day 40 检查抽取过程中遇到的坏图、按文件字节 SHA-256 去重并创建训练/验证/测试清单；0=Dog，1=Cat。所选候选中相同文件字节跨类别视为标签冲突并报错，不能静默归类。

默认每类最多取 500 张有效图像，按固定种子划分为 60%/20%/20%；先打乱索引再按批次解码为 64×64 RGB float32；默认双线性缩放。达到样本上限即停止扫描，因此坏图记录不代表全库统计。哈希无法识别不同编码但视觉内容相同的图片。用完整数据学习时调整上限，并保留每次实验的清单。来源网站变更或下载失败时，应从官方入口确认，不能关闭 TLS 验证来掩盖问题。

旧 `X.pickle/y.pickle` 已不再使用。清单、坏图记录、图形和模型按实验保存在 `outputs/experiments/`，不会改写原始图片。多份清单需用 `COURSE_PET_MANIFEST` 明确选择，详见[实验保存](../docs/experiments.md)。
