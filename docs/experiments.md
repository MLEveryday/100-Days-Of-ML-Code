# 实验保存与复现

Day 39～42 每次准备数据或开始训练，都在 `outputs/experiments/dayXX/时间戳_唯一后缀/` 创建新目录。不会根据相同种子复用目录，也不会覆盖旧模型。`COURSE_OUTPUT_DIR` 可改变输出根目录，相对路径仍以仓库根目录为准。

| 课程 | 保存内容 |
|---|---|
| Day 39 | config.json（种子、样本量、数据哈希、包版本、源文件哈希）、history.json、metrics.json、损失图、architecture.json、`.keras` 模型、completed.json |
| Day 40 | config.json（种子、样本上限和数据位置）、pets_manifest.json、样例图 |
| Day 41 | config.json、manifest.json 清单快照、history.json、metrics.json、classification.json、损失图、architecture.json、`.keras` 模型、completed.json |
| Day 42 | config.json、manifest.json、每种宽度的配置/训练历史/最佳模型/TensorBoard 日志、总 results.json |

文件路径会打印到终端。训练失败时可能留下部分文件，这表示未完成实验，不应当作有效结果。Day 39/41 的 completed.json 在模型保存并通过加载一致性检查后生成；Day 42 的 results.json 标记比较及最终评价完成。配置保存代码哈希用于追溯；它不是代码备份，复现时还应保留对应 Git 提交或工作区补丁。硬件和运行时差异仍可能引入数值波动。

## 选择猫狗数据清单

先运行 Day 40。只有一份新格式清单时，Day 41/42 自动使用并打印它；有多份时会停止并列出候选，要求明确指定，不会静默选择最新的一份。

Bash 示例（将路径替换为 Day 40 实际打印的路径）：

```bash
export COURSE_PET_MANIFEST="outputs/experiments/day40/实际目录名/pets_manifest.json"
python "Code/Day 41.py"
python "Code/Day 42.py"
```

PowerShell 使用 `$env:COURSE_PET_MANIFEST = "路径"`。Notebook 中可在读取数据单元前运行：

```python
import os
os.environ["COURSE_PET_MANIFEST"] = "outputs/experiments/day40/实际目录名/pets_manifest.json"
```

两门课都复制所选清单到自己的实验目录，并检查原始文件哈希。更换图片或样本上限后重新运行 Day 40，会创建新清单，旧清单保持不变。相同内容哈希用于查重，不保证不同编码的相似照片没有泄漏。

旧版 `outputs/pets_manifest.json` 不会被自动选中或删除。若要继续使用，可明确设置 `COURSE_PET_MANIFEST` 指向它，仍会进行文件和划分校验。

## Notebook 重跑与输出保护

- 重跑 Day 39/41 的训练单元会创建新实验目录；再运行其保存单元，将模型写入该新目录。
- 重跑 Day 42 的比较训练单元会创建新的完整对照目录，各宽度不复用旧日志。
- 单独重复运行已经成功写入 JSON/模型的保存单元会报 `FileExistsError`，防止把新状态写进旧实验。请从训练单元重新运行，或仅加载已有模型进行查看。
- Day 42 在**同一次训练内部**允许 ModelCheckpoint 更新本次最佳权重，这是早停/选模的一部分；不会更新其他实验目录中的模型。
- 图形保存在所属实验内；同一单元重新绘图可以刷新本次图形，配置、指标和模型采用防覆盖保存。

查看 TensorBoard：

```bash
tensorboard --logdir outputs/experiments/day42
```

自定义输出根目录时同步调整 `--logdir`。本次只用合成图像测试记录流程，不下载或运行真实猫狗完整训练。
