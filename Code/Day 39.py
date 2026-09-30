# %% [markdown]
# # Day 39：MNIST 与 Keras 3
#
# 目标：完成本地数据读取、训练/验证/测试、模型保存和加载。前置：Day 18、35～38。使用稳定 TensorFlow/Keras 依赖，不修改安装包源码，不需要网络下载 MNIST。
#
# 运行前请阅读[环境与运行说明](../docs/setup.md)。本课 `.py` 是教学源文件，配套 Markdown 和 Notebook 自动同步。图形保存到 `outputs/`，设置 `COURSE_SHOW_PLOTS=1` 可显示窗口。

# %%
from pathlib import Path
import sys

# 脚本从文件位置定位仓库；Notebook 从当前工作目录向上查找。
base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for candidate in (base, *base.parents):
    if (candidate / "Code" / "course_utils.py").is_file():
        code_dir = str(candidate / "Code")
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        break
else:
    raise FileNotFoundError("找不到课程仓库，请从仓库根目录或 Code 目录启动 Notebook。")
from course_utils import DATA, OUTPUT, finish_plot


# %% [markdown]
# ## 读取并缩放像素
#
# MNIST 已放在 datasets/mnist.npz。图像 uint8 像素转换为 float32 并除以 255；这是像素缩放，不是按行 L2 归一化。标签 0～9 使用 sparse categorical crossentropy。快速验证设 COURSE_SMOKE=1，仅验证流程，不代表模型质量。

# %%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from deep_utils import configure, array_dataset
configure()
from experiments import new_experiment, write_record
with np.load(DATA / "mnist.npz", allow_pickle=False) as source:
    X, y = source["x_train"], source["y_train"]
    X_test, y_test = source["x_test"], source["y_test"]
if os.environ.get("COURSE_SMOKE") == "1":
    X, y, X_test, y_test = X[:600], y[:600], X_test[:100], y_test[:100]
X = X.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
valid = array_dataset(X_valid, y_valid)
test = array_dataset(X_test, y_test)
print("Train/validation/test:", X_train.shape, X_valid.shape, X_test.shape)

# %% [markdown]
# ## 构建网络并训练
#
# 每次运行本训练单元都会创建独立实验目录，保存配置、训练历史和指标；保存模型单元使用同一目录。详见[实验保存](../docs/experiments.md)。
#
# Flatten 将 28×28 展开成 784 个特征；Dense(128) 使用 ReLU，10 类 softmax 输出总和为 1。验证集用于早停，测试集仅做最终评价。

# %%
import hashlib
configure()  # 单独重跑训练单元也重置种子和训练顺序
train = array_dataset(X_train, y_train, training=True)
epochs = 1 if os.environ.get("COURSE_SMOKE") == "1" else 5
run_dir = new_experiment("day39", {"seed": 42, "smoke": os.environ.get("COURSE_SMOKE") == "1", "epochs_max": epochs, "optimizer": "adam", "batch_size": 32, "train_size": len(X_train), "validation_size": len(X_valid), "test_size": len(X_test), "data_sha256": hashlib.sha256((DATA / "mnist.npz").read_bytes()).hexdigest()})
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
write_record(run_dir / "architecture.json", json.loads(model.to_json()))
history = model.fit(train, validation_data=valid, epochs=epochs, verbose=2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
test_metrics = model.evaluate(test, verbose=0, return_dict=True)
print("Held-out test:", test_metrics)
write_record(run_dir / "history.json", history.history)
write_record(run_dir / "metrics.json", test_metrics)
fig, ax = plt.subplots()
ax.plot(history.history["loss"], label="train")
ax.plot(history.history["val_loss"], label="validation")
ax.set(xlabel="Epoch", ylabel="Cross-entropy")
ax.legend()
finish_plot("day39_loss", run_dir)

# %% [markdown]
# ## 保存、加载和预测一致性
#
# Keras 3 使用 `.keras` 保存可重新加载的完整模型。SavedModel 部署导出使用 `model.export`，不是这里的保存/加载流程。不同硬件可能出现浮点差异，用合理容差比较。

# %%
model_path = run_dir / "day39_mnist.keras"
if model_path.exists():
    raise FileExistsError("模型已保存；请重跑训练单元创建新实验，不覆盖旧模型。")
model.save(model_path)
restored = tf.keras.models.load_model(model_path)
before = model(X_test[:10], training=False).numpy()
after = restored(X_test[:10], training=False).numpy()
np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
print("Predicted digits:", after.argmax(axis=1), "actual:", y_test[:10])
write_record(run_dir / "completed.json", {"saved_model": model_path.name, "reload_predictions_match": True})

# %% [markdown]
# ## 练习与检查
#
# 比较小样本与完整训练的差异。解释为何不要把 softmax 输出直接当作校准良好的置信度。检查一张错分图和它的类别概率，不仅报告总准确率。
