# 选择适合你的学习路线

Day 编号保留原项目历史顺序，不等于严格先修顺序。每课的先修导航给出直接依赖，可继续沿链接补齐基础；[课程目录](curriculum.md)仍按原 Day 编号查找。

## 零基础路线

适合已会 Python 变量、列表、函数、循环，但尚不熟悉数组、矩阵和导数的读者。若这些 Python 语法也不熟悉，请先完成 [Python 官方入门教程](https://docs.python.org/zh-cn/3/tutorial/)。先按[安装说明](setup.md)建立环境。

下面覆盖 54 天，每天只列一次，先修章节都在使用它们之前。每阶段可按自己的进度分多次学习；不要求一天学完。

### 数组、表格与绘图

[Day 45：NumPy：dtype、shape 与 ufunc](lessons/day-45.md) → [Day 46：NumPy：广播、聚合与布尔掩码](lessons/day-46.md) → [Day 47：NumPy：索引、排序与结构化数据](lessons/day-47.md) → [Day 48：pandas：索引、缺失值与 concat](lessons/day-48.md) → [Day 49：pandas：连接、分组与透视表](lessons/day-49.md) → [Day 50：pandas：字符串、时间与无泄漏特征](lessons/day-50.md) → [Day 51：Matplotlib：线图、散点与误差条](lessons/day-51.md) → [Day 52：Matplotlib：分布、色条和子图](lessons/day-52.md)

### 线代与微积分预备

[Day 26：线性代数：向量与线性变换](lessons/day-26.md) → [Day 27：线性代数：行列式、逆与子空间](lessons/day-27.md) → [Day 28：线性代数：点积与叉积](lessons/day-28.md) → [Day 29：线性代数：特征值与 PCA 预备](lessons/day-29.md) → [Day 53：Matplotlib：三维视图与二维对照](lessons/day-53.md) → [Day 30：微积分：导数与链式法则](lessons/day-30.md) → [Day 31：微积分：极限与隐函数](lessons/day-31.md) → [Day 32：微积分：Taylor 近似与曲率](lessons/day-32.md)

### 预处理与回归

[Day 1：数据预处理](../Code/Day%201_Data_Preprocessing.md) → [Day 2：简单线性回归](../Code/Day%202_Simple_Linear_Regression.md) → [Day 3：多元线性回归](../Code/Day%203_Multiple_Linear_Regression.md) → [Day 4：逻辑回归：从得分到概率](lessons/day-04.md) → [Day 5：逻辑回归：交叉熵与梯度](lessons/day-05.md) → [Day 8：逻辑回归：对数几率与正则化](lessons/day-08.md)

### 分类与评价

[Day 7：KNN：距离、邻居和投票](lessons/day-07.md) → [Day 9：SVM：线性分隔与支持向量](lessons/day-09.md) → [Day 12：SVM：软间隔与核函数](lessons/day-12.md) → [Day 20：正则化、验证与学习率](lessons/day-20.md) → [Day 6：逻辑回归](../Code/Day%206_Logistic_Regression.md) → [Day 10：比较 KNN 与线性 SVM](lessons/day-10.md) → [Day 11：K 近邻分类](../Code/Day%2011_K-NN.md) → [Day 13：线性支持向量机](../Code/Day%2013_SVM.md) → [Day 14：观察 SVM 间隔与 C](../Code/Day%2014_SVM_Margin.md) → [Day 15：朴素贝叶斯与基线比较](../Code/Day%2015_Naive_Bayes.md) → [Day 16：非线性核 SVM](../Code/Day%2016_Kernel_SVM.md) → [Day 19：感知机与学习问题](lessons/day-19.md)

### 树模型、学习理论与数据提取

[Day 23：决策树：熵、分裂与 CART](lessons/day-23.md) → [Day 25：决策树分类](../Code/Day%2025_Decision_Tree.md) → [Day 33：随机森林：抽样、特征与集成](lessons/day-33.md) → [Day 34：随机森林](../Code/Day%2034_Random_Forests.md) → [Day 24：经验风险与泛化风险](lessons/day-24.md) → [Day 22：Hoeffding 不等式与样本量](lessons/day-22.md) → [Day 21：从 HTML 提取结构化数据](../Code/Day%2021_HTML_Data_Collection.md)

### 神经网络与实验

[Day 17：从逻辑回归到神经元](lessons/day-17.md) → [Day 35：神经网络：层、形状与激活](lessons/day-35.md) → [Day 36：梯度下降：batch、epoch 与学习率](lessons/day-36.md) → [Day 37：反向传播：计算图与局部导数](lessons/day-37.md) → [Day 38：反向传播：梯度检查](lessons/day-38.md) → [Day 18：用 NumPy 实现两层神经网络](../Code/Day%2018_Numpy_Neural_Network.md) → [Day 39：MNIST 与 Keras 3](../Code/Day%2039.md) → [Day 40：猫狗数据准备与可复现划分](../Code/Day%2040.md) → [Day 41：小型卷积神经网络](../Code/Day%2041.md) → [Day 42：TensorBoard 与受控实验](../Code/Day%2042.md)

### 聚类

[Day 43：K-means：目标、初始化与局限](lessons/day-43.md) → [Day 44：K-means 聚类实现](../Code/Day%2044_KMeans.md) → [Day 54：层次聚类](../Code/Day%2054_Hierarchical_Clustering.md)

## 已有基础路线

适合已掌握 NumPy/pandas、矩阵乘法、概率基础和链式法则的读者。可以按 [Day 1～54 原顺序](curriculum.md)学习，也可按任务选择实现课：

- 回归：[Day 1：数据预处理](../Code/Day%201_Data_Preprocessing.md) → [Day 2：简单线性回归](../Code/Day%202_Simple_Linear_Regression.md) → [Day 3：多元线性回归](../Code/Day%203_Multiple_Linear_Regression.md)。
- 分类：[Day 6：逻辑回归](../Code/Day%206_Logistic_Regression.md) → [Day 11：K 近邻分类](../Code/Day%2011_K-NN.md) → [Day 13：线性支持向量机](../Code/Day%2013_SVM.md) → [Day 14：观察 SVM 间隔与 C](../Code/Day%2014_SVM_Margin.md) → [Day 15：朴素贝叶斯与基线比较](../Code/Day%2015_Naive_Bayes.md) → [Day 16：非线性核 SVM](../Code/Day%2016_Kernel_SVM.md) → [Day 25：决策树分类](../Code/Day%2025_Decision_Tree.md) → [Day 34：随机森林](../Code/Day%2034_Random_Forests.md)。
- 神经网络：[Day 18：用 NumPy 实现两层神经网络](../Code/Day%2018_Numpy_Neural_Network.md) → [Day 39：MNIST 与 Keras 3](../Code/Day%2039.md) → [Day 40：猫狗数据准备与可复现划分](../Code/Day%2040.md) → [Day 41：小型卷积神经网络](../Code/Day%2041.md) → [Day 42：TensorBoard 与受控实验](../Code/Day%2042.md)。
- 聚类：[Day 44：K-means 聚类实现](../Code/Day%2044_KMeans.md) → [Day 54：层次聚类](../Code/Day%2054_Hierarchical_Clustering.md)。

跳过理论前先做一次自测：能否说明训练集与测试集的职责、矩阵乘法后的 shape、均值/方差与概率的区别、链式法则如何传播梯度？不能时按该课先修链接回补，而不是跳过缺失知识继续复制代码。

## 如何使用练习答案

先独立尝试，再打开本课“练习提示、参考答案与自检”链接。答案分为提示、参考解法和自检标准三个折叠区。数值题可核对数值与容差；开放实验以数据划分、控制变量、完整记录为标准，不保证某个参数总能得最高分。

Day 39～42 的实验记录使用独立目录，详见[实验保存说明](experiments.md)。真实猫狗训练不是本次材料更新的验证内容。
