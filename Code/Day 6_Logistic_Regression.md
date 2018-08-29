<p align="center">
  <img src="https://github.com/MachineLearning100/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%204.jpg?raw=true">
</p>

### 数据集 | 社交网络

<p align="center">
  <img src="https://github.com/MachineLearning100/100-Days-Of-ML-Code/blob/master/Other%20Docs/data.png?raw=true">
</p>

该数据集包含了社交网络中用户的信息。这些信息涉及用户ID,性别,年龄以及预估薪资。一家汽车公司刚刚推出了他们新型的豪华SUV，我们尝试预测哪些用户会购买这种全新SUV。并且在最后一列用来表示用户是否购买。我们将建立一种模型来预测用户是否购买这种SUV，该模型基于两个变量，分别是年龄和预计薪资。因此我们的特征矩阵将是这两列。我们尝试寻找用户年龄与预估薪资之间的某种相关性，以及他是否购买SUV的决定。

### 步骤1 | 数据预处理
#### 导入库

```python
import numpy as numpy
import matplotlib.pyplot as plt
import pandas as pd
```

#### 导入数据集
[这里](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/Social_Network_Ads.csv)获取数据集 

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:,4].values
```

#### 将数据集分成训练集和测试集

```python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
```

#### 特征缩放

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### 步骤2 | 逻辑回归模型

该项工作的库将会是一个线性模型库，之所以被称为线性是因为逻辑回归是一个线性分类器，这意味着我们在二维空间中，我们两类用户（购买和不购买）将被一条直线分割。然后导入逻辑回归类。下一步我们将创建该类的对象，它将作为我们训练集的分类器。

#### 将逻辑回归应用于训练集

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
```

### 步骤3 | 预测

#### 预测测试集结果
```python
y_pred = classifier.predict(X_test)
```

### 步骤4 | 评估预测

我们预测了测试集。 现在我们将评估逻辑回归模型是否正确的学习和理解。因此这个混淆矩阵将包含我们模型的正确和错误的预测。

#### 生成混淆矩阵

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

#### 可视化

![](https://github.com/MachineLearning100/100-Days-Of-ML-Code/blob/master/Other%20Docs/LR_training.png?raw=true)
![](https://github.com/MachineLearning100/100-Days-Of-ML-Code/blob/master/Other%20Docs/LR_test.png?raw=true) 























