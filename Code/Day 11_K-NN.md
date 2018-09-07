# K近邻法 (K-NN)

<p align="center">
  <img src="https://github.com/MachineLearning100/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%207.jpg">
</p>

## 数据集 | 社交网络

<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Other%20Docs/data.PNG">
</p> 


## 导入相关库
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## 导入数据集
```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
```

## 将数据划分成训练集和测试集
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```
## 特征缩放
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
## 使用K-NN对训练集数据进行训练
```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
```
## 对测试集进行预测
```python
y_pred = classifier.predict(X_test)
```

## 生成混淆矩阵
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```
