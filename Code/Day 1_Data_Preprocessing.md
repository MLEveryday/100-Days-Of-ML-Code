# 数据预处理

<p align="center">
  <img src="https://github.com/MachineLearning100/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%201.jpg">
</p>

如图所示，通过6步完成数据预处理。

此例用到的[数据](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/Data.csv)，[代码](https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/Code/Day%201_Data_Preprocessing.py)。

## 第1步：导入库
```Python
import numpy as np
import pandas as pd
```
## 第2步：导入数据集
```python
//随后一列是label
dataset = pd.read_csv('Data.csv')//读取csv文件
X = dataset.iloc[ : , :-1].values//.iloc[行，列]
Y = dataset.iloc[ : , 3].values  // : 全部行 or 列；[a]第a行 or 列
                                 // [a,b,c]第 a,b,c 行 or 列
```
## 第3步：处理丢失数据
```python
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
```
## 第4步：解析分类数据
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
```
### 创建虚拟变量
```python
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
```
## 第5步：拆分数据集为训练集合和测试集合
```python
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
```
## 第6步：特征量化
```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```
