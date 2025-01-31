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
#最后一列是label
dataset = pd.read_csv('Data.csv') #这行代码使用`pandas`的`read_csv`函数读取名为`Data.csv`的CSV文件，并将其内容加载到一个名为`dataset`的DataFrame对象中。DataFrame是`pandas`库中用于处理表格数据的主要数据结构。
X = dataset.iloc[ : , :-1].values #这行代码使用`iloc`方法从`dataset`中提取所有行（`:`表示所有行）和除最后一列之外的所有列（`:-1`表示从第一列到倒数第二列）。`.values`将提取的数据转换为NumPy数组。这个数组通常被称为特征矩阵（`X`），用于机器学习中的输入特征。
Y = dataset.iloc[ : , 3].values  #这行代码使用`iloc`方法从`dataset`中提取所有行（`:`表示所有行）和第4列（索引为3，因为Python的索引从0开始）。`.values`将提取的数据转换为NumPy数组。这个数组通常被称为目标变量（`Y`），用于机器学习中的输出或标签。

#总结：
# `X` 是特征矩阵，包含输入数据的所有特征（除最后一列）。
# `Y` 是目标变量，包含输出数据（通常是最后一列）。
# 而这段代码通常用于数据预处理阶段，为后续的机器学习模型训练准备数据。
```
## 第3步：处理丢失数据
```python
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

#--------------分割线--------------
# 当使用的Python3.13, pip无法安装sklearn.preprocessing， 报错原因是该包已过时
# 解决方案是 pip install scikit-learn
# 如果安装新版本(>0.20), Imputer 在 scikit-learn 中被移除，替代方案是 SimpleImputer,
# 那么改造后的代码如下



from sklearn.impute import SimpleImputer

# 创建 SimpleImputer 实例，使用均值填充 NaN(Not a Number, 也就是缺失的数据)
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# 只对数值列（1:3，即第2列和第3列）进行填充
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
```
## 第4步：解析分类数据
```python
# 使用scikit-learn库中的LabelEncoder和OneHotEncoder来进行数据预处理
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 创建一个LabelEncoder对象，命名为labelencoder_X。LabelEncoder用于将分类数据（通常是字符串或类别）转换为数值形式。
labelencoder_X = LabelEncoder()

# 这行代码对特征矩阵X的第一列（索引为0）进行编码。具体来说：
#
# X[ : , 0] 表示X的所有行和第一列。
# labelencoder_X.fit_transform(X[ : , 0]) 对第一列的数据进行拟合（fit）和转换（transform），
# 将分类数据转换为数值形式。例如，如果第一列包含类别['A', 'B', 'C']，LabelEncoder可能会将其转换为[0, 1, 2]。
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
```
### 创建虚拟变量
```python
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)

#--------------------分割线---------------
#  IDE 提示 Unexpected argument，是因为 OneHotEncoder 的 categorical_features 参数
#  已被移除（从 scikit-learn 0.22+ 开始被废弃）。更建议以下写法
from sklearn.compose import ColumnTransformer
column_transformer = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])],  # 对第 0 列（Country）进行 OneHot 编码
    remainder='passthrough'  # 其他列保持不变
)
# column_transformer：这是一个 ColumnTransformer 对象（假设在之前的代码中已经定义），用于对数据集中的不同列应用不同的预处理步骤（例如标准化、独热编码等）。
# fit_transform(X)：fit_transform 方法会先拟合（fit）数据，然后对数据进行转换（transform）。它会根据 column_transformer 中定义的预处理步骤对 X 进行处理。
# np.array(...)：将转换后的结果转换为 NumPy 数组。column_transformer.fit_transform(X) 返回的结果可能是一个稀疏矩阵或 DataFrame，这里将其显式转换为 NumPy 数组以便后续使用。
# 总结：这行代码的作用是对特征矩阵 X 进行预处理（如标准化、独热编码等），并将结果转换为 NumPy 数组。
X = np.array(column_transformer.fit_transform(X))

# LabelEncoder()：创建一个 LabelEncoder 对象，命名为 labelencoder_Y。LabelEncoder 用于将分类数据（通常是字符串或类别）转换为数值形式。
# 总结：这行代码的作用是初始化一个 LabelEncoder 对象，用于对目标变量 Y 进行编码。
labelencoder_Y = LabelEncoder()

#fit_transform(Y)：fit_transform 方法会先拟合（fit）目标变量 Y，然后对其进行转换（transform）。LabelEncoder 会将 Y 中的分类数据（如字符串标签）转换为数值形式。例如，如果 Y 包含类别 ['Yes', 'No', 'Yes']，LabelEncoder 可能会将其转换为 [1, 0, 1]。
Y =  labelencoder_Y.fit_transform(Y)
```
## 第5步：拆分数据集为训练集合和测试集合
```python
#from sklearn.model_selection import train_test_split #新版
from sklearn.cross_validation import train_test_split #旧版

#使用 scikit-learn 库中的 train_test_split 函数，用于将数据集划分为训练集和测试集。
# 参数说明：
# X：特征矩阵（输入数据），通常是 NumPy 数组或 DataFrame。
# Y：目标变量（输出数据或标签），通常是 NumPy 数组或 Series。
# test_size=0.2：指定测试集的比例。这里设置为 0.2，表示将 20% 的数据划分为测试集，剩下的 80% 作为训练集。
# random_state=0：设置随机种子，确保每次运行代码时数据集的划分结果一致。如果不设置 random_state，每次运行代码时划分结果可能会不同。
# 返回值：
# X_train：训练集的特征矩阵。
# X_test：测试集的特征矩阵。
# Y_train：训练集的目标变量。
# Y_test：测试集的目标变量。
# 作用：
# 这行代码的作用是将数据集 X 和 Y 划分为训练集和测试集，以便后续的模型训练和评估。
# 训练集（X_train 和 Y_train）用于训练机器学习模型。
# 测试集（X_test 和 Y_test）用于评估模型的性能，确保模型在未见过的数据上也能表现良好。
# 总结：
# train_test_split 是机器学习中常用的数据划分工具，用于将数据集划分为训练集和测试集。
# 这行代码将数据集划分为 80% 的训练集和 20% 的测试集，并确保划分结果可重复（通过 random_state=0）。
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
```
## 第6步：特征量化
```python
from sklearn.preprocessing import StandardScaler
#StandardScaler 用于对数据进行标准化处理，即将数据转换为均值为 0、标准差为 1 的分布。
sc_X = StandardScaler()
# fit_transform 方法会先拟合（fit）训练集 X_train，计算其均值和标准差，然后对 X_train 进行标准化转换（transform）。标准化后的数据将具有均值为 0、标准差为 1 的特性。
X_train = sc_X.fit_transform(X_train)
# 使用训练集计算出的均值和标准差对测试集 X_test 进行标准化转换。注意，这里只调用 transform 而不是 fit_transform，因为测试集的标准化应该使用训练集的参数（均值和标准差），以确保数据分布一致。
X_test = sc_X.transform(X_test)
```
