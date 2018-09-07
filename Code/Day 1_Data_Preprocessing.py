#Day 1: Data Prepocessing

#Step 1: Importing the libraries
import numpy as np
import pandas as pd

#Step 2: Importing dataset
#使用pandas的read_csv方法读取本地文件为数据帧，
#然后从数据帧中制作自变量和因变量的矩阵和向量
dataset = pd.read_csv('../datasets/Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
print("Step 2: Importing dataset")
print("X")
print(X)
print("Y")
print(Y)

#Step 3: Handling the missing data
#为了不降低ml模型的性能，需要处理丢失数据，用sklearn.pre-
#processingk库的imputer类完成这项任务
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print("---------------------")
print("Step 3: Handling the missing data")
print("step2")
print("X")
print(X)

#Step 4: Encoding categorical data
#分类数据指的是含有标签值而不是数字值的变量，使用labelEncoder类解析
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
#Creating a dummy variable
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
print("---------------------")
print("Step 4: Encoding categorical data")
print("X")
print(X)
print("Y")
print(Y)

#Step 5: Splitting the datasets into training sets and Test sets
#使用train_test_split()方法拆分为测试集合和训练集合（20:80）
#现在的版本 cross_validation包下的功能都移到model_selection下面了 包括cross_validation 
#现在0.19应该只是warning,0.20版本可能就会直接报错了 
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
print("---------------------")
print("Step 5: Splitting the datasets into training sets and Test sets")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)

#Step 6: Feature Scaling
#使用StandardScalar类特征标准化或Z值归一化
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("---------------------")
print("Step 6: Feature Scaling")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
