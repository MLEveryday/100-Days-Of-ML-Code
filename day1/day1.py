import numpy
import pandas
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# from sklearn.cross_validation import train_test_split 在新版本被移除
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入数据
data = pandas.read_csv('Data.csv')
# 切割数据
X = data.iloc[:, :-1].values
# [['法国' 44.0 72000.0]
#  ['西班牙' 27.0 48000.0]
#  ['德国' 30.0 54000.0]
#  ['西班牙' 38.0 61000.0]
#  ['德国' 40.0 nan]
#  ['法国' 35.0 58000.0]
#  ['西班牙' nan 52000.0]
#  ['法国' 48.0 79000.0]
#  ['德国' 50.0 83000.0]
#  ['法国' 37.0 67000.0]]
Y = data.iloc[:, 3].values
# ['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print('Xt', X_train, '\nXtest', X_test, '\nYtest', Y_test, '\nYtrain', Y_train)
