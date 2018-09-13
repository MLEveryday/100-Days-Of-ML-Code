
# 常见问题解答
欢迎到Issues提问
 
### 1. “拆分数据”和“特征缩放”的顺序
可以先“拆分数据”，再“特征缩放”，但需要使用训练集的fit参数，去transform测试集，以保证参数相同。

详见[issue#41](https://github.com/MachineLearning100/100-Days-Of-ML-Code/issues/41)。

### 2. 3Blue1Brown视频
原作中提到的YouTube视频，在B站有[官方中文版](https://space.bilibili.com/88461692/#/)，README的链接也是到B站。
 
感谢网友在[issue#45](https://github.com/MachineLearning100/100-Days-Of-ML-Code/issues/45)的反馈。

### 3. Deep Learning basics with Python, TensorFlow and Keras
[文字版](https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/)的中文版见第39天到第42天。

详见[issue#52](https://github.com/MachineLearning100/100-Days-Of-ML-Code/issues/52)。

### 4.《Python数据科学手册》
**[高清中文版pdf](https://github.com/MachineLearning100/100-Days-Of-ML-Code/blob/master/Other%20Docs/Python%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E6%89%8B%E5%86%8C.zip)**，[Jupyter notebooks](https://github.com/jakevdp/PythonDataScienceHandbook)。
<br>仅作为个人学习，不能用于商业用途。

### 5. 微信群
见[issue#59](https://github.com/MachineLearning100/100-Days-Of-ML-Code/issues/59)

### 6. 常用工具推荐
见[issue#60](https://github.com/MLEveryday/100-Days-Of-ML-Code/issues/60)

### 7. sklearn版本
sklearn工具包0.19和0.20版本，cross_validation在0.20版本是将被移除的并转移到model_selection包下。要排除这些问题，注意平时运行时的出现warning即可。

见[issue63](https://github.com/MLEveryday/100-Days-Of-ML-Code/issues/63)。

### 8. Python版本
建议使用Python3。使用Python2.7运行示例代码时可能有问题，例如：线性回归如下代码中“1/4”，在Python2.7中1/4=0，可以改成“1.0/4”或者“0.25”。
```python
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 
```
