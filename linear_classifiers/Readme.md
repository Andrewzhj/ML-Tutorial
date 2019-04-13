# 分类算法

## 1.线性分类（Linear Classifier）

### 1.1 逻辑回归（LogisticRegression）

- 假设数据特征与分类目标之间存在线性关系。
- 主要用于离散变量的分类，常用来预测概率。
- 精确解析求值，计算量大，性能高（与SGDClassifier相比）


原理：  
使用逻辑函数拟合条件概率P(y=1/x)。


### 1.2 随机梯度参数估计（SGDClassifier）

- 与LogisticRegression的主要区别在于求值方法使用了随机梯度上升估计模型参数
- 10万以上的量级推荐使用


## 2. 支持向量机



# 实践

良/恶性乳腺癌肿瘤预测

## 数据来源
<https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/>  

