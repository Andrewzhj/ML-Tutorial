# 分类算法

## 1.线性分类（Linear Classifier）

### 1.1 逻辑回归（LogisticRegression）

- 假设数据特征与分类目标之间存在线性关系。
- 主要用于离散变量的分类，常用来预测概率。
- 精确解析求值，计算量大，性能消耗高（与SGDClassifier相比）
- 基于损失函数最小化，使用最小二乘法对参数精确求解



### 1.2 随机梯度参数估计（SGDClassifier）

- 与LogisticRegression的主要区别在于求值方法使用了随机梯度下降估计模型参数
- 10万以上的量级推荐使用
- 两种参数搜索方式：1、二分法搜素；2、回溯法线性搜索。


### 1.3 支持向量机(Support Vector Classifier)

- 线性可分支持向量机（硬间隔支持向量机）：训练数据线性可分
- 线性支持向量机（软件间隔支持向量机）：训练数据近似线性可分
- 非线性支持向量机：训练数据不可分（核技巧和软间隔最大化）






# 实践

良/恶性乳腺癌肿瘤预测

## 数据来源
<https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/>  

