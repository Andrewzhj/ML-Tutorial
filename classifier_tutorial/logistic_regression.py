# -*- coding: UTF-8 -*-
'''
@author: Andrewzhj
@contact: andrew_zhj@126.com
@file: logistic_regression.py
@time: 3/25/19 10:37 PM
@desc:
@note:
'''

import pandas as pd
import numpy as np
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
ssl._create_default_https_context = ssl._create_unverified_context


# 良/恶性乳腺癌肿瘤数据训练
class LogisticRegressionTrain(object):
    '''
        初始化数据地址
    '''

    def __init__(self):
        self.bathUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/"
        self.dataFile = "breast-cancer-wisconsin.data"
        self.column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
            'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
            'Blan Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    '''
        获取数据
    '''

    def get_data(self):
        url = self.bathUrl + self.dataFile
        # 读取数据
        data = pd.read_csv(url, names=self.column_names)
        # 将?转换为标准缺失值
        data = data.replace(to_replace='?', value=np.nan)
        # 丢弃带有缺失值的数据
        data = data.dropna(how='any')
        return data

    '''
        逻辑回归训练
        精确计算逻辑回归参数
    '''

    def logistic_train(self):
        data = self.get_data()
        # 创建特征列表
        x_train, x_test, y_train, y_test = train_test_split(data[self.column_names[1:10]], data[self.column_names[10]],
                                                            test_size=0.25, random_state=33)
        print(y_train.value_counts())
        print(y_test.value_counts())
        print(data.shape)
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        lr_y_predict = lr.predict(x_test)
        print("Accuracy of LR Classifier: ", lr.score(x_test, y_test))
        print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))

    '''
        随机梯度参数估计
        采用梯度法估计参数
    '''

    def sgdc_train(self):
        data = self.get_data()
        # 创建特征列表
        x_train, x_test, y_train, y_test = train_test_split(data[self.column_names[1:10]], data[self.column_names[10]],
                                                            test_size=0.25, random_state=33)
        print(y_train.value_counts())
        print(y_test.value_counts())
        print(data.shape)
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
        sgdc = SGDClassifier()
        sgdc.fit(x_train, y_train)
        sgdc_y_predict = sgdc.predict(x_test)
        print("Accuracy of SGD Classifier: ", sgdc.score(x_test, y_test))
        print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))

    '''
    支持向量机分类器
    '''
    def svc_train(self):
        data = self.get_data()
        # 创建特征列表
        x_train, x_test, y_train, y_test = train_test_split(data[self.column_names[1:10]], data[self.column_names[10]],
                                                            test_size=0.25, random_state=33)
        print(y_train.value_counts())
        print(y_test.value_counts())
        print(data.shape)
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
        lsvc = LinearSVC()
        lsvc.fit(x_train, y_train)
        y_predict = lsvc.predict(x_test)
        print("The Accuracy of Liner SVC is", lsvc.score(x_test, y_test))
        print(classification_report(y_test, y_predict))

    def decision_tree_train(self):
        data = self.get_data()
        # 创建特征列表
        x_train, x_test, y_train, y_test = train_test_split(data[self.column_names[1:10]], data[self.column_names[10]],
                                                            test_size=0.25, random_state=33)
        print(y_train.value_counts())
        print(y_test.value_counts())
        print(data.shape)
        vec = DictVectorizer(sparse=False)
        x_train = vec.fit_transform(x_train.to_dict(orient='record'))
        print(vec.feature_names_)
        x_test = vec.transform(x_test.to_dict(orient='record'))
        dtc = DecisionTreeClassifier()
        dtc.fit(x_train, y_train)
        y_predict = dtc.predict(x_test)
        print("The Accuracy of decision tree is ", dtc.score(x_test, y_test))
        print(classification_report(y_test, y_predict))


if __name__ == '__main__':
    cancerTrain = LogisticRegressionTrain()
    # cancerTrain.logistic_train()
    # cancerTrain.sgdc_train()
    # cancerTrain.svc_train()
    cancerTrain.decision_tree_train()