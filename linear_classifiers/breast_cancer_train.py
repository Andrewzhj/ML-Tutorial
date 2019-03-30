# -*- coding: UTF-8 -*-
'''
@author: Andrewzhj
@contact: andrew_zhj@126.com
@file: breast_cancer_train.py
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

ssl._create_default_https_context = ssl._create_unverified_context

# 良/恶性乳腺癌肿瘤数据训练
class CancerTrain(object):

    '''
        初始化数据地址
    '''
    def __init__(self):
        self.bathUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/"
        self.dataFile = "breast-cancer-wisconsin.data"

    '''
        获取数据
    '''
    def getData(self):
        url = self.bathUrl + self.dataFile
        column_names = [
            'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
            'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Blan Chromatin', 'Normal Nucleoli',
            'Mitoses', 'Class'
        ]
        # 读取数据
        data = pd.read_csv(url, names=column_names)
        # 将?转换为标准缺失值
        data = data.replace(to_replace='?', value=np.nan)
        # 丢弃带有缺失值的数据
        data = data.dropna(how='any')
        return data

    '''
        逻辑回归训练
    '''
    def logisticTrain(self):
        data = self.getData()
        # 创建特征列表
        column_names = [
            'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
            'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Blan Chromatin', 'Normal Nucleoli',
            'Mitoses', 'Class'
        ]
        x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]],
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
    '''
    def sgdcTrain(self):
        data = self.getData()
        # 创建特征列表
        column_names = [
            'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
            'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Blan Chromatin', 'Normal Nucleoli',
            'Mitoses', 'Class'
        ]
        x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]],
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

if __name__ == '__main__':
    cancerTrain = CancerTrain()
    cancerTrain.logisticTrain()
    cancerTrain.sgdcTrain()