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
        # 创建特征列表
        column_names = self.getColumnNames()
        # 读取数据
        data = pd.read_csv(url, names=column_names)
        # 将?转换为标准缺失值
        data = data.replace(to_replace='?', value=np.nan)
        # 丢弃带有缺失值的数据
        data = data.dropna(how='any')
        print(data.shape)
        return data

    def getColumnNames(self):
        column_names = [
            'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
            'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Blan Chromatin', 'Normal Nucleoli',
            'Mitoses', 'Class'
        ]
        return column_names

if __name__ == '__main__':
    cancerTrain = CancerTrain()
    cancerTrain.getData()