# -*- coding: UTF-8 -*-
'''
@author: Andrewzhj
@contact: andrew_zhj@126.com
@file: support_vector.py
@time: 4/21/19 11:58 AM
@desc:
@note:
'''

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


def digits_train():
    # 手写数据加载器
    digits = load_digits()
    # 数据分割，75%作为训练数据，25%作为测试数据
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
    print(y_train.shape)
    print(y_test.shape)

    ss = StandardScaler()
    # 对数据特征标准化
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    # 使用支持向量机分类器训练
    lsvc = LinearSVC()
    lsvc.fit(x_train, y_train)
    # 预测
    y_predict = lsvc.predict(x_test)
    print("The Accuracy of Liner SVC is", lsvc.score(x_test, y_test))
    print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))


if __name__ == '__main__':
    digits_train()