# -*- coding: UTF-8 -*-
'''
@author: Andrewzhj
@contact: andrew_zhj@126.com
@file: neighbors.py
@time: 4/21/19 10:53 PM
@desc: K 近邻分类器
@note:
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def iris_train():
    iris = load_iris()
    # print(iris.data.shape)
    # print(iris.DESCR)
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    knc = KNeighborsClassifier()
    knc.fit(x_train, y_train)
    y_predict = knc.predict(x_test)

    print("The accuracy of K-Nearest Neighbor Classifier is ", knc.score(x_test, y_test))
    print(classification_report(y_test, y_predict, target_names=iris.target_names))


if __name__ == '__main__':
    iris_train()