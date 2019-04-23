# -*- coding: UTF-8 -*-
'''
@author: Andrewzhj
@contact: andrew_zhj@126.com
@file: decision_tree.py
@time: 4/22/19 9:49 PM
@desc: 决策树
@note: 不要线性假设
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def titanic_train():
    titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    print(titanic.head(10))
    print(titanic.info())
    # 特征选择
    x = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']
    # 填充空缺信息
    x['age'].fillna(x['age'].mean(), inplace=True)
    x.info()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

    # 特征提取
    vec = DictVectorizer(sparse=False)
    x_train = vec.fit_transform(x_train.to_dict(orient='record'))
    print(vec.feature_names_)
    x_test = vec.transform(x_test.to_dict(orient='record'))
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    y_predict = dtc.predict(x_test)
    print(dtc.score(x_test, y_test))
    print(classification_report(y_predict, y_test, target_names=['died', 'survived']))


if __name__ == '__main__':
    titanic_train()