# -*- coding: UTF-8 -*-
'''
@author: Andrewzhj
@contact: andrew_zhj@126.com
@file: bayes.py
@time: 4/21/19 9:38 PM
@desc:
@note:
'''

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def text_train():
    news = fetch_20newsgroups(subset='all')
    print(len(news.data))
    # print(news.data[0])
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

    # 文本特征提取，只考虑词汇在文本中出现的频率
    # vec = CountVectorizer()
    # x_train = vec.fit_transform(x_train)
    # x_test = vec.transform(x_test)

    # 去除停用词
    # count_stop_vec = CountVectorizer(analyzer='word', stop_words='english')
    # x_train = count_stop_vec.fit_transform(x_train)
    # x_test = count_stop_vec.transform(x_test)

    # 文本特征提取，除了考虑词汇出现的频率，还考虑词汇在所有文本的数量，能消减高频没有意义的词汇带来的影响，挖掘更有意义的特征  
    # tfid_vec = TfidfVectorizer()
    # x_train = tfid_vec.fit_transform(x_train)
    # x_test = tfid_vec.transform(x_test)

    # 去除停用词
    tfid_stop_vec = TfidfVectorizer(analyzer='word', stop_words='english')
    x_train = tfid_stop_vec.fit_transform(x_train)
    x_test = tfid_stop_vec.transform(x_test)

    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)
    y_predict = mnb.predict(x_test)
    print("The accuracy of Native Bayes Classifier is ", mnb.score(x_test, y_test))
    print(classification_report(y_test, y_predict, target_names=news.target_names))


if __name__ == '__main__':
    text_train()
