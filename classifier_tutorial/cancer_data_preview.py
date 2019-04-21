# -*- coding: UTF-8 -*-
'''
@author: Andrewzhj
@contact: andrew_zhj@126.com
@file: cancer_data_preview.py
@time: 4/7/19 3:55 PM
@desc:
@note:
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# 训练数据
df_train = pd.read_csv("../dataset/breast-cancer-train.csv")
# print(df_train)

# 检测数据
df_test = pd.read_csv("../dataset/breast-cancer-test.csv")
# print(df_test)

# Type 等于 0 为负分类样本
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
# print(df_test_negative)

# Type 等于 1 为正分类样本
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]
# print(df_test_positive)

# 绘制肿瘤样本图
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel("Clump Thickness")
plt.ylabel('Cell Size')
# plt.show()

# 使用随机函数随机绘制一条直线 随机截距、系数
intercept = np.random.random([1])
coef = np.random.random([2])
lx = np.arange(0, 12)
ly = (-intercept - lx * coef[0]) / coef[1]
# plt.plot(lx, ly, c='yellow')
# plt.show()

# 使用逻辑回归分类（使用前10条训练样本）
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
print("Testing accuracy (10 training sample): ", (lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])))


# 绘制训练后的分类曲线
intercept = lr.intercept_
coef = lr.coef_[0, : ]
ly = (-intercept - lx * coef[0]) / coef[1]
plt.plot(lx, ly, c='green')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel("Clump Thickness")
plt.ylabel('Cell Size')
# plt.show()


# 使用逻辑回归分类（所有训练样本）
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
print("Testing accuracy (all training sample): ", (lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])))


# 绘制训练后的分类曲线
intercept = lr.intercept_
coef = lr.coef_[0, : ]
ly = (-intercept - lx * coef[0]) / coef[1]
plt.plot(lx, ly, c='blue')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel("Clump Thickness")
plt.ylabel('Cell Size')
plt.show()