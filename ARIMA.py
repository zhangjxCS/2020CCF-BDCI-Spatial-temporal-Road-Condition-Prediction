#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:28:44 2020

@author: gglj
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import re
import math
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression as lr
from collections import Counter


#历史与实时路况
traffic = pd.read_csv('traffic/20190730.txt',sep=' |;',names=['linkid','label','current_slice_id','future_slice_id','recent_feature_1','recent_feature_2','recent_feature_3','recent_feature_4','recent_feature_5','history_feature_1_1','history_feature_1_2','history_feature_1_3','history_feature_1_4','history_feature_1_5','history_feature_2_1','history_feature_2_2','history_feature_2_3','history_feature_2_4','history_feature_2_5','history_feature_3_1','history_feature_3_2','history_feature_3_3','history_feature_3_4','history_feature_3_5','history_feature_4_1','history_feature_4_2','history_feature_4_3','history_feature_4_4','history_feature_4_5'])
print(traffic.loc[0])
traffic_valid = pd.read_csv('traffic/20190729.txt',sep=' |;',names=['linkid','label','current_slice_id','future_slice_id','recent_feature_1','recent_feature_2','recent_feature_3','recent_feature_4','recent_feature_5','history_feature_1_1','history_feature_1_2','history_feature_1_3','history_feature_1_4','history_feature_1_5','history_feature_2_1','history_feature_2_2','history_feature_2_3','history_feature_2_4','history_feature_2_5','history_feature_3_1','history_feature_3_2','history_feature_3_3','history_feature_3_4','history_feature_3_5','history_feature_4_1','history_feature_4_2','history_feature_4_3','history_feature_4_4','history_feature_4_5'])
traffic_test = pd.read_csv('20190801_testdata.txt',sep=' |;',names=['linkid','label','current_slice_id','future_slice_id','recent_feature_1','recent_feature_2','recent_feature_3','recent_feature_4','recent_feature_5','history_feature_1_1','history_feature_1_2','history_feature_1_3','history_feature_1_4','history_feature_1_5','history_feature_2_1','history_feature_2_2','history_feature_2_3','history_feature_2_4','history_feature_2_5','history_feature_3_1','history_feature_3_2','history_feature_3_3','history_feature_3_4','history_feature_3_5','history_feature_4_1','history_feature_4_2','history_feature_4_3','history_feature_4_4','history_feature_4_5'])

#分割特征的函数（时间片：路况速度，eta速度，路况状态，车辆数）
def feature_split(x):
    y = re.split(':|,',x)
    return y


'''
#将离散变量1，2，3转换为连续变量
def transform(x):
    if x == 0 or x == 1:
        y = [1,0,0]
    elif x == 2:
        y = [0,1,0]
    else:
        y = [0,0,1]
    return y
'''


def transform(x):
    if x == 0:
        y = 1
    elif x == 4:
        y = 3
    else:
        y = x
    return y


#将history_feature分割，存储在history_label中
history_label = pd.DataFrame()
for i in range(5):
    print(i)
    for j in range(5):
        print(j)
        history_feature_split = pd.DataFrame(map(feature_split,traffic.iloc[:,4+5*i+j]))
        history_label[f'{i+1}_{j+1}'] = history_feature_split[3]
print(history_label.iloc[26,:])
print(history_label.head())

history_label_valid = pd.DataFrame()
for i in range(5):
    print(i)
    for j in range(5):
        print(j)
        history_feature_split_valid = pd.DataFrame(map(feature_split,traffic_valid.iloc[:,4+5*i+j]))
        history_label_valid[f'{i+1}_{j+1}'] = history_feature_split_valid[3]

history_label_test = pd.DataFrame()
for i in range(5):
    print(i)
    for j in range(5):
        print(j)
        history_feature_split_test = pd.DataFrame(map(feature_split,traffic_test.iloc[:,4+5*i+j]))
        history_label_test[f'{i+1}_{j+1}'] = history_feature_split_test[3]


#实际的label
real_label = copy.deepcopy(traffic.iloc[:,1])
real_label = real_label.map(transform)
real_label_valid = copy.deepcopy(traffic_valid.iloc[:,1])
real_label_valid = real_label_valid.map(transform)
'''
real_label = list(real_label)
real_label = np.array(real_label)
'''


#回归
model = lr()
model.fit(history_label, real_label)
print(model.score(history_label, real_label))
'''
y = np.array([[0,0],[1,1]])
x = np.array([[0,0,0],[1,1,1]])
model.fit(x,y)
model.score(x,y)
model.predict(x)
'''

#预测的label
'''
predict_label = model.predict(history_label).argmax(axis=1)+1
list(predict_label).count(3)
'''
def thre(x, thre1=1.4, thre2=1.9):
    if x <= thre1:
        y = 1
    elif x >= thre2:
        y = 3
    else:
        y = 2
    return y
predict_label_continuous = model.predict(history_label)
predict_label = list(map(thre,predict_label_continuous))
predict_label_continuous_valid = model.predict(history_label_valid)
predict_label_valid = list(map(thre,predict_label_continuous_valid))
predict_label_continuous_test = model.predict(history_label_test)
predict_label_test = list(map(thre,predict_label_continuous_test))


#评价
scores = f1_score(real_label, predict_label, average=None)
scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6
print(scores)
scores_valid = f1_score(real_label_valid, predict_label_valid, average=None)
scores_valid = scores_valid[0] * 0.2 + scores_valid[1] * 0.2 + scores_valid[2] * 0.6
print(scores_valid)


#提交
out = {'link':traffic_test['linkid'], 'current_slice_id':traffic_test['current_slice_id'], 'future_slice_id':traffic_test['future_slice_id'], 'label':predict_label_test}
out = pd.DataFrame(out)
out.to_csv('result_regression.csv', index=False)
