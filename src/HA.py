#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:24:50 2020

@author: gglj
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import re
import math
from sklearn.metrics import f1_score
from collections import Counter


#历史与实时路况
traffic = pd.read_csv('traffic/20190711.txt',sep=' |;',names=['linkid','label','current_slice_id','future_slice_id','recent_feature_1','recent_feature_2','recent_feature_3','recent_feature_4','recent_feature_5','history_feature_1_1','history_feature_1_2','history_feature_1_3','history_feature_1_4','history_feature_1_5','history_feature_2_1','history_feature_2_2','history_feature_2_3','history_feature_2_4','history_feature_2_5','history_feature_3_1','history_feature_3_2','history_feature_3_3','history_feature_3_4','history_feature_3_5','history_feature_4_1','history_feature_4_2','history_feature_4_3','history_feature_4_4','history_feature_4_5'])
print(traffic.loc[0])


#分割特征的函数（时间片：路况速度，eta速度，路况状态，车辆数）
def feature_split(x):
    y = re.split(':|,',x)
    return y


def transform(x):
    if x == 4:
        y = 3
    elif x == 0:
        y = 1
    else:
        y = x
    return y


#将history_feature分割，存储在history_label中
history_label = pd.DataFrame()
for i in range(4):
    print(i)
    for j in range(5):
        print(j)
        history_feature_split = pd.DataFrame(map(feature_split,traffic.iloc[:,9+5*i+j]))
        history_label[f'{i+1}_{j+1}'] = history_feature_split[3]
for column_name in history_label.columns:
    history_label[column_name] = history_label[column_name].astype('int')
    history_label[column_name] = history_label[column_name].map(transform)
print(history_label.iloc[26,:])
print(history_label.head())


#实际的label
real_label = copy.deepcopy(traffic.iloc[:,1])
real_label = real_label.astype('int')
real_label = real_label.map(transform)
print(real_label.head())


#预测的label
predict_label = history_label.mode(axis=1).iloc[:,0]


#评价
scores = f1_score(real_label, predict_label, average=None)
scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6
print(scores)


#测试集
traffic = pd.read_csv('20190801_testdata.txt',sep=' |;',names=['linkid','label','current_slice_id','future_slice_id','recent_feature_1','recent_feature_2','recent_feature_3','recent_feature_4','recent_feature_5','history_feature_1_1','history_feature_1_2','history_feature_1_3','history_feature_1_4','history_feature_1_5','history_feature_2_1','history_feature_2_2','history_feature_2_3','history_feature_2_4','history_feature_2_5','history_feature_3_1','history_feature_3_2','history_feature_3_3','history_feature_3_4','history_feature_3_5','history_feature_4_1','history_feature_4_2','history_feature_4_3','history_feature_4_4','history_feature_4_5'])
print(traffic.loc[0])


#分割特征的函数（时间片：路况速度，eta速度，路况状态，车辆数）
def feature_split(x):
    y = re.split(':|,',x)
    return y


def transform(x):
    if x == 4:
        y = 3
    elif x == 0:
        y = 1
    else:
        y = x
    return y


#将history_feature分割，存储在history_label中
history_label = pd.DataFrame()
for i in range(4):
    print(i)
    for j in range(5):
        print(j)
        history_feature_split = pd.DataFrame(map(feature_split,traffic.iloc[:,9+5*i+j]))
        history_label[f'{i+1}_{j+1}'] = history_feature_split[3]
for column_name in history_label.columns:
    history_label[column_name] = history_label[column_name].astype('int')
    history_label[column_name] = history_label[column_name].map(transform)
print(history_label.iloc[26,:])
print(history_label.head())


#预测的label
predict_label = history_label.mode(axis=1).iloc[:,0]


#提交
submit = pd.DataFrame()
submit['link'] = copy.deepcopy(traffic['linkid'])
submit['current_slice_id'] = copy.deepcopy(traffic['current_slice_id'])
submit['future_slice_id'] = copy.deepcopy(traffic['future_slice_id'])
submit['label'] = copy.deepcopy(predict_label)
submit['label'] = submit['label'].astype('int')
submit.to_csv('submit_lu.csv',sep=',',index=None)


print(traffic.iloc[20,:])
list(predict_label).count(0)


    


