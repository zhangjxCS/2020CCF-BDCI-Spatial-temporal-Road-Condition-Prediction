import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import re
import math


#道路属性
attr = pd.read_csv('attr.txt',sep='\t',names=['linkid','length','direction','pathclass','speedclass','LaneNum','speedlimit','level','width'])

'''
#拓扑
topo = pd.read_csv('topo.txt',sep='\t',names=['key','value'])
topo_dict = {}
for i in range(len(topo)):
    print(i)
    key = topo.iloc[i,0]
    value = list(map(int,topo.iloc[i,1].split(',')))
    topo_dict[key] = value
 '''

#历史与实时路况
traffic = pd.read_csv('20190701.txt',sep=' |;',names=['linkid','label','current_slice_id','future_slice_id','recent_feature_1'
    ,'recent_feature_2','recent_feature_3','recent_feature_4','recent_feature_5','history_feature_1_1','history_feature_1_2'
    ,'history_feature_1_3','history_feature_1_4','history_feature_1_5','history_feature_2_1','history_feature_2_2'
    ,'history_feature_2_3','history_feature_2_4','history_feature_2_5','history_feature_3_1','history_feature_3_2'
    ,'history_feature_3_3','history_feature_3_4','history_feature_3_5','history_feature_4_1','history_feature_4_2'
    ,'history_feature_4_3','history_feature_4_4','history_feature_4_5'])
print(traffic.loc[0])

'''
#描述性统计
attr_describe = attr.describe()
traffic_describe = traffic.describe()
'''

#分割特征的函数（时间片：路况速度，eta速度，路况状态，车辆数）
def feature_split(x):
    y = re.split(':|,',x)
    return y


#将recent_feature分割，存储在traffic_feature中
traffic_feature = copy.deepcopy(traffic.iloc[:,0:4])
for i in range(5):
    print(i)
    recent_feature_split = pd.DataFrame(map(feature_split,traffic.iloc[:,4+i]))
    traffic_feature[f'recent_feature_{i+1}_road_velocity'] = recent_feature_split[1]
    traffic_feature[f'recent_feature_{i+1}_eta_velocity'] = recent_feature_split[2]
    traffic_feature[f'recent_feature_{i+1}_road_condition'] = recent_feature_split[3]
    traffic_feature[f'recent_feature_{i+1}_car_num'] = recent_feature_split[4]
print(traffic_feature.iloc[0,:])


#连接道路属性和实时路况数据，linkid为主键
data = attr.join(traffic_feature.set_index('linkid'),on='linkid',how="right")
print(data.iloc[22051,:])