import pandas as pd
import lightgbm as lgb
import numpy as np
import copy
import re
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

def feature_split(x):
    y = re.split('[:,]',x)
    return y

def loaddata():
    # Load attributes data
    attr = pd.read_csv('attr.txt', sep='\t',names=['linkid', 'length', 'direction', 'pathclass', 'speedclass', 'LaneNum', 'speedlimit','level', 'width'])
    # Load topo data
    topo = pd.read_csv('topo.txt', sep='\t', names=['key', 'value'])
    inflow = {}
    outflow = {}
    """
    for i in range(len(topo)):
        print(i)
        key = topo.iloc[i, 0]
        value = list(map(int, topo.iloc[i, 1].split(',')))
        outflow[key] = value
        for j in value:
            if inflow.get(j, 0) == 0:
                inflow[j] = [key]
            else:
                inflow[j].append(key)
    """
    return attr, inflow, outflow

def loadtradata(file, attr, inflow, outflow):
    # Load traffic data
    name = ['linkid','label','current_slice_id','future_slice_id','recent_feature_1'
    ,'recent_feature_2','recent_feature_3','recent_feature_4','recent_feature_5','history_feature_1_1','history_feature_1_2'
    ,'history_feature_1_3','history_feature_1_4','history_feature_1_5','history_feature_2_1','history_feature_2_2'
    ,'history_feature_2_3','history_feature_2_4','history_feature_2_5','history_feature_3_1','history_feature_3_2'
    ,'history_feature_3_3','history_feature_3_4','history_feature_3_5','history_feature_4_1','history_feature_4_2'
    ,'history_feature_4_3','history_feature_4_4','history_feature_4_5']
    traffic = pd.read_csv(file, sep=' |;',names=name)
    traffic = traffic.join(attr.set_index('linkid'), on='linkid')
    # 连接道路属性和实时路况数据，linkid为主键
    traffic_feature = traffic.iloc[:, 0:4]
    list = ['_road_velocity', '_eta_velocity', '_road_condition' , '_car_num']
    for i in range(4, 29):
        new_feature = traffic.iloc[:, i].str.split(',', expand=True)
        column = [name[i] + j for j in list]
        traffic_feature[column[0]] = new_feature.iloc[:, 0].str.split(':', expand=True)[1]
        traffic_feature[column[1]] = new_feature[1]
        traffic_feature[column[2]] = new_feature[2]
        traffic_feature[column[3]] = new_feature[3].astype('float') / traffic['LaneNum'] / traffic['length']
        print(i)

    # 将数据集转换成数值形式
    traffic_feature = traffic_feature.apply(pd.to_numeric)
    # 预处理label部分
    label = traffic_feature['label']
    label[label < 1] = 1
    label[label > 3] = 3
    traffic_feature['label'] = label
    data = traffic_feature.join(attr.set_index('linkid'), on='linkid')
    return data

if __name__ == '__main__':
    # Load Data
    attr, inflow, outflow = loaddata()
    data = pd.DataFrame()
    for i in range(24, 31):
        filename = 20190700 + i
        strfile = str(filename) + '.txt'
        dataset = loadtradata(strfile, attr, inflow, outflow)
        df1 = dataset.loc[dataset['label'] == 1]
        df2 = dataset.loc[dataset['label'] == 2]
        df3 = dataset.loc[dataset['label'] == 3]
        num = df1.shape[0]//4
        random_df1 = df1.sample(n=num, random_state=3)
        df = pd.concat([random_df1, df2, df3])

        data = data.append(df)
    print(data.describe())
    print(data.info())
    # Shuffle dataset
    data = data.sample(frac=1, random_state=1)
    X_train = data.iloc[:, 2:]
    Y_train = data.iloc[:, 1]
    test = loadtradata('20190801_testdata.txt', attr, inflow, outflow)
    X_test = test.iloc[:, 2:]
    """
    # Decision Tree
    treeclf = tree.DecisionTreeClassifier()
    treeclf.fit(X_train, Y_train)
    y_crossval = treeclf.predict(X_crossval)
    scores = f1_score(Y_crossval, y_crossval, average=None)
    scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6
    print(scores)
    y_pred = treeclf.predict(X_test)
    """
    # LightGBM
    """
    # 参数调优
    params1 = {
        'max_depth': [20, 40, 80],
        'num_leaves': [320, 640, 1280],
    }
    params2 = {
        'min_child_samples': [40, 80, 160, 320],
        #'min_child_weight':[0],
    }
    params3 = {
        'feature_fraction': [0.6, 0.8, 1.0]
    }
    params4 = {
        'bagging_fraction': [0.6, 0.8, 1],
        'bagging_freq': [0, 1, 2],
    }
    params5 = {
        'reg_alpha': [0, 0.005, 0.1],
        'reg_lambda': [0.2, 0.4, 0.8],
    }
    params6 = {
        'cat_smooth': [0, 10, 20],
    }
    """
    gbm = lgb.LGBMClassifier(
                             objective='multiclass',
                             num_class=3,
                             max_depth=40,
                             num_leaves=1280,
                             learning_rate=0.03,
                             feature_fraction=1.0,
                             min_child_samples=40,
                             min_child_weight=0,
                             bagging_fraction=0.8,
                             bagging_freq=1,
                             reg_alpha=0,
                             reg_lambda=0.2,
                             cat_smooth=0,
                             num_iterations=500,
                             class_weight = {1: 0.2, 2: 0.2, 3: 0.6}
                             )
    """
    gsearch = GridSearchCV(gbm, param_grid=params5, scoring='f1_weighted', cv=3)
    gsearch.fit(X_train, Y_train)
    # Hyper parameter tuning
    print('参数的最佳取值:{0}'.format(gsearch.best_params_))
    print('最佳模型得分:{0}'.format(gsearch.best_score_))
    print(gsearch.cv_results_['mean_test_score'])
    print(gsearch.cv_results_['params'])
    """
    gbm.fit(X_train, Y_train)
    y_pred = gbm.predict(X_test)
    # Output
    out = {'link':test['linkid'], 'current_slice_id':test['current_slice_id'], 'future_slice_id':test['future_slice_id'], 'label':y_pred}
    out = pd.DataFrame(out)
    out.to_csv('result.csv', index=False)