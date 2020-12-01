import pandas as pd
import lightgbm as lgb
import numpy as np
import copy
import re
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

def feature_split(x):
    y = re.split('[:,]',x)
    return y

def loaddata():
    # Load attributes data
    attr = pd.read_csv('attr.txt', sep='\t',names=['linkid', 'length', 'direction', 'pathclass', 'speedclass', 'LaneNum', 'speedlimit','level', 'width'])
    # Load topo data
    topo = pd.read_csv('topo.txt', sep='\t', names=['key', 'value'])
    topo_dict = {}
    for i in range(len(topo)):
        print(i)
        key = topo.iloc[i, 0]
        value = list(map(int, topo.iloc[i, 1].split(',')))
        topo_dict[key] = value
    return attr, topo_dict

def loadtradata(file, attr, topo_dict):
    # Load traffic data
    name = ['linkid','label','current_slice_id','future_slice_id','recent_feature_1'
    ,'recent_feature_2','recent_feature_3','recent_feature_4','recent_feature_5','history_feature_1_1','history_feature_1_2'
    ,'history_feature_1_3','history_feature_1_4','history_feature_1_5','history_feature_2_1','history_feature_2_2'
    ,'history_feature_2_3','history_feature_2_4','history_feature_2_5','history_feature_3_1','history_feature_3_2'
    ,'history_feature_3_3','history_feature_3_4','history_feature_3_5','history_feature_4_1','history_feature_4_2'
    ,'history_feature_4_3','history_feature_4_4','history_feature_4_5']
    traffic = pd.read_csv(file, sep=' |;',names=name)
    traffic_feature = traffic.iloc[:, 0:4]
    list = ['_road_velocity', '_eta_velocity', '_road_condition', '_car_num']
    for i in range(4, traffic.shape[1]):
        new_feature = traffic.iloc[:, i].str.split(',', expand=True)
        column = [name[i] + j for j in list]
        traffic_feature[column[0]] = new_feature.iloc[:, 0].str.split(':', expand=True)[1]
        traffic_feature[column[1]] = new_feature[1]
        traffic_feature[column[2]] = new_feature[2]
        traffic_feature[column[3]] = new_feature[3]
        print(i)

    # 将数据集转换成数值形式
    traffic_feature = traffic_feature.apply(pd.to_numeric)
    # 预处理label部分
    label = traffic_feature['label']
    label[label < 1] = 1
    label[label > 3] = 3
    traffic_feature['label'] = label
    #连接道路属性和实时路况数据，linkid为主键
    data = traffic_feature.join(attr.set_index('linkid'), on='linkid')
    return data

if __name__ == '__main__':
    # Load Data
    attr, topo_dict = loaddata()
    data = pd.DataFrame()
    for i in range(1, 2):
        filename = 20190700 + i
        strfile = str(filename) + '.txt'
        dataset = loadtradata(strfile, attr, topo_dict)
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
    # Split data into train data and test data
    num = data.shape[0]
    X_train = data.iloc[:, 2:]
    Y_train = data.iloc[:, 1]

    test = loadtradata('20190801_testdata.txt', attr, topo_dict)
    X_test = test.iloc[:,2:]
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
    params1 = {
        'max_depth': [4, 8, 16],
        'num_leaves': [20, 40, 80],
    }
    params2 = {
        'min_child_samples': [10, 15, 20, 25, 30],
        'min_child_weight':[0, 0.0005, 0.001],
    }
    params3 = {
        'feature_fraction': [0.6, 0.8, 1.0]
    }
    params4 = {
        'bagging_fraction': [0.6, 0.8, 1],
        'bagging_freq': [1, 2, 4],
    }
    params5 = {
        'reg_alpha': [0.1, 0.2, 0.4],
        'reg_lambda': [0.1, 0.2, 0.4],
    }
    params6 = {
        'cat_smooth': [0, 10, 20],
    }
    gbm = lgb.LGBMClassifier(
                             objective='multiclass',
                             num_class=3,
                             is_unbalance=True,
                             max_depth=16,
                             num_leaves=80,
                             learning_rate=0.1,
                             feature_fraction=1.0,
                             min_child_samples=21,
                             min_child_weight=0.001,
                             bagging_fraction=1,
                             bagging_freq=2,
                             reg_alpha=0.001,
                             reg_lambda=8,
                             cat_smooth=0,
                             num_iterations=200,
                             )
    gsearch = GridSearchCV(gbm, param_grid=params1, scoring='f1_macro', cv=3)
    gsearch.fit(X_train, Y_train)
    # Hyper parameter tuning
    print('参数的最佳取值:{0}'.format(gsearch.best_params_))
    print('最佳模型得分:{0}'.format(gsearch.best_score_))
    print(gsearch.cv_results_['mean_test_score'])
    print(gsearch.cv_results_['params'])
    y_pred = gbm.predict(X_test)
    """
    # Output
    out = {'link':test['linkid'], 'current_slice_id':test['current_slice_id'], 'future_slice_id':test['future_slice_id'], 'label':y_pred}
    out = pd.DataFrame(out)
    out.to_csv('result.csv', index=False)
    """