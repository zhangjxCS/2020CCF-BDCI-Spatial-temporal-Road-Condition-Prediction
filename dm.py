import pandas as pd
import lightgbm as lgb
import numpy as np
import copy
import re
from sklearn import svm, tree
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

def feature_split(x):
    y = re.split('[:,]',x)
    return y

def loaddata():
    # Load attributes data
    attr = pd.read_csv('attr.txt', sep='\t',
                       names=['linkid', 'length', 'direction', 'pathclass', 'speedclass', 'LaneNum', 'speedlimit',
                              'level', 'width'])
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
    traffic = pd.read_csv(file, sep=' |;',names=['linkid','label','current_slice_id','future_slice_id','recent_feature_1'
    ,'recent_feature_2','recent_feature_3','recent_feature_4','recent_feature_5','history_feature_1_1','history_feature_1_2'
    ,'history_feature_1_3','history_feature_1_4','history_feature_1_5','history_feature_2_1','history_feature_2_2'
    ,'history_feature_2_3','history_feature_2_4','history_feature_2_5','history_feature_3_1','history_feature_3_2'
    ,'history_feature_3_3','history_feature_3_4','history_feature_3_5','history_feature_4_1','history_feature_4_2'
    ,'history_feature_4_3','history_feature_4_4','history_feature_4_5'])

    # 将recent_feature分割，存储在traffic_feature中
    traffic_feature = copy.deepcopy(traffic.iloc[:,0:4])
    for i in range(5):
        print(i)
        recent_feature_split = pd.DataFrame(map(feature_split,traffic.iloc[:,4+i]))
        traffic_feature[f'recent_feature_{i+1}_road_velocity'] = recent_feature_split[1]
        traffic_feature[f'recent_feature_{i+1}_eta_velocity'] = recent_feature_split[2]
        traffic_feature[f'recent_feature_{i+1}_road_condition'] = recent_feature_split[3]
        traffic_feature[f'recent_feature_{i+1}_car_num'] = recent_feature_split[4]
    # 将数据集转换成数值形式
    traffic_feature = traffic_feature.apply(pd.to_numeric)
    # 预处理label部分
    label = traffic_feature['label']
    label[label < 1] = 1
    label[label > 3] = 3
    traffic_feature['label'] = label
    for i in range(5):
        condition = traffic_feature[f'recent_feature_{i + 1}_road_condition']
        condition[condition < 1] = 1
        condition[condition > 3] = 3
        traffic_feature[f'recent_feature_{i + 1}_road_condition'] = condition
    traffic_feature.head()
    #连接道路属性和实时路况数据，linkid为主键
    data = traffic_feature.join(attr.set_index('linkid'), on='linkid')
    return data

if __name__ == '__main__':
    # Load Data
    attr, topo_dict = loaddata()
    data = pd.DataFrame()
    for i in range(1, 31):
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
    X_train = data.iloc[:int(num*0.8), 2:]
    Y_train = data.iloc[:int(num*0.8), 1]
    X_crossval = data.iloc[int(num*0.8):, 2:]
    Y_crossval = data.iloc[int(num*0.8):, 1]
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
    Y_train = Y_train - 1
    Y_crossval = Y_crossval - 1
    train_data = lgb.Dataset(X_train, label=Y_train)
    validation_data = lgb.Dataset(X_crossval, label=Y_crossval)
    params = {'learning_rate':0.1, 'lambda_l2':0.2, 'max_depth':8, 'objective':'multiclass', 'num_class':3}
    gbm = lgb.train(params, train_data, valid_sets=[validation_data])
    y_pred = np.argmax(gbm.predict(X_test), axis=1) + 1

    out = {'link':test['linkid'], 'current_slice_id':test['current_slice_id'], 'future_slice_id':test['future_slice_id'], 'label':y_pred}
    out = pd.DataFrame(out)
    out.to_csv('result.csv', index=False)