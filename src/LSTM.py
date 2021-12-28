#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:28:13 2020

@author: gglj
"""


import pandas as pd
import numpy as np
import copy
import re
from sklearn import svm, tree
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
'''
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm
from paddle.fluid.layers import dynamic_lstm
from paddle.fluid import layers
import paddle


def feature_split(x):
    y = re.split('[:,]',x)
    return y

def loaddata():
    # Load attributes data
    attr = pd.read_csv('attr.txt', sep='\t',names=['linkid', 'length', 'direction', 'pathclass', 'speedclass', 'LaneNum', 'speedlimit','level', 'width'])
    # Load topo data
    topo = pd.read_csv('topo.txt', sep='\t', names=['key', 'value'])
    topo_dict = {}
    '''
    for i in range(len(topo)):
        print(i)
        key = topo.iloc[i, 0]
        value = list(map(int, topo.iloc[i, 1].split(',')))
        topo_dict[key] = value
    '''
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

def data_loader(x_data=None, x_data2=None, y_data=None, batch_size=1024):
    def reader():
        batch_data = []
        batch_data2 = []
        batch_labels = []
        for i in range(x_data.shape[0]):
            
            batch_labels.append(y_data.iloc[i])
            batch_data.append(x_data[i])
            batch_data2.append(x_data2[i])

            if len(batch_data) == batch_size:
                batch_data = np.array(batch_data).astype('float32')
                batch_data2 = np.array(batch_data2).astype('float32')
                batch_labels = np.array(batch_labels).astype('int')
                yield batch_data, batch_data2, batch_labels
                batch_data = []
                batch_data2 = []
                batch_labels = []
        if len(batch_data) > 0:
            batch_data = np.array(batch_data).astype('float32')
            batch_data2 = np.array(batch_data2).astype('float32')
            batch_labels = np.array(batch_labels).astype('int')
            yield batch_data, batch_data2, batch_labels
            batch_data = []
            batch_data2 = []
            batch_labels = []
    return reader

def feature_engineering(X):
    #道路属性特征
    attr_lstm = np.array(pd.concat([X.iloc[:,-8:],X.iloc[:,-8:],X.iloc[:,-8:],X.iloc[:,-8:],X.iloc[:,-8:]],axis=1)).reshape((len(X),5,8))
    #与预测时间的时间差
    time_lstm = X.iloc[:,1] - X.iloc[:,0]
    time_lstm = np.array(pd.concat([time_lstm+4,time_lstm+3,time_lstm+2,time_lstm+1,time_lstm],axis=1)).reshape((len(X),5,1))
    #近期路况特征
    road_lstm = np.array(X.iloc[:,2:22]).reshape((len(X),5,4))
    #合并
    X_lstm = np.concatenate((road_lstm,attr_lstm,time_lstm),axis=2)
    return X_lstm

def feature_engineering2(X):
    #道路属性特征
    attr_lstm = np.array(pd.concat([X.iloc[:,-8:],X.iloc[:,-8:],X.iloc[:,-8:],X.iloc[:,-8:]],axis=1)).reshape((len(X),4,8))
    #与预测时间的时间差
    time_lstm = pd.DataFrame(np.ones(len(X)),columns = ['time_difference'])
    time_lstm = np.array(pd.concat([time_lstm*4,time_lstm*3,time_lstm*2,time_lstm],axis=1)).reshape((len(X),4,1))
    #近期路况特征
    road_lstm = np.array(X.iloc[:,22:102]).reshape((len(X),4,20))
    #合并
    X_lstm = np.concatenate((road_lstm,attr_lstm,time_lstm),axis=2)
    return X_lstm

class base_model(fluid.dygraph.Layer):
    def __init__(self, classes_num: int):
        super().__init__()
        self.hidden_size = 128
        self.batchNorm1d = paddle.nn.BatchNorm1d(5)
        self.lstm   = paddle.nn.LSTM(input_size=13, hidden_size=self.hidden_size, direction="bidirectional")
        
        self.avgpool1d = paddle.nn.AvgPool1d(kernel_size=self.hidden_size*2, stride=self.hidden_size*2)
        self.maxpool1d = paddle.nn.MaxPool1d(kernel_size=self.hidden_size*2, stride=self.hidden_size*2)


    def forward(self, input):
        #input:（batch_size, max_len, dim)
        
        x = self.batchNorm1d(input)
        x.stop_gradient = True

        rnn_out = self.lstm(x)[0]
        mean_out = self.avgpool1d(rnn_out)
        max_out = self.maxpool1d(rnn_out)
        r_shape = (mean_out.shape[0], mean_out.shape[1])
        mean_pool_out = layers.reshape(mean_out, shape=r_shape)
        max_pool_out = layers.reshape(max_out, shape=r_shape)
        add_output = mean_pool_out + max_pool_out
        concat_output = layers.concat((mean_pool_out, max_pool_out), axis=1)

        output = layers.fc(concat_output, size=3)
        return output

class base_model2(fluid.dygraph.Layer):
    def __init__(self, classes_num: int):
        super().__init__()
        self.hidden_size = 128
        self.batchNorm1d = paddle.nn.BatchNorm1d(4)
        self.lstm   = paddle.nn.LSTM(input_size=29, hidden_size=self.hidden_size, direction="bidirectional")
        
        self.avgpool1d = paddle.nn.AvgPool1d(kernel_size=self.hidden_size*2, stride=self.hidden_size*2)
        self.maxpool1d = paddle.nn.MaxPool1d(kernel_size=self.hidden_size*2, stride=self.hidden_size*2)


    def forward(self, input):
        #input:（batch_size, max_len, dim)
        
        x = self.batchNorm1d(input)
        x.stop_gradient = True

        rnn_out = self.lstm(x)[0]
        mean_out = self.avgpool1d(rnn_out)
        max_out = self.maxpool1d(rnn_out)
        r_shape = (mean_out.shape[0], mean_out.shape[1])
        mean_pool_out = layers.reshape(mean_out, shape=r_shape)
        max_pool_out = layers.reshape(max_out, shape=r_shape)
        add_output = mean_pool_out + max_pool_out
        concat_output = layers.concat((mean_pool_out, max_pool_out), axis=1)

        output = layers.fc(concat_output, size=3)
        return output

if __name__ == '__main__':
    # Load Data
    attr, topo_dict = loaddata()
    data = pd.DataFrame()
    for i in range(24, 31):
        print(i)
        filename = 20190700 + i
        strfile = 'traffic/' + str(filename) + '.txt'
        dataset = loadtradata(strfile, attr, topo_dict)
        '''
        df1 = dataset.loc[dataset['label'] == 1]
        df2 = dataset.loc[dataset['label'] == 2]
        df3 = dataset.loc[dataset['label'] == 3]
        num = df1.shape[0]//4
        random_df1 = df1.sample(n=num, random_state=3)
        df = pd.concat([random_df1, df2, df3])
        data = data.append(df)
        '''
        data = data.append(dataset)
    print(data.describe())
    print(data.info())
    # Shuffle dataset
    data = data.sample(frac=1, random_state=1)
    # Split data into train data and test data
    num = data.shape[0]
    
    X_train = data.iloc[:int(num*0.8), 2:]
    Y_train = data.iloc[:int(num*0.8), 1]
    Y_train = Y_train - 1
    X_train_lstm = feature_engineering(X_train)
    X_train_lstm2 = feature_engineering2(X_train)
    train_loader = data_loader(X_train_lstm, X_train_lstm2, Y_train, 1024)
    
    X_crossval = data.iloc[int(num*0.8):, 2:]
    Y_crossval = data.iloc[int(num*0.8):, 1]
    Y_crossval = Y_crossval - 1
    X_crossval_lstm = feature_engineering(X_crossval)
    X_crossval_lstm2 = feature_engineering2(X_crossval)
    valid_loader = data_loader(X_crossval_lstm, X_crossval_lstm2, Y_crossval, 1024)
    
    test = loadtradata('20190801_testdata.txt', attr, topo_dict)
    X_test = test.iloc[:,2:]
    Y_test = test.iloc[:,1]
    X_test_lstm = feature_engineering(X_test)
    X_test_lstm2 = feature_engineering2(X_test)
    test_loader = data_loader(X_test_lstm, X_test_lstm2, Y_test, 1024)
    
    #LSTM
    with fluid.dygraph.guard():

        program = fluid.default_main_program()
        program.random_seed = 2020
        model = base_model(3)
        model2 = base_model2(3)
        print('start training ... {} kind'.format(3))
        model.train()
        model2.train()
        epoch_num = 10
        # 定义优化器
        opt = fluid.optimizer.Adam(learning_rate=0.001, parameter_list=model.parameters())
        opt2 = fluid.optimizer.Adam(learning_rate=0.001, parameter_list=model2.parameters())
        
        best_acc = 0
        valid_acc = 0
        #val_acc = 0
        print('start training ... {} kind'.format(3))
        for epoch in range(10):
            all_loss = 0
            model.train()
            model2.train()
            
            for batch_id,data in enumerate(train_loader()):
                #print(batch_id)
                x_data, x_data2, y_data = data
                x = paddle.to_tensor(x_data)
                x2 = paddle.to_tensor(x_data2)
                label = paddle.to_tensor(y_data)
                label = paddle.fluid.one_hot(label, depth=3)
                # 运行模型前向计算，得到预测值
                logits = (model(x) + model2(x2)) / 2
                # 进行loss计算
                softmax_logits = fluid.layers.softmax(logits)
                loss = fluid.layers.cross_entropy(softmax_logits, label, soft_label=True)
                avg_loss = fluid.layers.mean(loss)
                all_loss += avg_loss.numpy()
                avg_l = all_loss/(batch_id + 1)
                if(batch_id % 100 == 0):
                    print("epoch: {}, batch_id: {}, loss is: {}, valid acc is: {}".format(epoch, batch_id, avg_loss.numpy(), valid_acc))
                avg_loss.backward()
                opt.minimize(avg_loss)
                opt2.minimize(avg_loss)
                model.clear_gradients()
                model2.clear_gradients()
                # break
            model.eval()
            model2.eval()
            accuracies = []
            losses = []
            for batch_id, data in enumerate(valid_loader()):
                x_data, x_data2, y_data = data
                x_data = fluid.dygraph.to_variable(x_data)
                x_data2 = fluid.dygraph.to_variable(x_data2)
                label = fluid.dygraph.to_variable(y_data)
                # 运行模型前向计算，得到预测值
                logits = (model(x_data) + model2(x_data2)) / 2
                # 计算sigmoid后的预测概率，进行loss计算
                pred = fluid.layers.softmax(logits)
    
                scores = f1_score(y_true=y_data, y_pred=pred.numpy().argmax(axis=1), average=None)
    
                scores = scores[0]*0.2 + scores[1]*0.2 + scores[2]*0.6
                accuracies.append(scores)
            valid_acc = np.mean(accuracies)
        pred_array = np.array([])
        for batch_id, data in enumerate(test_loader()):
            x_data, x_data2, y_data = data
            x_data = fluid.dygraph.to_variable(x_data)
            x_data2 = fluid.dygraph.to_variable(x_data2)
            label = fluid.dygraph.to_variable(y_data)
            # 运行模型前向计算，得到预测值
            logits = (model(x_data) + model2(x_data2)) / 2
            # 
            pred_array = np.append(pred_array,fluid.layers.softmax(logits).numpy().argmax(axis=1))
        pred_array = pred_array + 1
        '''
        scores = f1_score(Y_test, pred_array, average=None)
        scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6
        #fluid.io.save_params(fluid.Executor(fluid.CPUPlace()),'model')
        '''
        out = {'link':test['linkid'], 'current_slice_id':test['current_slice_id'], 'future_slice_id':test['future_slice_id'], 'label':pred_array.astype('int')}
        out = pd.DataFrame(out)
        out.to_csv('result_lstm2_last7day.csv', index=False)
        
        
            
            
    
'''
    model = Sequential()
    #model.add(tf.keras.layers.SimpleRNN(1))
    model.add(tf.keras.layers.LSTM(1))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X_train_lstm, Y_train, epochs=100, batch_size=1024, verbose=2)
    model.summary()
'''
    
    
'''
    # Decision Tree
    treeclf = tree.DecisionTreeClassifier()
    treeclf.fit(X_train, Y_train)
    y_crossval = treeclf.predict(X_crossval)
    scores = f1_score(Y_crossval, y_crossval, average=None)
    scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6
    print(scores)
    y_pred = treeclf.predict(X_test)
    
    # LightGBM
    Y_train = Y_train - 1
    Y_crossval = Y_crossval - 1
    train_data = lgb.Dataset(X_train, label=Y_train)
    validation_data = lgb.Dataset(X_crossval, label=Y_crossval)
    params = {'learning_rate':0.1, 'lambda_l2':0.2, 'max_depth':8, 'objective':'multiclass', 'num_class':3}
    gbm = lgb.train(params, train_data, valid_sets=[validation_data])
    y_pred = np.argmax(gbm.predict(X_test), axis=1) + 1
'''

   
