import numpy as np
import pandas as pd
import os
import datetime
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


def get_info(x):
    return [i.split(":")[-1] for i in x.split(" ")]
def get_speed(x):
    return np.array([i.split(",")[0] for i in x],dtype='float16')
def get_eta(x):
    return np.array([i.split(",")[1] for i in x],dtype="float16")
def get_state(x):
    return np.array([i.split(",")[2] for i in x])
def get_cnt(x):
    return np.array([i.split(",")[3] for i in x],dtype="int16")


def get_feature(input_file_path_his, input_file_path_attr,input_file_path_topo, mode):
    # his
    df = pd.read_csv(input_file_path_his, sep=";", header=None)
    df["link"] = df[0].apply(lambda x: x.split(" ")[0]).astype(int)
    df["label"] = df[0].apply(lambda x: x.split(" ") [1]).astype(int)
    df["current_slice_id"] = df[0].apply(lambda x: x.split(" ")[2]).astype(int)
    df["future_slice_id"] = df[0].apply(lambda x: x.split(" ")[3]).astype(int)
    df["time_diff"] = df["future_slice_id"] - df["current_slice_id"]
    df = df.drop([0], axis=1)

    if mode == "is_train":
        df["label"] = df["label"].map(lambda x: 3 if x >= 3 else x)
        df['label'] -= 1
    else:
        df = df.drop(["label"], axis=1)

    df["current_state_last"] = df[1].apply(lambda x: x.split(" ")[-1].split(":")[-1])
        # 路况速度,eta速度,路况状态,参与路况计算的车辆数
    df["current_speed"] = df["current_state_last"].apply(lambda x: x.split(",")[0])
    df["current_eat_speed"] = df["current_state_last"].apply(lambda x: x.split(",")[1])
    df["current_state"] = df["current_state_last"].apply(lambda x: x.split(",")[2])
    df["current_count"] = df["current_state_last"].apply(lambda x: x.split(",")[3])
    df = df.drop(["current_state_last"], axis=1)
    for i in tqdm(range(1, 6, 1)):
        flag = f"his_{(6-i)*7}"
        df["history_info"] = df[i].apply(get_info)

        # speed
        df["his_speed"] = df["history_info"].apply(get_speed)
        df[f'{flag}_speed_mean'] = df["his_speed"].apply(lambda x: x.mean())

        # eta
        df["his_eta"] = df["history_info"].apply(get_eta)
        df[f"{flag}_eta_mean"] = df["his_eta"].apply(lambda x: x.mean())


        # state
        df["his_state"] = df["history_info"].apply(get_state)
        df[f"{flag}_state_max"] = df["his_state"].apply(lambda x: Counter(x).most_common()[0][0])
        df[f"{flag}_state_min"] = df["his_state"].apply(lambda x: Counter(x).most_common()[-1][0])

        # cnt
        df["his_cnt"] = df["history_info"].apply(get_cnt)
        df[f"{flag}_cnt_mean"] = df["his_cnt"].apply(lambda x: x.mean())
        df = df.drop([i, "history_info", "his_speed", "his_eta", "his_state", "his_cnt"], axis=1)
        # break

    df2 = pd.read_csv(input_file_path_attr, sep='\t',
                       names=['link', 'length', 'direction', 'path_class', 'speed_class',
                              'LaneNum', 'speed_limit', 'level', 'width'], header=None)
    df = df.merge(df2, on='link', how='left')

    if mode =="is_train":
        output_file_path =f"./data/{mode}_{input_file_path_his.split('/')[-1].split('.')[0]}" +".csv"
        df.to_csv(output_file_path,index =False,mode='w', header=True)

    else:
        output_file_path=f"./data/{input_file_path_his.split('/')[-1].split('.')[0]}" +".csv"
        df.to_csv(output_file_path,index = False,mode='w', header=True)
    # print(df.dtypes)

if __name__ =="__main__":
    print(datetime.datetime.now())
    #训练集
    get_feature(input_file_path_his="20190701.txt",\
                input_file_path_attr="attr.txt",\
                input_file_path_topo="topo.txt",mode="is_train")
    #测试集
    get_feature(input_file_path_his="20190801_testdata.txt",\
                input_file_path_attr="attr.txt",\
                input_file_path_topo="topo.txt",mode="is_test")
    print(datetime.datetime.now())