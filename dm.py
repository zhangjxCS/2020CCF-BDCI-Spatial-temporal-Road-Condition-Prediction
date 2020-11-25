import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data preprocess
topo = pd.read_csv('topo.txt', sep = '	', header=None, names=['linkid', 'downid'])
attr = pd.read_csv('attr.txt', sep = '	', header=None, names=['linkid', 'length', 'direction', 'pathclass', 'speedclass',
                                                                   'LaneNume', 'speedlimit', 'level', 'width'])
data0701 = pd.read_csv('20190701.txt', sep=';', header=None)
data0701[0].str.split(' ', expand=True)
