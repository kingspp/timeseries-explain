from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
import numpy as np


class CricketXDataset(Dataset):
    def __init__(self):
        super().__init__(
            name='cricket_x',
            meta_file_path=os.path.join(NTE_MODULE_PATH, 'data', 'real', 'cricketx', 'meta.json'))

    def load_train_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real','cricketx', 'train.csv'))
        df = df.drop(['id'], axis=1)
        train_label = df[df.columns[-1]]
        train_label = pd.Series(
            [0 if train_label[x] == 6 else 1 if train_label[x] == 12 else 2 for x in range(len(train_label))])
        idx = np.where(train_label < 2)[0]
        train_label = train_label.values[idx]
        train_data = df[df.columns[:-1]].to_numpy()[idx]
        print("columns are ")
        print(df.columns)
        ss = df.iloc[idx,:]
        ss.to_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real','cricketx', 'subset_train_data.csv'))
        return train_data, train_label

    def load_test_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'cricketx', 'test.csv'))
        df = df.drop(['id'], axis=1)
        test_label = df[df.columns[-1]]
        test_label = pd.Series(
            [0 if test_label[x] == 6 else 1 if test_label[x] == 12 else 2 for x in range(len(test_label))])
        idx = np.where(test_label != 2)[0]
        test_label = test_label.values[idx]
        test_data = df[df.columns[:-1]].to_numpy()[idx]
        ss = df.iloc[idx,:]
        ss.to_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real','cricketx', 'subset_test_data.csv'))
        return test_data, test_label


