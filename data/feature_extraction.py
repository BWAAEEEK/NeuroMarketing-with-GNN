import os.path
import pickle
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# label 데이터가 전부 .LAB 확장자로 되어 있어서 .txt로 바꾸는 함수. 한번만 실행시키면 됨.
def LAB_to_txt(path):
    # :param path: 데이터셋 path. (본인 디렉토리 path)+/Data-EEG-25-users-Neuromarketing (str)
    files = glob.glob(path+'/labels/*.LAB')
    for name in files:
        if not os.path.isdir(name):
            src = os.path.splitext(name)
            os.rename(name, src[0]+'.txt')


class Dataset(object):
    def __init__(self, path, split=None, window_size=1, scaler='standard', batch_size=8, random_state=42):
        '''
        Input:
        :param path: 데이터셋 path. (본인 디렉토리 path)+/Data-EEG-25-users-Neuromarketing (str)
        :param split: train valid test 분리용 파라미터. Ex. [0.8, 0.1, 0.1] (list)
        :param window_size: 윈도우를 몇초로 할지 1, 2, 4초 실험해보셈  (int)
        :param scaler: StandardScaler - 'standard', MinMaxScaler - 'minmax', 안할거면 None
        :param batch_size: batch_size (int)
        :param random_state: seed (int)
        '''

        self.feature_path = glob.glob(path+'/25-users/*')
        self.label_path = glob.glob(path+'/labels/*')
        self.window = window_size
        self.channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        self.sampling_rate = 128
        self.scaler = StandardScaler() if (scaler is not None) and (scaler == 'standard') else MinMaxScaler() if (scaler is not None) and (scaler == 'minmax') else None
        self.random_state = random_state
        self.split = split if split is not None else [0.8, 0.1, 0.1]
        self.batch_size = batch_size

    def read_data(self, feature_path, label_path):
        df = pd.read_csv(feature_path, sep=' ', header=None)
        op = open(label_path, 'r')
        label = op.read()
        op.close()
        return df, str(label)

    def data_to_df(self):
        data = None
        for i in tqdm(range(len(self.feature_path))):
            r_data, label = self.read_data(self.feature_path[i], self.label_path[i])
            r_data['label'] = 1 if str(label) == 'Like' else 0
            if i >= 1:
                data = pd.concat([data, r_data], axis=0)
            else:
                data = r_data
        y = data[['label']]
        X = data.drop('label', axis=1)
        return X, y

    def create_window(self, X_df, y_df):
        window_size = self.window * self.sampling_rate
        xs = []
        ys = []
        for i in tqdm(range(X_df.shape[0] // window_size)):
            x = X_df.iloc[(i * window_size):(i * window_size + window_size)].values
            y = y_df.iloc[i * window_size + window_size - 1].values
            xs.append(x.T)
            ys.append(y.T)
        return np.array(xs), np.array(ys)

    def make_Tensor(self, array):
        return torch.from_numpy(array).float()

    def return_data(self):
        X, y = self.data_to_df()
        if self.scaler:
            self.scaler.fit(X)
            X = pd.DataFrame(self.scaler.transform(X), columns=self.channels)
        X_win, y_win = self.create_window(X, y)
        # print(X_win.shape, y_win.shape)

        like_idx = np.where(y_win == 1)
        dislike_idx = np.where(y_win == 0)
        like_X = X_win[like_idx[0], :, :]
        like_y = y_win[like_idx[0], :]
        dislike_X = X_win[dislike_idx[0], :, :]
        dislike_y = y_win[dislike_idx[0], :]
        like_shuffled = np.random.permutation(like_X.shape[0])
        dislike_shuffled = np.random.permutation(dislike_X.shape[0])

        train_like = round(like_X.shape[0]*self.split[0])
        train_dislike = round(dislike_X.shape[0]*self.split[0])
        val_like = round(like_X.shape[0] * (self.split[0]+self.split[1]))
        val_dislike = round(dislike_X.shape[0] * (self.split[0]+self.split[1]))
        test_like = round(like_X.shape[0] * (self.split[0]+self.split[1]+self.split[2]))
        test_dislike = round(dislike_X.shape[0] * (self.split[0]+self.split[1]+self.split[2]))

        X_train = np.concatenate([like_X[like_shuffled[:train_like]], dislike_X[dislike_shuffled[:train_dislike]]], 0)
        y_train = np.concatenate([like_y[like_shuffled[:train_like]], dislike_y[dislike_shuffled[:train_dislike]]], 0)
        X_val = np.concatenate([like_X[like_shuffled[train_like:val_like]], dislike_X[dislike_shuffled[train_dislike:val_dislike]]], 0)
        y_val = np.concatenate([like_y[like_shuffled[train_like:val_like]], dislike_y[dislike_shuffled[train_dislike:val_dislike]]], 0)
        X_test = np.concatenate([like_X[like_shuffled[val_like:test_like]], dislike_X[dislike_shuffled[val_dislike:test_dislike]]], 0)
        y_test = np.concatenate([like_y[like_shuffled[val_like:test_like]], dislike_y[dislike_shuffled[val_dislike:test_dislike]]], 0)

        # print(X_train.shape, y_train.shape)
        # print(X_val.shape, y_val.shape)
        # print(X_test.shape, y_test.shape)

        if True in np.isnan(X_train):
            print('nan in X_train_win')
            quit()
        if True in np.isnan(X_val):
            print('nan in X_val_win')
            quit()
        if True in np.isnan(X_test):
            print('nan in X_test_win')
            quit()

        train_tensor = {"input" : self.make_Tensor(X_train), "label": self.make_Tensor(y_train)}
        val_tensor = {"input": self.make_Tensor(X_val), "label": self.make_Tensor(y_val)}
        test_tensor = {"input": self.make_Tensor(X_test), "label":  self.make_Tensor(y_test)}

        print("Data Preprocessing is Done.")
        return {'train': train_tensor, 'valid': val_tensor, 'test': test_tensor}


if __name__=="__main__":
    path = './Data-EEG-25-users-Neuromarketing'
    a = Dataset(path=path, window_size=4).return_data()

    with open("feature.pkl", "wb") as f:
        pickle.dump(a, f)