
import numpy as np
import os
import random as rd
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import torch
from keras.utils.np_utils import to_categorical
class data_helper:
    def __init__(self):
        file_train = open('./data/train.txt', 'r', encoding='utf-8').readlines()
        file_test = open('./data/test.txt', 'r', encoding='utf-8').readlines()
        file_val = open('./data/valid.txt', 'r', encoding='utf-8').readlines()
        data_train=[]
        data_test=[]
        data_val=[]
        for line_data in file_train:
            data_train.append([float(i) for i in line_data.rstrip().split(' ')])
        self.train=np.array(data_train)

        for line_data in file_test:
            data_test.append([float(i) for i in line_data.rstrip().split(' ')])
        self.test=np.array(data_test)

        for line_data in file_val:
            data_val.append([float(i) for i in line_data.rstrip().split(' ')])
        self.valid=np.array(data_val)


    def get_train(self):
        data = self.train
        train_X = torch.tensor(data[:, :-1], dtype=torch.float32)
        train_Y_onehot = to_categorical(data[:, -1])
        train_Y = torch.tensor(train_Y_onehot, dtype=torch.float32)

        torch_dataset = Data.TensorDataset(train_X, train_Y)
        train_loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        return train_loader

    def get_val_data(self):
        data = self.valid
        valid_X = torch.tensor(data[:, :-1], dtype=torch.float32)
        valid_Y_onehot = to_categorical(data[:, -1])
        valid_Y = torch.tensor(valid_Y_onehot, dtype=torch.float32)

        torch_dataset = Data.TensorDataset(valid_X, valid_Y)
        val_loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        return val_loader

    def get_test_data(self):
        data = self.test
        test_X = torch.tensor(data[:, :-1], dtype=torch.float32)
        test_Y_onehot = to_categorical(data[:, -1])
        test_Y = torch.tensor(test_Y_onehot, dtype=torch.float32)
        torch_dataset = Data.TensorDataset(test_X, test_Y)
        test_loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        return test_loader
if __name__=='__main__':
    d=data_helper()
    print(d.get_test_data())
    print()