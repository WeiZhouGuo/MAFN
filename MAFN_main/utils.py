
import numpy as np
import os
import random as rd
from sklearn.model_selection import StratifiedKFold, train_test_split
import xlrd
import torch

def get_split_data():
    file_train = open('../data/data_metabric.txt', 'r', encoding='utf-8').readlines()
    data_origin = []
    for line_data in file_train:
        data_origin.append([float(i) for i in line_data.rstrip().split(' ')])
    data_origin = np.array(data_origin)
    data = data_origin[:, :-1]
    label = data_origin[:, -1]
    skf = StratifiedKFold(n_splits=10)
    i = 1
    train_index = []
    test_index = []
    for train_indx, test_indx in skf.split(data, label):
        if(i%2==0):
            print(i, 'fold #####')
            train_index.append(train_indx)
            test_index.append(test_indx)
        print(test_indx)
        i += 1
    print('\n')
    print(test_index[0])
    np.save('../data/train_index.npy', train_index)
    np.save('../data/test_index.npy', test_index)

def get_label():
    file_train = open('../data/data_metabric.txt', 'r', encoding='utf-8').readlines()
    data_origin = []
    for line_data in file_train:
        data_origin.append([float(i) for i in line_data.rstrip().split(' ')])
    data_origin = np.array(data_origin)
    label = data_origin[:, -1].tolist()
    return label

def get_time_state():
    path = '../data/metabric_time_state.xlsx'
    data = xlrd.open_workbook(path)
    table = data.sheet_by_name('sheet1')
    colum = table.ncols
    row = table.nrows
    state = []
    os_time = []
    for i in range(0,row):
        state.append(table.cell(i,1).value)
        os_time.append(table.cell(i, 0).value)
    return os_time,state

def load_data():
    file_train = open('../data/data_metabric.txt', 'r', encoding='utf-8').readlines()
    data_origin = []
    for line_data in file_train:
        data_origin.append([float(i) for i in line_data.rstrip().split(' ')])
    data_origin = np.array(data_origin)
    data = data_origin[:, :-1].tolist()
    return data

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    torch.cuda.manual_seed_all(5)
    torch.manual_seed(5)
    if classname.find('Linear') != -1:

        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-0.04, 0.04)
        m.bias.data.fill_(0)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.deterministic = True

def regularize_weights(model, reg_type=None):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg

if __name__=='__main__':
    get_split_data()
