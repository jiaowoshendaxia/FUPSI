#coding=utf-8
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
import math
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

#STResnet+UrbanSG  sr4 train4 data_process4 test4
def get_dataloader(datapath, len_closeness,len_period,len_trend,scaler_X, scaler_Y, batch_size, ext_flag= True,mode='train',map_H = 32,map_W = 32,day_len = 48,channel =1):
    datapath = os.path.join(datapath, mode)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    X = torch.FloatTensor(np.load(os.path.join(datapath, 'X.npy'))) / scaler_X
    print(X.shape)
    # 这里 X.npy是1530个32*32的粗粒度图像，Y.npy是1530个128*128的细粒度图像
    Y = torch.FloatTensor(np.expand_dims(np.load(os.path.join(datapath, 'Y.npy')),1))/ scaler_Y

    print(Y.shape)

    len_list = np.array([len_closeness, len_period * day_len, len_trend * day_len * 7])
    max_len = np.max(len_list)
    print(max_len)

    block_len = max_len+1
    block_num = len(X) - block_len + 1  # block_len = input_window + output_window 101
    # total of [N - block_len + 1] blocks
    # where block_len = input_window + output_window
    train_xc = np.zeros((block_num,len_closeness,channel,map_H,map_W))
    train_xp = np.zeros((block_num,len_period,channel,map_H,map_W))
    train_xt = np.zeros((block_num,len_trend,channel,map_H,map_W))
    train_label = []
    train_pre = []
    if ext_flag == True:
        ext = torch.FloatTensor(np.load(os.path.join(datapath, 'ext.npy')))
        train_ext = []


    for i in range(block_num):

        for j in range(len_closeness):
            train_xc[i, j] = X[i + block_len - (len_closeness - j) - 1]

        for j in range(len_period):
            train_xp[i,j]=X[i + block_len - (len_period-j)*day_len - 1]
        for j in range(len_trend):
            train_xt[i,j]=X[i + block_len - (len_period-j)*day_len*7 - 1]
        train_pre.append(X[i + block_len-1])
        train_label.append(Y[i + block_len-1])
        if ext_flag == True:
            train_ext.append(ext[i + block_len-1])# 1-100 1为需预测长度


    np.save(os.path.join(datapath,'XC_{}.npy'.format(len_closeness)), train_xc)
    np.save(os.path.join(datapath, 'XP_{}.npy'.format(len_period)),train_xp)
    np.save(os.path.join(datapath, 'XT_{}.npy'.format(len_trend)),train_xt)
    np.save(os.path.join(datapath, 'X_next.npy'), np.array([item.numpy() for item in train_pre]))
    np.save(os.path.join(datapath, 'Y_lable.npy'), np.array([item.numpy() for item in train_label]))

    xc = torch.FloatTensor(np.load(os.path.join(datapath, 'XC_{}.npy'.format(len_closeness))))
    xp = torch.FloatTensor(np.load(os.path.join(datapath, 'XP_{}.npy'.format(len_period))))
    xt = torch.FloatTensor(np.load(os.path.join(datapath, 'XT_{}.npy'.format(len_trend))))
    x_pre = torch.FloatTensor(np.load(os.path.join(datapath, 'X_next.npy')))
    label = torch.FloatTensor(np.load(os.path.join(datapath, 'Y_lable.npy')))
    if ext_flag == True:
        np.save(os.path.join(datapath, 'ext_lable.npy'), np.array([item.numpy() for item in train_ext]))
        ext = torch.FloatTensor(np.load(os.path.join(datapath, 'ext_lable.npy')))
    else:
        ext = torch.zeros([block_num])

    # label = label.unsqueeze(2)
    # label = label.unsqueeze(3)
    print(label.shape)

    data = torch.utils.data.TensorDataset(xc,xp,xt,ext,x_pre,label)


    assert len(label) == len(xc)
    print('# {} samples: {}'.format(mode, len(data)))

    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader

def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))