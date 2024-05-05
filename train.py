import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle, random, sys  # , h5py
import mltools, rmldataset2016
from rmlmodels.LSTMModel import LSTMNet, CNNLSTM_Net, BiLSTMNet
# import rmlmodels.BidLSTMModel as bidlstm
import csv
from datetime import datetime  # 用于计算时间
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, precision_score, \
    recall_score, f1_score, confusion_matrix, accuracy_score
import os
from rmlmodels.encoder import Net


def log(fname, s):
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()


def get_upper_SNR(lbl, snrs, data_idx, X_data, Y_data):
    X_list = []
    Y_list = []
    for snr in snrs:
        if snr >= -6:
            data_SNRs = [lbl[x][1] for x in data_idx]
            X_list.append(X_data[np.where(np.array(data_SNRs) == snr)])
            Y_list.append(Y_data[np.where(np.array(data_SNRs) == snr)])
    return (np.concatenate(X_list), np.concatenate(Y_list))


# Set up some params
EPOCH = 100  # number of epochs to train on
BATCH_SIZE = 2048  # training batch size
process_num = 16  # linux可改为8
LEARNING_RATE = 1e-6
L2_DECAY = 1e-5
max_acc = 0.5  # 预训练模型的acc
weight_path = "/root/LSTM2torch/weights/"
weight_file_name = 'net_params-0.8856474358974359.pth'
# print(batch_size)

# 设置GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device={device}")
# 设置随机种子
torch.manual_seed(3407)

(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (
train_idx, val_idx, test_idx) = rmldataset2016.load_data()

# 获取比-6dB大的数据集
X_train, Y_train = get_upper_SNR(lbl, snrs, train_idx, X_train, Y_train)
X_val, Y_val = get_upper_SNR(lbl, snrs, val_idx, X_val, Y_val)

train_loader = DataLoader(list(zip(X_train, Y_train)), batch_size=BATCH_SIZE, num_workers=process_num, pin_memory=True,
                          shuffle=True)
valid_loader = DataLoader(list(zip(X_val, Y_val)), batch_size=BATCH_SIZE, num_workers=process_num, pin_memory=True,
                          shuffle=True)
# test_loader = DataLoader(data_test, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=False)


# model = LSTMNet().to(device)
# model = CNNLSTM_Net().to(device)
# model = Net().to(device)
model = BiLSTMNet().to(device)

if os.path.exists(weight_path + weight_file_name):
    try:
        model.load_state_dict(torch.load(weight_path + weight_file_name))
        print("加载模型" + weight_path + weight_file_name)
    except:
        x = torch.load(weight_path + weight_file_name)
        del x['hidden_out.bias']
        del x['hidden_out.weight']
        model.load_state_dict(x, strict=False)
        print("去掉最后一层并加载模型" + weight_path + weight_file_name)

else:
    print("未加载模型")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # 分类问题
# mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[EPOCH // 8, EPOCH // 4,EPOCH // 2, EPOCH // 4 * 3], gamma=0.5)

# 训练+验证
train_loss = []
valid_loss = []
min_valid_loss = np.inf
for i in range(EPOCH):
    total_train_loss = []
    model.train()  # 进入训练模式
    for step, (batch_data, batch_label) in enumerate(train_loader):
        #         lr = set_lr(optimizer, i, EPOCH, LR)
        batch_data = batch_data.type(torch.FloatTensor).to(device)  #
        batch_label = batch_label.type(torch.FloatTensor).to(device)  # CrossEntropy的target是longtensor，且要是1-D，不是one hot编码形式

        prediction = model(batch_data)
        loss = loss_func(prediction, batch_label)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        total_train_loss.append(loss.item())
    train_loss.append(np.mean(total_train_loss))  # 存入平均交叉熵

    total_valid_loss = []
    pred_list = []
    target_list = []
    model.eval()  # 进入验证模式
    with torch.no_grad():
        for step, (batch_data, batch_label) in enumerate(valid_loader):
            batch_data = batch_data.type(torch.FloatTensor).to(device)
            batch_label = batch_label.type(torch.FloatTensor).to(device)
            prediction = model(batch_data)  # rnn output
            #         h_s = h_s.data        # repack the hidden state, break the connection from last iteration
            #         h_c = h_c.data        # repack the hidden state, break the connection from last iteration
            loss = loss_func(prediction, batch_label)  # calculate loss
            total_valid_loss.append(loss.item())
            pred_list.append(torch.argmax(prediction, dim=1).cpu())
            target_list.append(torch.argmax(batch_label, dim=1).cpu())

    # 计算
    pred_list = torch.cat(pred_list, 0).numpy()
    target_list = torch.cat(target_list, 0).numpy()

    f1 = f1_score(y_true=target_list, y_pred=pred_list, average='macro')  # 也可以指定micro模式
    acc = accuracy_score(y_true=target_list, y_pred=pred_list)
    # recall = recall_score(y_true=target_list, y_pred=pred_list, average='macro')  # 也可以指定micro模式

    valid_loss.append(np.mean(total_valid_loss))

    if (valid_loss[-1] < min_valid_loss):
        torch.save({'epoch': i, 'model': model, 'train_loss': train_loss,
                    'valid_loss': valid_loss}, './LSTM.model')  # 保存字典对象，里面'model'的value是模型
        #         torch.save(optimizer, './LSTM.optim')     # 保存优化器
        min_valid_loss = valid_loss[-1]

    # 编写日志
    log_string = ('iter: [{:d}/{:d}], acc: {:0.4f}, f1: {:0.4f}, train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                  'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), EPOCH,
                                                                  acc, f1,
                                                                  train_loss[-1],
                                                                  valid_loss[-1],
                                                                  min_valid_loss,
                                                                  optimizer.param_groups[0]['lr'])
    # mult_step_scheduler.step()  # 学习率更新
    # 服务器一般用的世界时，需要加8个小时，可以视情况把加8小时去掉
    print(str(datetime.now()) + ': ')
    print(log_string)  # 打印日志
    # log('./LSTM.log', log_string)  # 保存日志
    if acc > max_acc:
        print(f"save model in {weight_path}net_params-{acc}.pth")
        torch.save(model.state_dict(), weight_path + f'net_params-{acc}.pth')
    max_acc = max(acc, max_acc)
