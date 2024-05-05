import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle, random, sys#, h5py
import mltools,rmldataset2016_11
from rmlmodels.LSTMModel import LSTMNet
# import rmlmodels.BidLSTMModel as bidlstm
import csv
from datetime import datetime  # 用于计算时间
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import os
from rmlmodels.encoder import Net

(mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = rmldataset2016_11.load_data()

# print(test_idx)

def get_only_AMSSR(X_data,Y_data):
    X = X_data[np.where(Y_data[:][2]>0)]
    Y = Y_data[np.where(Y_data[:][2]>0)]
    return (X,Y)

# X_train, Y_train = get_upper_SNR(lbl, snrs, train_idx, X_train, Y_train)
X_val, Y_val = get_only_AMSSR(X_val, Y_val)

print(Y_val)