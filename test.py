import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle, random, sys#, h5py
import mltools,rmldataset2016
from rmlmodels.LSTMModel import LSTMNet
# import rmlmodels.BidLSTMModel as bidlstm
import csv
from datetime import datetime  # 用于计算时间
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import os
from rmlmodels.encoder import Net


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set up some params
EPOCH =  1000    # number of epochs to train on
BATCH_SIZE = 1024  # training batch size
process_num = 8 # linux可改为8
LEARNING_RATE = 5e-3

def predict(model,filepath):
    print("加载模型"+filepath)
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    model.eval()

    (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = rmldataset2016.load_data()

    pred_list = []
    target_list = []
    # Plot confusion matrix
    test_loader = DataLoader(list(zip(X_test,Y_test)), batch_size=BATCH_SIZE, num_workers=process_num, pin_memory=True, shuffle=False)
    with torch.no_grad():
        for step, (batch_data, batch_label) in enumerate(test_loader):
            batch_data = batch_data.type(torch.FloatTensor).to(device)
            batch_label = batch_label.type(torch.FloatTensor).to(device)
            prediction = model(batch_data) 
            pred_list.append(prediction.cpu())
            target_list.append(batch_label.cpu())
    # 计算
    test_Y_hat = torch.cat(pred_list,0).numpy()
    # test_Y = torch.cat(target_list,0).numpy()

    classes = mods
    confnorm,_,_ = mltools.calculate_confusion_matrix(Y_test,test_Y_hat,classes)
    mltools.plot_confusion_matrix(confnorm, labels=['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'],save_filename='figure/lstm3_total_confusion.png')

    # Plot confusion matrix
    acc = {}
    acc_mod_snr = np.zeros( (len(classes),len(snrs)) )
    i = 0
    for snr in snrs:

        # extract classes @ SNR
        # test_SNRs = map(lambda x: lbl[x][1], test_idx)
        test_SNRs = [lbl[x][1] for x in test_idx]

        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # estimate classes
        test_i_loader = DataLoader(list(zip(test_X_i,test_Y_i)), batch_size=BATCH_SIZE, num_workers=process_num, pin_memory=True, shuffle=False)
        pred_list = []
        target_list = []
        with torch.no_grad():
            for step, (batch_data, batch_label) in enumerate(test_i_loader):
                batch_data = batch_data.type(torch.FloatTensor).to(device)
                batch_label = batch_label.type(torch.FloatTensor).to(device)
                prediction = model(batch_data) 
                pred_list.append(prediction.cpu())
                target_list.append(batch_label.cpu())
        # 计算
        test_Y_i_hat = torch.cat(pred_list,0).numpy()
        confnorm_i,cor,ncor = mltools.calculate_confusion_matrix(test_Y_i,test_Y_i_hat,classes)
        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)
        with open('acc111.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([result])
        mltools.plot_confusion_matrix(confnorm_i, labels=['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'], title="Confusion Matrix",save_filename="figure/Confusion(SNR=%d)(ACC=%2f).png" % (snr,100.0*acc[snr]))

        acc_mod_snr[:,i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i,axis=1),3)
        i = i +1

    #plot acc of each mod in one picture
    dis_num=10
    for g in range(int(np.ceil(acc_mod_snr.shape[0]/dis_num))):
        assert (0 <= dis_num <= acc_mod_snr.shape[0])
        beg_index = g*dis_num
        end_index = np.min([(g+1)*dis_num,acc_mod_snr.shape[0]])

        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Mod")

        for i in range(beg_index,end_index):
            plt.plot(snrs, acc_mod_snr[i], label=classes[i])
            # 设置数字标签
            for x, y in zip(snrs, acc_mod_snr[i]):
                plt.text(x, y, y, ha='center', va='bottom', fontsize=8)

        plt.legend()
        plt.grid()
        plt.savefig('figure/acc_with_mod_{}.png'.format(g+1))
        plt.close()
    #save acc for mod per SNR
    fd = open('predictresult/acc_for_mod_on_lstm.dat', 'wb')
    pickle.dump((acc_mod_snr), fd)
    fd.close()

    # Save results to a pickle file for plotting later
    print(acc)
    fd = open('predictresult/lstm.dat','wb')
    pickle.dump( (acc) , fd )

    # Plot accuracy curve
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title(" Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')


if __name__ == '__main__':
    filepath = "/root/LSTM2torch/weights/net_params-0.8856474358974359.pth"
    
    model = LSTMNet().to(device)
    # model = Net().to(device)
    
    predict(model,filepath)