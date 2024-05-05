"""CLDNNLike model for RadioML.

# Reference:

- [CONVOLUTIONAL,LONG SHORT-TERM MEMORY, FULLY CONNECTED DEEP NEURAL NETWORKS ]

Adapted from code contributed by Mika.
"""
import os
import torch.nn as nn
import torch



class CNNLSTM_Net(nn.Module):
    def __init__(self):
        super(CNNLSTM_Net, self).__init__()
        self.LSTM_input_size = 2
        self.bilstm_hidden_layers = 32
        self.num_layers = 1

        self.dense_dim = 32
        self.output_dim = 10

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, (5,1)), 
            nn.Dropout(p=0.5), 
            nn.ReLU(),
            nn.MaxPool2d((2,1)), 
            nn.Conv2d(8, 1, (3,1)), 
            nn.Dropout(p=0.5), 
            nn.ReLU(),
            nn.MaxPool2d((2,1)), 
        )
        self.bilstm = nn.LSTM(
            input_size=self.LSTM_input_size,
            hidden_size=self.bilstm_hidden_layers,
            num_layers=self.num_layers,
            batch_first=True, 
            bidirectional=True
        )
        self.dense = nn.Sequential(
            nn.Linear(self.bilstm_hidden_layers*2*30, self.dense_dim),  # 最后一个时序的输出接一个全连接层
            nn.Linear(self.dense_dim, self.output_dim)  # 最后一个时序的输出接一个全连接层
        )
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=0)
        self.h_s = None
        self.h_c = None

    def forward(self, x):
        # 这里x.size(0)就是batch_size
        # Initialize cell state
        h0 = torch.rand(self.num_layers*2, x.size(0), self.bilstm_hidden_layers, device=x.device)
        c0 = torch.rand(self.num_layers*2, x.size(0), self.bilstm_hidden_layers, device=x.device)

        x = x.view(x.size(0),1, 128, 2)
        
        x = self.conv(x)
        x = x.view(x.size(0), 30, self.LSTM_input_size)
        x, (h_s, h_c) = self.bilstm(x,(h0,c0))  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        x = self.flatten(x)
        # print(x.shape)
        output = self.dense(x)
        return self.softmax(output)


class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.hidden_dim = 128
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.hidden_out = nn.Linear(self.hidden_dim, 10)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        # x = torch.flatten(x,1)
        # print(x.shape)
        h0 = torch.rand(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        # 这里x.size(0)就是batch_size

        # Initialize cell state
        c0 = torch.rand(self.num_layers, x.size(0), self.hidden_dim, device=x.device)

        r_out, (h_s, h_c) = self.lstm(x,(h0.detach(), c0.detach()))  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output[:,-1,:]


class LSTMNet11(nn.Module):
    def __init__(self):
        super(LSTMNet11, self).__init__()
        self.hidden_dim = 128
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.hidden_out = nn.Linear(self.hidden_dim, 11)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        # x = torch.flatten(x,1)
        # print(x.shape)
        h0 = torch.rand(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        # 这里x.size(0)就是batch_size

        # Initialize cell state
        c0 = torch.rand(self.num_layers, x.size(0), self.hidden_dim, device=x.device)

        r_out, (h_s, h_c) = self.lstm(x,(h0.detach(), c0.detach()))  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output[:,-1,:]

class BiLSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.hidden_dim = 128
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.hidden_out = nn.Linear(self.hidden_dim*2, 10)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        # x = torch.flatten(x,1)
        # print(x.shape)
        h0 = torch.rand(self.num_layers*2, x.size(0), self.hidden_dim, device=x.device)
        # 这里x.size(0)就是batch_size

        # Initialize cell state
        c0 = torch.rand(self.num_layers*2, x.size(0), self.hidden_dim, device=x.device)

        r_out, (h_s, h_c) = self.lstm(x,(h0.detach(), c0.detach()))  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output[:,-1,:]

class BiLSTMNet11(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.hidden_dim = 128
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.hidden_out = nn.Linear(self.hidden_dim*2, 11)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        # x = torch.flatten(x,1)
        # print(x.shape)
        h0 = torch.rand(self.num_layers*2, x.size(0), self.hidden_dim, device=x.device)
        # 这里x.size(0)就是batch_size

        # Initialize cell state
        c0 = torch.rand(self.num_layers*2, x.size(0), self.hidden_dim, device=x.device)

        r_out, (h_s, h_c) = self.lstm(x,(h0.detach(), c0.detach()))  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output[:,-1,:]