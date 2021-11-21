import torch
import torch.nn as nn
import torch.nn.functional as F   # 激励函数的库
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

class MLP(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self, dropout_coef):
        super(MLP,self).__init__()
        # 初始化三层神经网络，两个全连接的隐藏层，一个输出层
        self.dropout_coef = dropout_coef
        self.fc1 = nn.Linear(512*512, 64*64, bias=True)  
        self.fc2 = nn.Linear(64*64, 32*32, bias=True)
        self.fc3 = nn.Linear(32*32, 256, bias=True)
        self.fc4 = nn.Linear(256, 4, bias=True)   # 输出层
        
    def forward(self,din):
        '''
        Feedforward Function
        :param din: Input values
        :return: dout
        '''

        din = din.view(-1, 512*512)      # 将一个多行的Tensor,拼接成一行
        dout = self.fc1(din)     # 使用 relu 激活函数
        dout = F.dropout(dout, p=self.dropout_coef)
        dout = F.relu(dout)
        dout = self.fc2(dout)
        dout = F.dropout(dout, p=self.dropout_coef)
        dout = F.relu(dout)
        dout = F.relu(self.fc3(dout))
        dout = F.dropout(dout, p=self.dropout_coef)
        dout = F.relu(dout)
        dout = self.fc4(dout)  # 输出层使用 softmax 激活函数
        return dout
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)