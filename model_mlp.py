import torch
import torch.nn as nn
import torch.nn.functional as F   # 激励函数的库
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

class MLP(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self):
        super(MLP,self).__init__()
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = nn.Linear(512*512,128*128)  
        self.fc2 = nn.Linear(128*128,64*64)
        self.fc3 = nn.Linear(64*64, 256)
        self.fc4 = nn.Linear(256,4)   # 输出层
        
    def forward(self,din):
        '''
        Feedforward Function
        :param din: Input values
        :return: dout
        '''

        din = din.view(-1, 512*512)              # 将一个多行的Tensor,拼接成一行
        dout = F.relu(self.fc1(din))             # 使用 relu 激活函数
        dout = F.relu(self.fc2(dout))
        dout = F.relu(self.fc3(dout))
        dout = self.fc4(dout)  # 输出层使用 softmax 激活函数
        return dout