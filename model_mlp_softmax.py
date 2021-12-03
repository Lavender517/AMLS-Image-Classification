import torch
import torch.nn as nn
import torch.nn.functional as F   # 激励函数的库
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

class MLP_softmax(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self):
        super(MLP_softmax, self).__init__()
        # 初始化单层全连接网络，即softmax regression
        self.fc1 = nn.Linear(512*512, 4) # Output has 64*64 neurons, add bias
        
    def forward(self, din):
        '''
        Feedforward Function
        :param din: Input values
        :return: dout
        '''

        din = din.view(-1, 512*512)      # 将一个多行的Tensor,拼接成一行
        dout = self.fc1(din)
        return dout
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): # Check if Object 'm' is the known type: nn.Linear, return True or False
                nn.init.kaiming_normal_(m.weight.data)