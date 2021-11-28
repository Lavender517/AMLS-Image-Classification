import torch
from torch import nn


class AlexNet(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self, dropout_coef):
        super(AlexNet, self).__init__()
        # 初始化四层神经网络，三个全连接的隐藏层，一个输出层
        self.dropout_coef = dropout_coef
        self.fc1 = nn.Linear(512*512, 64*64, bias=True) # Output has 64*64 neurons, add bias
        self.fc2 = nn.Linear(64*64, 32*32, bias=True)
        self.fc3 = nn.Linear(32*32, 256, bias=True)
        self.fc4 = nn.Linear(256, 4, bias=True)   # 输出层
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=self.dropout_coef)
        
    def forward(self, din):
        '''
        Feedforward Function
        :param din: Input values
        :return: dout
        '''

        din = din.view(-1, 512*512)      # 将一个多行的Tensor,拼接成一行
        dout = self.fc1(din)
        dout = F.dropout(dout, p=self.dropout_coef)    
        dout = F.relu(dout)             # Use ReLu activation function
        dout = self.fc2(dout)
        dout = F.dropout(dout, p=self.dropout_coef)    
        dout = F.relu(dout)             # Use ReLu activation function
        dout = self.fc3(dout)
        dout = F.dropout(dout, p=self.dropout_coef)    
        dout = F.relu(dout)             # Use ReLu activation function
        dout = self.fc4(dout)  # 输出层使用 softmax 激活函数
        return dout
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): # Check if Object 'm' is the known type: nn.Linear, return True or False
                nn.init.kaiming_normal_(m.weight.data)


    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过度拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))