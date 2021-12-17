import torch
import torch.nn as nn
import torch.nn.functional as F   # Libraries providing activation functions

class MLP(torch.nn.Module):
    def __init__(self, dropout_coef):
        super(MLP, self).__init__()
        # Initialize a four-layer neural network, two fully connected hidden layers, and one output layer
        self.dropout_coef = dropout_coef
        self.fc1 = nn.Linear(512*512, 64*64, bias=True) # Output has 64*64 neurons, add bias
        self.fc2 = nn.Linear(64*64, 32*32, bias=True)
        self.fc3 = nn.Linear(32*32, 256, bias=True)
        self.fc4 = nn.Linear(256, 4, bias=True)   # Output Layer
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=self.dropout_coef)
        
    def forward(self, din):
        '''
        Feedforward Function
        :param din: Input values
        :return: dout
        '''

        din = din.view(-1, 512*512)     # Concatenate a multi-line Tensor into one line
        dout = self.fc1(din)
        dout = F.dropout(dout, p=self.dropout_coef)    
        dout = F.relu(dout)             # Use ReLu activation function
        dout = self.fc2(dout)
        dout = F.dropout(dout, p=self.dropout_coef)    
        dout = F.relu(dout)             # Use ReLu activation function
        dout = self.fc3(dout)
        dout = F.dropout(dout, p=self.dropout_coef)    
        dout = F.relu(dout)             # Use ReLu activation function
        dout = self.fc4(dout)  # Output layer use Softmax function
        return dout
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): # Check if Object 'm' is the known type: nn.Linear, return True or False
                nn.init.kaiming_normal_(m.weight.data)