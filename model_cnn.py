import torch
import torch.nn as nn
import torch.nn.functional as F   # Activation Function

class CNN(nn.Module):
    def __init__(self, dropout_coef):
        super(CNN, self).__init__()
        self.dropout_coef = dropout_coef
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2)
        # self.conv1 = nn.Conv2d(1, 5, kernel_size=5, padding=2)
        # self.conv2 = nn.Conv2d(5, 10, kernel_size=5, padding=2)
        # self.conv3 = nn.Conv2d(10, 20, kernel_size=5, padding=2)
        self.conv_drop = nn.Dropout2d(p=self.dropout_coef)
        self.fc1 = nn.Linear(20*128*128, 60*60)
        self.fc2 = nn.Linear(60*60, 4)
        # self.fc3 = nn.Linear(16*16, 4)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv2(x)), 2))
        # x = F.relu(F.max_pool2d(self.conv_drop(self.conv3(x)), 2))
        x = x.view(-1, 20*128*128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x)
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): # Check if Object 'm' is the known type: nn.Linear, return True or False
                nn.init.kaiming_normal_(m.weight.data)
    # def initialize(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_in')