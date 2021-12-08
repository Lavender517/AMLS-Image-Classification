import torch
import torch.nn as nn
from torchvision import models

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Sequential(nn.Linear(4096, 4))
        for param in model.classifier[6].parameters():
            param.requires_grad = True
        self.model = model

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        return x
    
    # def initialize(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear): # Check if Object 'm' is the known type: nn.Linear, return True or False
    #             nn.init.kaiming_normal_(m.weight.data)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')