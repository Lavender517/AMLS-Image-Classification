import torch
import torch.nn as nn
from torchvision import models

class VGG16(nn.Module): # VGG-16 Model
    def __init__(self):
        super(VGG16, self).__init__()
        model = models.vgg16(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4),
        )
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

class pre_trained_VGG16(nn.Module): # Pre-trained VGG-16 model
    def __init__(self):
        super(pre_trained_VGG16, self).__init__()
        model = models.vgg16(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4),
        )
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x