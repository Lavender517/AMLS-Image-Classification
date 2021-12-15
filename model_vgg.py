import torch
import torch.nn as nn
from torchvision import models

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
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

    #     model = models.vgg16(pretrained=True)
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     model.classifier[6] = nn.Sequential(nn.Linear(4096, 4))
    #     for param in model.classifier[6].parameters():
    #         param.requires_grad = True
    #     self.model = model

