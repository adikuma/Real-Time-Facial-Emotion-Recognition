import torch.nn as nn
from .base_model import BaseModel

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        return self.block(x)

class Net(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)    
        )
        
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)    
        )
        
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flat_features = 512 * 3 * 3
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
                
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x