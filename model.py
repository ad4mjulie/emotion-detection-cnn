import sys
from unittest.mock import MagicMock
# Mock _lzma to avoid ModuleNotFoundError in environments where it's missing (e.g. macOS pyenv without xz)
sys.modules['_lzma'] = MagicMock()

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    """
    CNN Architecture designed for 48x48 grayscale input.
    Based on standard architectures for the FER-2013 dataset.
    """
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.25)
        
        # Block 4
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(0.25)
        
        # Flatten and Dense layers
        # Input 48x48 -> Pool1 (24x24) -> Pool2 (12x12) -> Pool3 (6x6) -> Pool4 (3x3)
        self.fc1 = nn.Linear(512 * 3 * 3, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.drop_fc2 = nn.Dropout(0.5)
        
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.drop1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(F.elu(self.bn4(self.conv4(x)))))
        
        # Flatten
        x = x.view(-1, 512 * 3 * 3)
        
        # Fully connected layers
        x = self.drop_fc1(F.elu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc2(F.elu(self.bn_fc2(self.fc2(x))))
        
        x = self.output(x)
        return x
