import sys
from unittest.mock import MagicMock

# Robust mock for _lzma to satisfy the lzma standard library on macOS environments missing the C extension
mock_lzma = MagicMock()
# Constants expected by lzma.py
for name in ['FORMAT_XZ', 'FORMAT_ALONE', 'FORMAT_RAW', 'FORMAT_AUTO', 
            'CHECK_NONE', 'CHECK_CRC32', 'CHECK_CRC64', 'CHECK_SHA256',
            'FILTER_LZMA1', 'FILTER_LZMA2', 'FILTER_DELTA', 'FILTER_X86', 'FILTER_IA64', 
            'FILTER_ARM', 'FILTER_ARMTHUMB', 'FILTER_SPARC', 'FILTER_POWERPC',
            'MF_BT2', 'MF_BT3', 'MF_BT4', 'MF_HC3', 'MF_HC4', 'MODE_READ', 'MODE_WRITE']:
    setattr(mock_lzma, name, 1) # Value doesn't matter as long as it exists
sys.modules['_lzma'] = mock_lzma

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A standard Residual Block with two convolutional layers and a shortcut connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.elu(out)
        return out

class EmotionCNN(nn.Module):
    """
    Advanced CNN Architecture using Residual Blocks and Global Average Pooling.
    Designed for 48x48 grayscale input.
    """
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        self.in_channels = 64
        
        # Initial Convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.elu = nn.ELU(inplace=True)
        
        # Residual Layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Global Average Pooling + Output
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.elu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

