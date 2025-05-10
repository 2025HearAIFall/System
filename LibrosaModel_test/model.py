import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNEmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool2d((9, 23))  # Adaptive Pooling
        self.fc1 = nn.Linear(32 * 9 * 23, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Adaptive Pooling
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
