import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Calculate fc1 input size based on input dimensions
        # CIFAR/SVHN: 32x32 -> 28x28 -> 14x14 -> 10x10 -> 5x5 -> 16*5*5 = 400
        # FMNIST: 28x28 -> 24x24 -> 12x12 -> 8x8 -> 4x4 -> 16*4*4 = 256
        # We will handle this dynamically or assume 32x32 (pad FMNIST)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # If input is 28x28 (FMNIST), pad to 32x32
        if x.shape[2] == 28:
            x = F.pad(x, (2, 2, 2, 2))
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
