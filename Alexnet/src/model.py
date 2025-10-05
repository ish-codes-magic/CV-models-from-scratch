"""AlexNet model architecture."""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet implementation for CIFAR-10 classification.
    
    Architecture:
    - 5 Convolutional layers with BatchNorm and ReLU activation
    - 3 Fully connected layers with Dropout
    - Input: 227x227x3 RGB images
    - Output: 10 classes (CIFAR-10)
    """
    
    def __init__(self, num_classes=10):
        """
        Initialize AlexNet model.
        
        Args:
            num_classes (int): Number of output classes
        """
        super(AlexNet, self).__init__()
        
        # Layer 1: Conv11/s4 + BN + ReLU + MaxPool
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Layer 2: Conv5/s1 + BN + ReLU + MaxPool
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Layer 3: Conv3/s1 + BN + ReLU
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=False)
        )
        
        # Layer 4: Conv3/s1 + BN + ReLU
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=False)
        )
        
        # Layer 5: Conv3/s1 + BN + ReLU + MaxPool
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Fully connected layers
        # After layer5: 256 x 6 x 6 = 9216
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(inplace=False)
        )
        
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, 227, 227)
            
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes)
        """
        # Convolutional layers
        out = self.layer1(x)   # (N, 96, 27, 27)
        out = self.layer2(out)  # (N, 256, 13, 13)
        out = self.layer3(out)  # (N, 384, 13, 13)
        out = self.layer4(out)  # (N, 384, 13, 13)
        out = self.layer5(out)  # (N, 256, 6, 6)
        
        # Flatten
        out = out.reshape(out.size(0), -1)  # (N, 9216)
        
        # Fully connected layers
        out = self.fc(out)    # (N, 4096)
        out = self.fc1(out)   # (N, 4096)
        out = self.fc2(out)   # (N, num_classes)
        
        return out
    
    def get_num_parameters(self):
        """
        Calculate total number of parameters in the model.
        
        Returns:
            tuple: (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

