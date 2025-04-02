import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_filters=[32, 64, 128, 256, 512], kernel_size=5, num_dense=256, activation=F.relu):
        super(CNN, self).__init__()
        self.activation = activation  # Set activation function

        # Define max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, num_filters[0], kernel_size),
            self.activation(),
            self.pool,
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size),
            self.activation(),
            self.pool,
            nn.Conv2d(num_filters[1], num_filters[2], kernel_size),
            self.activation(),
            self.pool,
            nn.Conv2d(num_filters[2], num_filters[3], kernel_size),
            self.activation(),
            self.pool,
            nn.Conv2d(num_filters[3], num_filters[4], kernel_size),
            self.activation(),
            self.pool
        )
        

        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)  # Flatten before passing to linear layer
        x = self.fc_layers(x)
        return x