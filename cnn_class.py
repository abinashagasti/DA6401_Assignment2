import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_filters=[32, 64, 128, 256, 512], kernel_size=5, num_dense=256, activation=F.relu):
        super(CNN, self).__init__()
        self.activation = activation  # Set activation function
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size, padding=0)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, padding=0)
        self.conv3 = nn.Conv2d(num_filters[1], num_filters[2], kernel_size, padding=0)
        self.conv4 = nn.Conv2d(num_filters[2], num_filters[3], kernel_size, padding=0)
        self.conv5 = nn.Conv2d(num_filters[3], num_filters[4], kernel_size, padding=0)

        # Define max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling

        # Compute final feature map size after 5 conv + pooling layers
        final_size = 224 // (2**5)  # Divide by 2 for each maxpool layer

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters[4] * final_size * final_size, num_dense)
        self.fc2 = nn.Linear(num_dense, 10)  # Output layer (10 classes)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))
        x = self.pool(self.activation(self.conv4(x)))
        x = self.pool(self.activation(self.conv5(x)))
        
        x = torch.flatten(x, 1)  # Flatten before FC layer
        x = self.activation(self.fc1(x))
        x = self.fc2(x)  # No activation here (will use CrossEntropyLoss)
        return x