import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_filters=[32, 64, 128, 256, 512], kernel_size=3, num_dense=256, num_classes=10, activation=F.relu):
        super(CNN, self).__init__()
        self.activation = activation  # Set activation function
        self.pool = nn.MaxPool2d(2, 2)

        # Convolutional layers
        self.convs = nn.ModuleList()
        in_channels = 3  # Initial input channels (RGB)
        for out_channels in num_filters:
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            in_channels = out_channels  # Update input channels for next layer

        # Dummy input to calculate flattened size
        self.flattened_size = self._get_flattened_size((3, 224, 224))
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, num_dense)
        self.fc2 = nn.Linear(num_dense, num_classes)

    def _get_flattened_size(self, input_shape):
        """Passes a dummy tensor through conv layers to compute flattened size."""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for conv in self.convs:
                x = self.pool(self.activation(conv(x)))
            return x.numel()

    def forward(self, x):
        for conv in self.convs:
            x = self.pool(self.activation(conv(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x