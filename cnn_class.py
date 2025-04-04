import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_filters=[32, 64, 128, 256, 512], kernel_size=[5,5,5,3,3], num_dense=[256], num_classes=10, activation=F.relu, use_batchnorm=False, use_dropout=False, dropout_prob=0.2, padding=1):
        super(CNN, self).__init__()
        self.activation = activation  # Set activation function
        self.pool = nn.MaxPool2d(2, 2)
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        assert len(num_filters)==len(kernel_size), "Mismatch in kernel_size num_filters length!"

        # Convolutional layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()
        in_channels = 3  # Initial input channels (RGB)
        for i in range(len(num_filters)):
            self.convs.append(nn.Conv2d(in_channels, num_filters[i], kernel_size[i], padding=padding))
            if self.use_batchnorm:
                self.bns.append(nn.BatchNorm2d(num_filters[i]))
            in_channels = num_filters[i]  # Update input channels for next layer

        # Dummy input to calculate flattened size
        self.flattened_size = self._get_flattened_size((3, 224, 224))
        
        # Fully connected layers
        input_size = self.flattened_size
        for output_size in num_dense:
            self.fcs.append(nn.Linear(input_size, output_size))
            input_size = output_size
        self.fcs.append(nn.Linear(input_size, num_classes))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob) if self.use_dropout else nn.Identity()

        # Initialize weights
        self._initialize_weights()

    def _get_flattened_size(self, input_shape):
        """Passes a dummy tensor through conv layers to compute flattened size."""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for conv in self.convs:
                x = self.pool(self.activation(conv(x)))
            return x.numel()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(len(self.convs)):
            x = (self.convs[i](x))
            if self.use_batchnorm:
                x = self.bns[i](x)
            x = self.pool(self.activation(x))
        x = x.view(x.size(0), -1)  # Flatten
        for i in range(len(self.fcs)-1):
            x = self.activation(self.fcs[i](x))
            x = self.dropout(x)
        x = self.fcs[len(self.fcs)-1](x)
        return x