import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN class, which inherits from nn.Module
class CNN(nn.Module):
    # The constructor method initializes the model.
    def __init__(self, num_filters=[32, 64, 128, 256, 512], kernel_size=[3,3,3,3,3], num_dense=[512], num_classes=10, activation=F.relu, use_batchnorm=True, use_dropout=True, dropout_prob=0.12, padding=1, img_size = 224):
        super(CNN, self).__init__()
        
        # Store configuration options
        self.activation = activation  # Set activation function
        self.pool = nn.MaxPool2d(2, 2) # Set max pooling layer with kernel size = 2, stride = 2
        self.use_batchnorm = use_batchnorm # Bool deciding usage of batch normalization
        self.use_dropout = use_dropout # Bool deciding usage of dropout

        # Ensure kernel_size and num_filters are of length corresponding to number of layers
        assert len(num_filters)==len(kernel_size), "Mismatch in kernel_size num_filters length!"

        # Initialize container modules for convolutional layers, batch norms and fully connected layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        in_channels = 3  # Initial input channels (RGB)
        # Build convolutional layers
        for i in range(len(num_filters)):
            # If padding is not set, use same padding
            if padding is None:
                padding = kernel_size[i] // 2
            # Add conv2D layer
            self.convs.append(nn.Conv2d(in_channels, num_filters[i], kernel_size[i], padding=padding))
            # Add batchnorm if enabled
            if self.use_batchnorm:
                self.bns.append(nn.BatchNorm2d(num_filters[i]))
            in_channels = num_filters[i]  # Update input channels for next layer

        # Dummy input to calculate flattened size
        self.flattened_size = self._get_flattened_size((3, img_size, img_size))
        
        # Fully connected layers
        input_size = self.flattened_size
        for output_size in num_dense:
            self.fcs.append(nn.Linear(input_size, output_size))
            input_size = output_size
        self.fcs.append(nn.Linear(input_size, num_classes)) # Final classification layer
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob) if self.use_dropout else nn.Identity()

        # Initialize weights
        self._initialize_weights()

    def _get_flattened_size(self, input_shape):
        """
        Computes the number of features after passing an input image through
        all convolutional and pooling layers — used to determine the input
        size for the first fully connected (dense) layer.

        Parameters:
        -----------
        input_shape : tuple
            Shape of input image (e.g., (3, 224, 224) for RGB image of size 224x224)

        Returns:
        --------
        int : The total number of features (flattened) after conv+pool layers
        """

        # Disable gradient computation
        with torch.no_grad():
            # Create dummy tensor with batch size 1 and given input shape
            x = torch.zeros(1, *input_shape)
            # Pass through each convolutional layer
            for i, conv in enumerate(self.convs):
                # Apply convolution, activation and pooling
                x = self.pool(self.activation(conv(x)))
                # if i % 2 == 1:
                #     x = self.pool(x)
            return x.numel() # Return total features after all convolutional layers
        
    def _initialize_weights(self):
        """
        Initializes weights of the CNN layers using standard initialization techniques:
        - He (Kaiming) initialization for Conv2d layers (best for ReLU activations)
        - Constant initialization for BatchNorm2d layers
        - Xavier (Glorot) initialization for Linear (fully connected) layers
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He/Kaiming normal initialization is recommended for layers with ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Biases are usually initialized to 0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm weights (scale γ) initialized to 1
                # Bias (shift β) initialized to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Glorot initialization for fully connected layers
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the CNN model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, H, W), where H and W are the spatial 
            dimensions of the image (e.g., 224 x 224).

        Returns:
        --------
        torch.Tensor
            Output logits tensor of shape (batch_size, num_classes), representing 
            the raw scores for each class.
        """

        # Pass input through convolutional blocks
        for i in range(len(self.convs)):
            x = (self.convs[i](x)) # Apply convolution operation
            if self.use_batchnorm: 
                x = self.bns[i](x) # Apply batch norm if enabled
            x = self.pool(self.activation(x)) # Apply activation and pooling
            # if i%2==1:
            #     x = self.pool(self.activation(x))
            # else:
            #     x = self.activation(x)
        x = x.view(x.size(0), -1)  # Flatten convolution layer output
        # Pass through all fully connected layers
        for i in range(len(self.fcs)-1):
            x = self.activation(self.fcs[i](x)) 
            x = self.dropout(x) # Dropout only after dense layers
        x = self.fcs[len(self.fcs)-1](x) # Finaly layer does not have activation and dropout
        # Softmax output is not used, because nn.CrossEntropyLoss() expects logits
        return x