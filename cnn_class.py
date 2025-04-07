import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
            pad = kernel_size[i] // 2
            self.convs.append(nn.Conv2d(in_channels, num_filters[i], kernel_size[i], padding=pad))
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
            for i, conv in enumerate(self.convs):
                x = self.pool(self.activation(conv(x)))
                # if i % 2 == 1:
                #     x = self.pool(x)
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
            # if i%2==1:
            #     x = self.pool(self.activation(x))
            # else:
            #     x = self.activation(x)
        x = x.view(x.size(0), -1)  # Flatten
        for i in range(len(self.fcs)-1):
            x = self.activation(self.fcs[i](x))
            x = self.dropout(x)
        x = self.fcs[len(self.fcs)-1](x)
        return x

def pretrained_model(model_type: str = "ResNet18", num_classes = 10, k = 1):

    if model_type == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "VGG":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_type == "EfficientNetV2":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == "VisionTransformer":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    # Freeze all layers
    if k != 1:
        for param in model.parameters():
            param.requires_grad = False

        # Get all named parameters
        named_params = list(model.named_parameters())
        
        # Unfreeze the last `k` layers
        for name, param in named_params[-k:]:
            param.requires_grad = True
            print(f"Unfreezing layer: {name}")

    return model