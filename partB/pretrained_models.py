import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
    elif model_type == "GoogLeNet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "InceptionV3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

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