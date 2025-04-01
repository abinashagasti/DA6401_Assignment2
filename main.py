import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Define transformations (you can modify this as needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (change if needed)
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Define dataset paths
train_dir = "../inaturalist_12k/train"
val_dir = "../inaturalist_12k/val"

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Check class mapping (optional)
print("Class names:", train_dataset.classes)  # List of class names
print("Class indices:", train_dataset.class_to_idx)  # Mapping class → index