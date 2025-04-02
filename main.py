import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from cnn_class import CNN
from utils import *


def main():
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

    # Check class mapping (optional)
    # print("Class names:", train_dataset.classes)  # List of class names
    # print("Class indices:", train_dataset.class_to_idx)  # Mapping class → index

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model instantiation with flexible hyperparameters
    model = CNN().to(device)

    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    loss_fn = nn.CrossEntropyLoss() # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Optimizer

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, batch_size, device)
        val_loop(val_loader, model, loss_fn, device)
    print("Done!")

if __name__=="__main__":
    mp.set_start_method('fork')
    main()