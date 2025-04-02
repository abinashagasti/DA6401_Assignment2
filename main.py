import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm

from cnn_class import CNN
from utils import *


def main():

   # Define dataset paths
    train_dir = "../inaturalist_12K/train"
    test_dir = "../inaturalist_12K/val"

    learning_rate = 1e-3
    batch_size = 64
    max_epochs = 30

    train_loader, val_loader, test_loader = dataset_split(train_dir, test_dir, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model instantiation with flexible hyperparameters
    num_filters = [16,32,64,128,256]
    model = CNN(num_filters=num_filters, num_dense=1024).to(device)
    model.apply(init_weights)

    loss_fn = nn.CrossEntropyLoss() # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) # Optimizer
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_loop(train_loader, val_loader, model, loss_fn, optimizer, scheduler=None, device=device, max_epochs=max_epochs)
    test_loop(test_loader, model, loss_fn, device)

if __name__=="__main__":
    mp.set_start_method('fork')
    main()