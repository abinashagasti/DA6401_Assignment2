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
    max_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    num_workers = 16 if torch.cuda.is_available() else 4 if torch.backends.mps.is_available() else 1

    train_loader, val_loader, test_loader = dataset_split(train_dir, test_dir, batch_size=batch_size, num_workers=num_workers)

    # Model instantiation with flexible hyperparameters
    num_filters = [128, 128, 256, 256, 512]
    num_dense = [1024]
    model = CNN(num_filters=num_filters, num_dense=num_dense, use_batchnorm=False, use_dropout=False)
    model = model.to(device)
    model.apply(init_weights)

    loss_fn = nn.CrossEntropyLoss() # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_loop(train_loader, val_loader, model, loss_fn, optimizer, scheduler=scheduler, device=device, max_epochs=max_epochs)
    test_loop(test_loader, model, loss_fn, device)

if __name__=="__main__":
    mp.set_start_method('fork')
    main()