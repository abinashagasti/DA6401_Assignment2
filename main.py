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

    learning_rate = 5e-4
    batch_size = 32
    max_epochs = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    num_workers = 16 if torch.cuda.is_available() else 4 if torch.backends.mps.is_available() else 1

    train_loader, val_loader, test_loader = dataset_split(train_dir, test_dir, batch_size=batch_size, num_workers=num_workers)

    # Model instantiation with flexible hyperparameters
    num_filters = [32, 64, 128, 256]
    kernel_size = [3, 3, 3, 3]
    num_dense = [128]
    dropout_prob = 0.3
    use_dropout = True
    use_batchnorm = False
    model = CNN(num_filters=num_filters, num_dense=num_dense, kernel_size=kernel_size, use_batchnorm=use_batchnorm, use_dropout=use_dropout, dropout_prob=dropout_prob)
    model = model.to(device)
    # model.apply(init_weights)

    loss_fn = nn.CrossEntropyLoss() # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_loop(train_loader, val_loader, model, loss_fn, optimizer, scheduler=scheduler, device=device, max_epochs=max_epochs, patience_stop=10)
    test_loop(test_loader, model, loss_fn, device)
    print(f"Training Parameters:")
    print(f"Learning Rate: {learning_rate}, num_filters: {num_filters}, kernel_size: {kernel_size}, num_dense: {num_dense}, batch_size: {batch_size}, dropout: {dropout_prob if use_dropout else None}, batchnorm: {use_batchnorm}")

if __name__=="__main__":
    mp.set_start_method('spawn')
    main()