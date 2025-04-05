import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm

import wandb
import yaml

from cnn_class import CNN
from utils import *


def train():

    # wandb.init() # Initialize wandb run
    wandb.init(resume="allow")
    config = wandb.config # Config for wandb sweep

    # Experiment name
    wandb.run.name = f"lr_{config.learning_rate:.4f}_bs_{config.batch_size}_#filters_{config.num_filters}_#neurons_{config.num_dense}_drop_prob_{config.dropout_prob}_\
    kernel_{config.kernel_size}_padding_{config.padding}_data_aug_{config.data_augmentation}_bn_{config.use_batchnorm}_ac_{config.activation}_op_{config.optimizer}_wd_{config.weight_decay:.4f}"

   # Define dataset paths
    train_dir = "../inaturalist_12K/train"
    test_dir = "../inaturalist_12K/val"

    max_epochs = 10
    learning_rate = config.learning_rate
    batch_size = config.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    num_workers = 16 if torch.cuda.is_available() else 4 if torch.backends.mps.is_available() else 1

    train_loader, val_loader, _ = dataset_split(train_dir, test_dir, batch_size=batch_size, num_workers=num_workers, augmentation=config.data_augmentation)

    # Model instantiation with flexible hyperparameters
    # num_filters = [256, 128, 64, 32, 16]
    num_filters = config.num_filters
    kernel_size = config.kernel_size
    num_dense = config.num_dense
    dropout_prob = config.dropout_prob
    use_dropout = True
    use_batchnorm = config.use_batchnorm
    activation = get_activation(config.activation)

    model = CNN(num_filters=num_filters, num_dense=num_dense, kernel_size=kernel_size, activation=activation, use_batchnorm=use_batchnorm, use_dropout=use_dropout, dropout_prob=dropout_prob)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss() # Loss function
    optimizer = get_optimizer(optimizer_name=config.optimizer, model=model, learning_rate=learning_rate, weight_decay=config.weight_decay) # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"Training Parameters:")
    print(f"Learning Rate: {learning_rate}, num_filters: {num_filters}, kernel_size: {kernel_size}, num_dense: {num_dense}, batch_size: {batch_size}, dropout: {dropout_prob if use_dropout else None}, batchnorm: {use_batchnorm}")
    train_loop(train_loader, val_loader, model, loss_fn, optimizer, scheduler=scheduler, device=device, max_epochs=max_epochs, patience_stop=7, wandb_log=True)
    # test_loop(test_loader, model, loss_fn, device)

    wandb.finish()

if __name__=="__main__":
    mp.set_start_method('spawn')
    with open("sweep1.yaml", "r") as file:
        sweep_config = yaml.safe_load(file) # Read yaml file to store hyperparameters 

    # Initialize sweep
    # sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment_2")
    entity = "ee20d201-indian-institute-of-technology-madras"
    project = "DA6401_Assignment_2"
    sweep_id = "5t7ap5rq"
    # api = wandb.Api()

    # sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # if len(sweep.runs) >= 10:
    #     api.stop_sweep(sweep.id)
    #     print(f"Sweep {sweep.id} stopped after {len(sweep.runs)} runs.")

    # Start sweep agent
    wandb.agent(sweep_id, function=train, count=13, project=project)  # Run 10 experiments