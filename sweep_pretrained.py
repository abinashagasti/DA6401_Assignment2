import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

import time
import wandb
import yaml

from cnn_class import CNN, pretrained_model
from utils import *


def train():

    wandb.init() # Initialize wandb run
    # wandb.init(resume="allow")
    config = wandb.config # Config for wandb sweep

    # Experiment name
    wandb.run.name = f"{config.model_type}"

   # Define dataset paths
    train_dir = "../inaturalist_12K/train"
    test_dir = "../inaturalist_12K/val"

    max_epochs = 20
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    weight_decay = config.weight_decay

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    num_workers = 20 if torch.cuda.is_available() else 4 if torch.backends.mps.is_available() else 1

    train_loader, val_loader, _ = dataset_split(train_dir, test_dir, batch_size=batch_size, num_workers=num_workers, augmentation=True)

    model_type = config.model_type # Options = ["ResNet18", "ResNet50", "GoogLeNet", "VGG", "InceptionV3", "EfficientNetV2", "VisionTransformer"]
    model = pretrained_model(model_type, k=config.trainable_layers)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss() # Loss function
    optimizer = get_optimizer(optimizer_name=config.optimizer, model=model, learning_rate=learning_rate, weight_decay=weight_decay) # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"Training with {config.model_type}:")
    train_loop(train_loader, val_loader, model, loss_fn, optimizer, scheduler=scheduler, device=device, max_epochs=max_epochs, patience_stop=7, wandb_log=True)
    print("Resting the machine for 120 seconds.")
    time.sleep(120)

    wandb.finish()

if __name__=="__main__":
    # mp.set_start_method('spawn')
    with open("sweep_pretrained_fine.yaml", "r") as file:
        sweep_config = yaml.safe_load(file) # Read yaml file to store hyperparameters 

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment_2")
    entity = "ee20d201-indian-institute-of-technology-madras"
    project = "DA6401_Assignment_2"
    # sweep_id = "5t7ap5rq" # sweep1.yaml
    # sweep_id = "86x4jb7r" # sweep2.yaml
    # sweep_id = "j1h5tb43" # sweep3.yaml
    # sweep_id = "k9ldb4jj" # sweep1_100.yaml
    # sweep_id = "ex1e7bbi" # sweep4.yaml, finetuning sweep
    # sweep_id = "kxqousx4" # sweep_pretrained_fine.yaml
    # api = wandb.Api() 

    # sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # if len(sweep.runs) >= 10:
    #     api.stop_sweep(sweep.id)
    #     print(f"Sweep {sweep.id} stopped after {len(sweep.runs)} runs.")

    # Start sweep agent
    wandb.agent(sweep_id, function=train, count=15, project=project)  # Run 10 experiments