import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

from cnn_class import CNN, pretrained_model
from utils import *


def main(run_purpose: str = "train", model_type: str = "customCNN"):

    # Define dataset paths
    train_dir = "../inaturalist_12K/train"
    test_dir = "../inaturalist_12K/val"

    learning_rate = 6e-4
    batch_size = 64
    max_epochs = 10
    use_augmentation = True

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    num_workers = 16 if torch.cuda.is_available() else 4 if torch.backends.mps.is_available() else 1

    train_loader, val_loader, test_loader = dataset_split(train_dir, test_dir, batch_size=batch_size, num_workers=num_workers, augmentation=use_augmentation)

    # Model instantiation with flexible hyperparameters
    # num_filters = [256, 128, 64, 32, 16]
    num_filters = [32, 64, 128, 256, 512]
    kernel_size = [3, 3, 3, 3, 3]
    num_dense = [512]
    dropout_prob = 0.12
    use_dropout = True
    use_batchnorm = True
    activation = get_activation("ReLU")

    if model_type == "customCNN":
        model = CNN(num_filters=num_filters, num_dense=num_dense, kernel_size=kernel_size, use_batchnorm=use_batchnorm, use_dropout=use_dropout, dropout_prob=dropout_prob, activation=activation)
    elif model_type == "resnet50":
        model = pretrained_model(model_type=model_type)

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss() # Loss function

    if run_purpose == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=7e-4) # Optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        print(f"Training Parameters:")
        print(f"Learning Rate: {learning_rate}, num_filters: {num_filters}, kernel_size: {kernel_size}, num_dense: {num_dense}, batch_size: {batch_size}, dropout: {dropout_prob if use_dropout else None}, batchnorm: {use_batchnorm}")
        train_loop(train_loader, val_loader, model, loss_fn, optimizer, scheduler=scheduler, device=device, max_epochs=max_epochs, patience_stop=5)
    else:
        model.load_state_dict(torch.load(f'../best_model_{model_type}.pth', weights_only=True, map_location=device))
        class_names = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]
        # test_loop(test_loader, model, loss_fn, device, class_names, save_confusion_matrix=False, wandb_log=False)
        obtain_sample_predictions(model, test_dir, device, class_names, wandb_log = True, show_plot = False)
    
if __name__=="__main__":
    # mp.set_start_method('spawn')
    run_purpose = "test"
    model_type = "customCNN"
    main(run_purpose)