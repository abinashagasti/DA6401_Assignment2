import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
# from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from lightning_class import LitCNN  # assuming your model is in lit_cnn.py
from utils import *

def main():
    # ------------------ Hyperparameters ------------------ #
    batch_size = 64
    learning_rate = 6e-4
    max_epochs = 20
    val_split = 0.2
    img_size = 224

    # ------------------ Transforms ------------------ #
    train_dir = "../inaturalist_12K/train"
    test_dir = "../inaturalist_12K/val"

    train_loader, val_loader, test_loader = dataset_split(train_dir, test_dir, batch_size=batch_size, augmentation=True)

    # ------------------ Model ------------------ #
    model = LitCNN(
        num_filters=[32, 64, 128, 256, 512],
        kernel_size=[3, 3, 3, 3, 3],
        num_dense=[512],
        lr=learning_rate,
        dropout_prob=0.12,
        use_batchnorm=True,
        use_dropout=True,
        padding=1,
        img_size=img_size
    )

    # ------------------ Callbacks ------------------ #
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="best_model")
    early_stopping = EarlyStopping(monitor="val_acc", mode="max", patience=5)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ------------------ Trainer ------------------ #
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=10,
    )

    # ------------------ Training ------------------ #
    trainer.fit(model, train_loader, val_loader)

    # ------------------ Testing (Optional) ------------------ #
    trainer.test(model, dataloaders=test_loader)

if __name__ == "__main__":
    main()