import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from cnn_class import CNN, pretrained_model
from utils import *

def main(use_pretrained: bool = False):
    # Define dataset paths
    train_dir = "../inaturalist_12K/train"
    test_dir = "../inaturalist_12K/val"

    learning_rate = 1e-4
    batch_size = 32
    max_epochs = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    num_workers = 16 if torch.cuda.is_available() else 4 if torch.backends.mps.is_available() else 1

    train_loader, val_loader, test_loader = dataset_split(train_dir, test_dir, batch_size=batch_size, num_workers=num_workers, augmentation=True)

    if use_pretrained:
        # Load a pretrained ResNet18 model
        model_type = "VGG" # Options = ["ResNet18", "ResNet50", "VGG", "EfficientNetV2", "VisionTransformer"]
        model = pretrained_model(model_type, k=1)

    else:
        # Define custom CNN
        num_filters = [32, 64, 128, 256, 512]
        kernel_size = [3, 3, 3, 3, 3]
        num_dense = [1024]
        dropout_prob = 0.3
        use_dropout = True
        use_batchnorm = True
        model = CNN(num_filters=num_filters, num_dense=num_dense, kernel_size=kernel_size,
                    use_batchnorm=use_batchnorm, use_dropout=use_dropout, dropout_prob=dropout_prob)

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print(f"Training with {model_type if use_pretrained else 'Custom CNN'}")
    if not use_pretrained:
        print(f"Training Parameters:")
        print(f"Learning Rate: {learning_rate}, num_filters: {num_filters}, kernel_size: {kernel_size}, num_dense: {num_dense}, batch_size: {batch_size}, dropout: {dropout_prob if use_dropout else None}, batchnorm: {use_batchnorm}")
    train_loop(train_loader, val_loader, model, loss_fn, optimizer, scheduler=scheduler, device=device, max_epochs=max_epochs, patience_stop=5)
    class_names = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]
    test_loop(test_loader, model, loss_fn, device, class_names)

if __name__ == "__main__":
    use_pretrained = False  # Toggle this to switch between CNN and ResNet
    main(use_pretrained)