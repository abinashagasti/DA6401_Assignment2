import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm 
import torch.nn.functional as F
import wandb
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random
from torchvision.io import read_image

class EnsurePortrait:
    """
    Custom transform to ensure image is in portrait orientation.

    If the image is in landscape mode (i.e., width > height),
    it rotates the image 90 degrees counterclockwise to convert it to portrait mode.
    If the image is already in portrait or square shape, it is returned unchanged.

    Methods:
    --------
    __call__(img: Image.Image) -> Image.Image
        Applies the rotation transformation if needed.
    """
    def __call__(self, img: Image.Image):
        if img.width > img.height:
            return img.rotate(90, expand=True)
        return img

# Now wrap them to apply different transforms
class TransformedDataset(torch.utils.data.Dataset):
    """
    A wrapper dataset that applies a given transform to an existing dataset or subset.

    Useful when using the same dataset object but applying different transforms 
    (e.g., augmentations during training but not during validation or testing).

    Parameters:
    -----------
    subset : torch.utils.data.Dataset
        The dataset (or a Subset) whose items we want to transform.
    transform : callable
        A transform to apply on the images (e.g., torchvision.transforms.Compose).
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # Return transformed image and label
        img, label = self.subset[index]
        return self.transform(img), label

    def __len__(self):
        return len(self.subset)

def dataset_split(train_dir, test_dir, batch_size=64, num_workers=4, val_split=0.2, augmentation=False, img_size = 224):
    """
    Splits the dataset into training, validation, and test sets, with optional data augmentation for training.

    Parameters:
    -----------
    train_dir : str
        Directory path containing training data in subfolders per class.
    test_dir : str
        Directory path containing test data in subfolders per class.
    batch_size : int
        Number of samples per batch in the DataLoader.
    num_workers : int
        Number of worker threads for data loading.
    val_split : float
        Fraction of training data to be used for validation (default: 0.2).
    augmentation : bool
        If True, apply data augmentation to training set.
    img_size : int
        Desired size (height and width) to resize input images.

    Returns:
    --------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    """
    base_transforms = [
        # EnsurePortrait(),
        transforms.Resize((img_size, img_size)),  # Resize all images to standard size
    ]

    # Apply augmentation if enabled
    if augmentation:
        transform_train = transforms.Compose(base_transforms + [
            transforms.RandomHorizontalFlip(0.5), # Randomly flip images
            transforms.RandomRotation(10), # Rotate images ±10 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Slight color changes
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standardize across 3 channels      
        ])
    else: # If augmentation is disabled, only resize and standardize the data
        transform_train = transforms.Compose(base_transforms + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standardize across 3 channels      
        ])

    # Only resizing and standardization of data for validation and test sets
    transform_val = transforms.Compose(base_transforms + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Set pixel values to 0.5 mean and std across 3 channels      
        ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_val)

    # Load the full train dataset once
    full_dataset = datasets.ImageFolder(root=train_dir)

    # Define the split sizes
    val_size = int(val_split * len(full_dataset))  
    train_size = len(full_dataset) - val_size  

    # Get shuffled indices
    indices = torch.randperm(len(full_dataset)).tolist()  
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    # Define dataset subsets using the same full_dataset
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    # Apply different transforms to train and val
    train_dataset = TransformedDataset(train_subset, transform=transform_train)
    val_dataset = TransformedDataset(val_subset, transform=transform_val)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

def train_loop(train_loader, val_loader, model, loss_fn, optimizer, scheduler, device=torch.device('cpu'), max_epochs=5, patience_stop=10, wandb_log=False):
    """
    Trains a PyTorch model with early stopping and optional learning rate scheduling and W&B logging.

    Parameters:
    -----------
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    model : nn.Module
        The model to be trained
    loss_fn : torch.nn loss function
        Loss function to use (e.g., CrossEntropyLoss)
    optimizer : torch.optim
        Optimizer to update model weights
    scheduler : torch.optim.lr_scheduler (optional)
        Learning rate scheduler that steps using validation loss
    device : torch.device
        Device to run training on (default: 'cpu')
    max_epochs : int
        Maximum number of training epochs
    patience_stop : int
        Early stopping threshold — number of epochs to wait after last improvement
    wandb_log : bool
        Whether to log metrics to Weights & Biases
    """

    # Initialize best values for early stopping
    best_val_loss = float('inf')
    best_val_accuracy = 0
    epochs_without_improvement = 0

    # Training for specified number of epochs
    for epoch in tqdm(range(max_epochs), desc="Training Progress", unit="epoch"):
        model.train()  # Set model to training mode
        
        total_train_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        # Training loop
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad() # Clear previous gradients
            pred = model(X) # Forward pass
            loss = loss_fn(pred, y) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update parameters

            # Track loss & accuracy
            total_train_loss += loss.item() * X.size(0)  # Sum batch loss
            correct_train += (pred.argmax(1) == y).sum().item()  # Count correct predictions
            total_train_samples += X.size(0)  # Count total samples

        # Compute train loss & accuracy
        avg_train_loss = total_train_loss / total_train_samples
        train_accuracy = correct_train / total_train_samples * 100

        # Validation loop
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0.0
        correct_val = 0
        total_val_samples = 0

        with torch.no_grad():  # Disable gradient computation for validation
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                
                pred_val = model(X_val)
                loss_val = loss_fn(pred_val, y_val)

                # Track loss & accuracy
                total_val_loss += loss_val.item() * X_val.size(0)
                correct_val += (pred_val.argmax(1) == y_val).sum().item()
                total_val_samples += X_val.size(0)

        # Compute validation loss & accuracy
        avg_val_loss = total_val_loss / total_val_samples
        val_accuracy = correct_val / total_val_samples * 100

        # Log to wandb if enabled
        if wandb_log:
                wandb.log({
                    "epoch": epoch+1,
                    "training_loss": avg_train_loss,
                    "training_accuracy": train_accuracy,
                    "validation_loss": avg_val_loss,
                    "validation_accuracy": val_accuracy
                })

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{max_epochs}] → Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%\n")
        
        # Step the learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "../best_model.pth")
        else:
            epochs_without_improvement += 1

        # Track best accuracy for reporting
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        # Early stopping condition
        if epochs_without_improvement >= patience_stop:
            print(f"Early stopping triggered after {patience_stop} epochs without improvement.")   
            break     

    print(f"Training complete. Best validation loss = {best_val_loss:.4f}; best validation accuracy = {best_val_accuracy:.2f}%")

def test_loop(test_loader, model, loss_fn, device=torch.device('cpu'), class_names=None, save_confusion_matrix=True):
    """
    Evaluates the model on the test dataset and optionally plots a confusion matrix.

    Parameters:
    -----------
    test_loader : DataLoader
        DataLoader containing test data.
    model : torch.nn.Module
        Trained model to evaluate.
    loss_fn : torch.nn.Module
        Loss function used for evaluation (e.g., nn.CrossEntropyLoss).
    device : torch.device
        Device to perform computation on (default: CPU).
    class_names : list of str, optional
        List of class names for labeling the confusion matrix axes.
    save_confusion_matrix : bool
        Whether to display and save the confusion matrix (default: True).

    Returns:
    --------
    all_labels : list
        Ground truth labels from the test set.
    all_preds : list
        Model predictions on the test set.
    """
    model.eval()  # Set model to evaluation mode
    total_test_loss = 0.0
    correct_test = 0
    total_test_samples = 0

    # Store predicted and true labels
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for testing
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            pred_test = model(X_test)
            loss_test = loss_fn(pred_test, y_test)

            # Track loss & accuracy
            total_test_loss += loss_test.item() * X_test.size(0)
            correct_test += (pred_test.argmax(1) == y_test).sum().item()
            total_test_samples += X_test.size(0)

            # Collect predictions and labels
            all_preds.extend(pred_test.argmax(1).cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())

    # Combine all batches
    # all_preds = torch.cat(all_preds)
    # all_labels = torch.cat(all_labels)

    # Compute test loss & accuracy
    avg_test_loss = total_test_loss / total_test_samples
    test_accuracy = correct_test / total_test_samples * 100

    # Print test summary
    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%\n")

    # Compute and plot confusion matrix
    if save_confusion_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names if class_names else range(len(cm)),
                    yticklabels=class_names if class_names else range(len(cm)))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('../confusion_matrix.png')
        plt.show()

    return all_labels, all_preds

def imshow(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Unnormalize and display image."""
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean  # Unnormalize
    img = np.clip(img, 0, 1)
    return img

def obtain_sample_predictions(model, test_dir, device, class_names=None, img_size=224, wandb_log=True, show_plot=False):
    """
    Samples 3 images from each of the first 10 classes in test_dir,
    predicts using the given model, and plots a 3x10 grid showing predictions.

    Correct predictions are in green, wrong in red.
    Logs to W&B if wandb_log=True.
    """

    model.eval() # Switch model to evaluation mode

    # Image Preprocessing
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Get Class Folders
    class_folders = sorted([f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))])
    if class_names is None:
        class_names = class_folders

    fig, axes = plt.subplots(3, 10, figsize=(14, 5))
    correct = 0
    total = 0

    for col, class_name in enumerate(class_folders[:10]):  # 10 classes → 10 columns
        class_path = os.path.join(test_dir, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(image_files) < 3:
            continue
        sampled_files = random.sample(image_files, 3)

        for row in range(3):
            ax = axes[row, col]
            img_file = sampled_files[row]
            img_path = os.path.join(class_path, img_file)

            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform_val(image).unsqueeze(0).to(device)

            # Model prediction
            with torch.no_grad():
                output = model(image_tensor)
                pred_idx = output.argmax(1).item()
                pred_label = class_names[pred_idx]

            is_correct = (pred_label == class_name)
            if is_correct:
                correct += 1
            total += 1

            # Plotting
            raw_img = plt.imread(img_path)
            ax.imshow(raw_img)

            # True label on first row only (in blue)
            if row == 0:
                ax.set_title(f"True class: {class_name}", fontsize=9, color="blue")

            # Predicted label in green/red based on correctness
            color = "green" if is_correct else "red"
            ax.set_xlabel(f"Predicted: {pred_label}", fontsize=8, color=color)

            ax.set_xticks([])
            ax.set_yticks([])

    acc = 100 * correct / total
    print(f"\nSample Prediction Accuracy: {acc:.2f}% ({correct}/{total})")
    plt.tight_layout()

    # Wandb logging if enabled
    if wandb_log:
        wandb.init(project="DA6401_Assignment_2", name="Sample Prediction Grid")
        wandb.log({"Predictions Grid (3x10)": wandb.Image(fig)})
        wandb.finish()

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def get_activation(activation: str):
    """
    Returns the corresponding PyTorch functional activation function 
    based on the input string.

    Parameters:
    -----------
    activation : str
        Name of the activation function. Supported: 'ReLU', 'GELU', 'SiLU', 'Mish'

    Returns:
    --------
    Callable:
        A PyTorch activation function (from torch.nn.functional)

    Raises:
    -------
    ValueError:
        If the provided activation name is not recognized.
    """
    activation = activation.lower()
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "silu":
        return F.silu
    elif activation == "mish":
        return F.mish
    else:
        raise ValueError(f"Unsupported activation function: {activation}. Choose from ['ReLU', 'GELU', 'SiLU', 'Mish']")
    
def get_optimizer(optimizer_name: str, model, learning_rate: float, weight_decay: float):
    """
    Returns a PyTorch optimizer based on the given name.

    Parameters:
    -----------
    optimizer_name : str
        Name of the optimizer. Supported: 'adam', 'sgd'
    model : torch.nn.Module
        The model whose parameters will be optimized.
    learning_rate : float
        The learning rate for the optimizer.
    weight_decay : float
        The weight decay (L2 regularization) factor.

    Returns:
    --------
    torch.optim.Optimizer
        The corresponding optimizer instance.

    Raises:
    -------
    ValueError:
        If an unsupported optimizer name is provided.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from ['adam', 'sgd']")
