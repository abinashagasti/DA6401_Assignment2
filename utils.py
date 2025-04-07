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
from PIL import Image

# Custom transform to ensure portrait orientation
class EnsurePortrait:
    def __call__(self, img: Image.Image):
        if img.width > img.height:
            return img.rotate(90, expand=True)
        return img

# Now wrap them to apply different transforms
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        return self.transform(img), label

    def __len__(self):
        return len(self.subset)

def dataset_split(train_dir, test_dir, batch_size=64, num_workers=4, val_split=0.2, augmentation=False):
    # Define transformations 
    base_transforms = [
        # EnsurePortrait(),
        transforms.Resize((224, 224)),  # Resize to portrait shape (HxW)
    ]

    if augmentation:
        transform_train = transforms.Compose(base_transforms + [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Slight color changes
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Set pixel values to 0.5 mean and std across 3 channels      
        ])
    else:
        transform_train = transforms.Compose(base_transforms + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Set pixel values to 0.5 mean and std across 3 channels      
        ])

    transform_val = transforms.Compose(base_transforms + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Set pixel values to 0.5 mean and std across 3 channels      
        ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_val)

    # Load the full dataset once (without transforms initially)
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

    # Load full dataset
    # full_dataset = datasets.ImageFolder(root=train_dir)  

    # # Define the split sizes
    # val_size = int(val_split * len(full_dataset))  
    # train_size = len(full_dataset) - val_size  

    # # Get shuffled indices
    # indices = torch.randperm(len(full_dataset)).tolist()  
    # train_indices, val_indices = indices[:train_size], indices[train_size:]

    # # Create subsets with different transforms
    # train_subset = Subset(datasets.ImageFolder(root=train_dir, transform=transform_train), train_indices)
    # val_subset = Subset(datasets.ImageFolder(root=train_dir, transform=transform_val), val_indices)

    # # Create DataLoaders
    # train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

def train_loop(train_loader, val_loader, model, loss_fn, optimizer, scheduler, device=torch.device('cpu'), max_epochs=5, patience_stop=10, wandb_log=False):

    best_val_loss = float('inf')
    best_val_accuracy = 0
    epochs_without_improvement = 0

    for epoch in tqdm(range(max_epochs), desc="Training Progress", unit="epoch"):
        model.train()  # Set model to training mode
        
        total_train_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        # Training loop
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            # Forward pass
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()

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

        if wandb_log:
                wandb.log({
                    "epoch": epoch,
                    "training_loss": avg_train_loss,
                    "training_accuracy": train_accuracy,
                    "validation_loss": avg_val_loss,
                    "validation_accuracy": val_accuracy
                })

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{max_epochs}] → Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%\n")
        
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "../best_model.pth")
        else:
            epochs_without_improvement += 1

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        if epochs_without_improvement >= patience_stop:
            print(f"Early stopping triggered after {patience_stop} epochs without improvement.")   
            break     

    print(f"Training complete. Best validation loss = {best_val_loss:.4f}; best validation accuracy = {best_val_accuracy:.2f}%")

def test_loop(test_loader, model, loss_fn, device=torch.device('cpu'), class_names=None, save_confusion_matrix=True):
    model.eval()  # Set model to evaluation mode
    total_test_loss = 0.0
    correct_test = 0
    total_test_samples = 0

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

def get_activation(activation: str):
    if activation == "ReLU":
        return F.relu
    elif activation == "GELU":
        return F.gelu
    elif activation == "SiLU":
        return F.silu
    elif activation == "Mish":
        return F.mish
    
def get_optimizer(optimizer_name: str, model, learning_rate: float, weight_decay: float):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9, weight_decay=weight_decay)
