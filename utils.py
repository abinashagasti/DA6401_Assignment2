import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 


def dataset_split(train_dir, test_dir, batch_size=64, num_workers=4, val_split=0.2):
    # Define transformations (you can modify this as needed)
    transform1 = transforms.Compose([
    transforms.Resize((224, 224)),  # Uniform sizing of all images
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images randomly
    transforms.RandomRotation(20),           # Rotate images by ±20°
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Slight color changes
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Set pixel values to 0.5 mean and std across 3 channels
])
    
    transform2 = transforms.Compose([
    transforms.Resize((224, 224)),  # Uniform sizing of all images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Set pixel values to 0.5 mean and std across 3 channels
])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform1)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform2)

    # Define the split sizes
    val_size = int(val_split * len(train_dataset))  # 20% for validation
    train_size = len(train_dataset) - val_size  # 80% for training

    # Split the dataset
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def train_loop(train_loader, val_loader, model, loss_fn, optimizer, scheduler, device=torch.device('cpu'), max_epochs=5):
    for epoch in tqdm(range(max_epochs), desc="Training Progress", unit="epoch"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.train()  # Set model to training mode
        
        total_train_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        # Training loop
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
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

        if scheduler is not None:
            scheduler.step(avg_val_loss)
            print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{max_epochs}] → Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%\n")
        

def test_loop(test_loader, model, loss_fn, device=torch.device('cpu')):
    model.eval()  # Set model to evaluation mode
    total_test_loss = 0.0
    correct_test = 0
    total_test_samples = 0

    with torch.no_grad():  # Disable gradient computation for testing
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            pred_test = model(X_test)
            loss_test = loss_fn(pred_test, y_test)

            # Track loss & accuracy
            total_test_loss += loss_test.item() * X_test.size(0)  # Sum batch loss
            correct_test += (pred_test.argmax(1) == y_test).sum().item()  # Count correct predictions
            total_test_samples += X_test.size(0)  # Count total samples

    # Compute test loss & accuracy
    avg_test_loss = total_test_loss / total_test_samples
    test_accuracy = correct_test / total_test_samples * 100

    # Print test summary
    print(f"Test Results → Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%\n")