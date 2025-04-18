import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import multiprocessing

from pretrained_models import pretrained_model
from utils import *

def main(args):
    user = args.wandb_entity
    project = args.wandb_project
    display_name = "test_run"

    if args.wandb_login:
        wandb.init(entity=user, project=project, name=display_name) # Initialize wandb experiment. 
        # config = wandb.config
    
    # Define dataset paths
    train_dir = args.data_directory+'/train'
    test_dir = args.data_directory+'/val'

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    max_epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    num_workers = max(1, multiprocessing.cpu_count() - 2) if torch.cuda.is_available() else 4 if torch.backends.mps.is_available() else 1

    use_augmentation = args.use_augmentation
    train_loader, val_loader, test_loader = dataset_split(train_dir, test_dir, batch_size=batch_size, num_workers=num_workers, augmentation=use_augmentation)

    # Load a pretrained ResNet18 model
    model_type = args.model_type # Options = ["ResNet18", "ResNet50", "GoogLeNet", "VGG", "InceptionV3", "EfficientNetV2", "VisionTransformer"]
    model = pretrained_model(model_type, k=1)

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, learning_rate=learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"Training with {model_type}")
    train_loop(train_loader, val_loader, model, loss_fn, optimizer, scheduler=scheduler, device=device, max_epochs=max_epochs, patience_stop=10, wandb_log=wandb_login)
    # class_names = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]
    # test_loop(test_loader, model, loss_fn, device, class_names)

    if args.wandb_login:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp","--wandb_project",default="DA6401_Assignment_2", help="Project name used to track experiments in Weights & Biases dashboard",type=str)
    parser.add_argument("-we","--wandb_entity",default="ee20d201-indian-institute-of-technology-madras", help="Wandb Entity used to track experiments in the Weights & Biases dashboard",type=str)
    parser.add_argument("-d","--data_directory",required=True,type=str,help="Path containing inaturalist dataset.")
    parser.add_argument("-e","--epochs",default=20,help="Number of epochs to train neural network",type=int)
    parser.add_argument("-b","--batch_size",default=32,help="Batch size used to train neural network",type=int)
    parser.add_argument("-m","--model_type",default="EfficientNetV2",choices=["ResNet18", "ResNet50", "GoogLeNet", "VGG", "InceptionV3", "EfficientNetV2", "VisionTransformer"],help="Model Type")
    parser.add_argument("-k","--trainable_layers",default=1,help="Number of trainable layers.",type=int)
    parser.add_argument("-da","--use_augmentation",default=True,action="store_false",help="Use data augmentation for training.")
    parser.add_argument("-pad","--padding",default=1,choices=[None, 1, 2],help="Padding used in each layer.")
    parser.add_argument("-o","--optimizer",default="adam",choices=["sgd", "adam"], help="optimizer to train neural network",type=str)
    parser.add_argument("-lr","--learning_rate",default=0.00028, help="Learning rate used to optimize model parameters",type=float)
    parser.add_argument("-w_d","--weight_decay",default=0.0007, help="Weight decay used by optimizers",type=float)
    parser.add_argument("-wbl","--wandb_login",default=False,action="store_true", help="Login data onto wandb.ai")
    args = parser.parse_args()
    main(args)