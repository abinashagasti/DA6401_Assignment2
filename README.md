# DA6401 Assignment 2

This repository contains the code implementation for **Assignment 2** of the **DA6401** course. It is organized into two parts — `partA` and `partB` — corresponding to the respective sections of the assignment.

---

## Part A

### Description

Part A implements and trains a **custom CNN model** from scratch for the given dataset. It includes all necessary utilities, model definition, and experiment scripts.

### ▶️ How to Run

You can customize the training run using the following optional arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-wp`, `--wandb_project` | `str` | `DA6401_Assignment_2` | Project name for tracking experiments on Weights & Biases |
| `-we`, `--wandb_entity` | `str` | `ee20d201-indian-institute-of-technology-madras` | Your wandb entity/team name |
| `-e`, `--epochs` | `int` | `20` | Number of training epochs |
| `-b`, `--batch_size` | `int` | `64` | Batch size used for training |
| `-is`, `--img_size` | `int` | `224` | Size to which all input images are resized |
| `-f`, `--num_filters` | `list[int]` | `[32, 64, 128, 256, 512]` | Number of filters in each convolutional layer |
| `-k`, `--kernel_size` | `list[int]` | `[3, 3, 3, 3, 3]` | Kernel sizes for each convolutional layer |
| `-da`, `--use_augmentation` | `flag` | `True` | Disable data augmentation by including this flag |
| `-nd`, `--num_dense` | `list[int]` | `[512]` | Number of neurons in each fully connected (dense) layer |
| `-dr`, `--use_dropout` | `flag` | `True` | Disable dropout by including this flag |
| `-bn`, `--use_batchnorm` | `flag` | `True` | Disable batch normalization by including this flag |
| `-drprob`, `--dropout_prob` | `float` | `0.12` | Dropout probability value |
| `-pad`, `--padding` | `int` or `None` | `1` | Padding used in each CNN layer (`None`, `1`, or `2`) |
| `-o`, `--optimizer` | `str` | `"adam"` | Optimizer to use (`sgd`, `adam`) |
| `-lr`, `--learning_rate` | `float` | `0.0006` | Learning rate for optimizer |
| `-w_d`, `--weight_decay` | `float` | `0.0007` | Weight decay (L2 regularization) |
| `-a`, `--activation` | `str` | `"ReLU"` | Activation function (`ReLU`, `GELU`, `SiLU`, `Mish`) |
| `-wbl`, `--wandb_login` | `flag` | `False` | Use this flag to enable logging to Weights & Biases |

> **Note:** The `-d` or `--data_directory` argument is **required** and must be provided.

```bash
python3 partA/train.py -d "/path/to/dataset" [optional arguments]
```

Example:

```bash
python3 partA/train.py -d "../../inaturalist_12K" -e 25 -b 64 -a GELU -drprob 0.1 -f [16,32,64,128,256] -nd [512] -wbl
```

### Key Files

- `train.py` – Main training script  
- `train.ipynb` – Jupyter notebook version of the training pipeline  
- `utils.py` – Helper functions for data loading, evaluation, etc.  
- `CNN_class.py` – Contains the custom CNN model definition  
- `sweep_train.py` – Script for running wandb hyperparameter sweeps  
- `*.yaml` – Sweep configuration files for wandb experiments
- `alternate_main.py`, `main.py` - Used for initial runs and obtaining test data visualisation
- `lightning_main.py`, `lightning_class.py` - Experiments using pytorch-lightning

---

## Part B

### Description

Part B uses **pretrained models** (e.g., ResNet, VGG) for transfer learning. It follows a similar structure and workflow to Part A.

### ▶️ How to Run

You can customize the training run using the following optional arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-wp`, `--wandb_project` | `str` | `DA6401_Assignment_2` | Project name for tracking experiments on Weights & Biases |
| `-we`, `--wandb_entity` | `str` | `ee20d201-indian-institute-of-technology-madras` | Your wandb entity/team name |
| `-e`, `--epochs` | `int` | `20` | Number of training epochs |
| `-b`, `--batch_size` | `int` | `32` | Batch size used for training |
| `-m`, `--model_type` | `str` | `EfficientNetV2` | Model architecture (`ResNet18`, `ResNet50`, `GoogLeNet`, `VGG`, `InceptionV3`, `EfficientNetV2`, `VisionTransformer`) |
| `-k`, `--trainable_layers` | `int` | `1` | Number of trainable layers (for fine-tuning) |
| `-da`, `--use_augmentation` | `flag` | `True` | Use this flag to disable data augmentation |
| `-pad`, `--padding` | `int` or `None` | `1` | Padding used in each CNN layer (`None`, `1`, or `2`) |
| `-o`, `--optimizer` | `str` | `sgd` | Optimizer to use (`sgd`, `adam`) |
| `-lr`, `--learning_rate` | `float` | `0.000281` | Learning rate for optimizer |
| `-w_d`, `--weight_decay` | `float` | `0.000827` | Weight decay (L2 regularization) |
| `-wbl`, `--wandb_login` | `flag` | `False` | Use this flag to enable logging to Weights & Biases |

**Note:** The `-d` or `--data_directory` argument is **required** and must be provided when running the script.

```bash
python3 partB/train.py -d "/path/to/dataset" [optional arguments]
```

Example:

```bash
python3 partB/train.py -d "../../inaturalist_12K" -e 30 -b 64 -m ResNet18 -wbl
```

Uses the same command-line interface as Part A

### Key Files

- `train.py` – Main training script  
- `train.ipynb` – Jupyter notebook version of the training pipeline  
- `utils.py` – Shared utility functions (same as in Part A)  
- `pretrained_models.py` – Contains loading and customization of pretrained models  
- `sweep_pretrained.py` – Script for wandb sweeps with pretrained models  
- `*.yaml` – Configuration files for sweep experiments
- `alternate_main.py`, `main.py` - Used for initial runs and obtaining test data visualisation

---

## WANDB Report

Report Link - https://wandb.ai/ee20d201-indian-institute-of-technology-madras/DA6401_Assignment_2/reports/DA6401-Assignment-2--VmlldzoxMjE0MzUzNg?accessToken=lw6miyrg7ahn2p6f3r5vvx3eszhmagf66t0page4x8sj9okaeh4seif31f6zdcc5