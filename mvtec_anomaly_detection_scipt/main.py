import os
import torch
import torch.nn as nn
from model.conv_autoencoder import ConvAutoencoderWithClassification
from datasets_utils.dataset_split import split_dataset
from datasets_utils.dataset_class import DynamicMVTecDataset
from datasets_utils.data_loader import get_loaders
import torchvision.transforms as T
from torch.utils.data import DataLoader
from training.train import train_and_validate, load_checkpoint

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# infos
info = {
    "class":"bottle",
    "augmentation": "cutpaste",
    "checkpoint_path": "experiment_04_bottle_cutpaste_conv_autoencoder",
    "log_path": "experiment_04_bottle_cutpaste_conv_autoencoder"
        }

# Data transform
transform = transforms = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

# Dataset setup
dataset = DynamicMVTecDataset(good_image_dir=f"../datasets/mvtec/{info['class']}/train/good/", transform=transform, augmentations=[f"{info['augmentation']}"])

# Dataset setup
train_data, val_data = split_dataset(dataset=dataset)

# get loader
train_loader, val_loader = get_loaders(train_data=train_data, val_data=val_data, batch_size=32, num_workers=4) # 32, 4

#Model setup
# model = ResNetEncDec().to(device)
model = ConvAutoencoderWithClassification().to(device)

# Criterion and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Checkpoint path
checkpoint_path = f"experiments/{info['checkpoint_path']}/checkpoints/model_epoch_49_valloss_0.0613.pth"

# Resume training if checkpoint exists
start_epoch = 0
best_val_loss = float('inf')
if os.path.exists(checkpoint_path):
    model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path, device)


# Train and validate
num_epochs = 100
train_and_validate(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device,
    log_dir=f"experiments/{info['log_path']}/logs",
    save_path=f"experiments/{info['checkpoint_path']}/checkpoints",
    start_epoch=start_epoch,
    best_val_loss=best_val_loss
)