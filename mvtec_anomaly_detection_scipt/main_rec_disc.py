import os
import torch
import torch.nn as nn
from model.resnet18_enc_dec_with_discriminator import ResNetEncDecWithDiscrimination
from datasets_utils.dataset_split import split_dataset
from datasets_utils.dataset_class import DynamicMVTecDataset
from datasets_utils.data_loader import get_loaders
import torchvision.transforms as T
from torch.utils.data import DataLoader
from training.train_rec_disc import train_and_validate, load_checkpoint
# from training.loss_functions_ssim_mse_focal import ReconstructionLoss, FocalLoss


# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# infos
info = {
    "class":"hazelnut",
    "augmentation": "cutpaste",
    "checkpoint_path": "experiment_03_hazelnut_cutpaste_mse_bce",
    "log_path": "experiment_03_hazelnut_cutpaste_mse_bce",
    "experiment_dir": "experiments_rec_disc_hazelnut"
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
train_loader, val_loader = get_loaders(train_data=train_data, val_data=val_data, batch_size=16, num_workers=4) # 32, 4

#Model setup
model = ResNetEncDecWithDiscrimination().to(device)

# Loss functions
# rec_loss_fn = ReconstructionLoss()
rec_loss_fn = nn.MSELoss()
disc_loss_fn = nn.BCELoss()

# Criterion and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Checkpoint path
checkpoint_path = f"{info['experiment_dir']}/{info['checkpoint_path']}/checkpoints/model_epoch_80_valloss_0.0683.pth"

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
    rec_loss_fn=rec_loss_fn,
    disc_loss_fn=disc_loss_fn,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device,
    log_dir=f"{info['experiment_dir']}/{info['log_path']}/logs",
    save_path=f"{info['experiment_dir']}/{info['checkpoint_path']}/checkpoints",
    start_epoch=start_epoch,
    best_val_loss=best_val_loss
)