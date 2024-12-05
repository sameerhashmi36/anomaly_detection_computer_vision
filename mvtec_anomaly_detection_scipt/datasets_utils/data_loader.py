import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

def get_loaders(train_data, val_data, batch_size=32, num_workers=4):

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
