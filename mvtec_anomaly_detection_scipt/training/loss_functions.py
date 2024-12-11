import torch.nn as nn

def compute_loss(original, reconstructed, anomaly_map, anomaly_gt, reconstruction_loss_fn = nn.MSELoss(), discrimination_loss_fn = nn.BCELoss()):
    # Reconstruction Loss
    reconstruction_loss = reconstruction_loss_fn(reconstructed, original)
    
    # Discrimination Loss
    discrimination_loss = discrimination_loss_fn(anomaly_map, anomaly_gt)
    
    # Weighted sum of losses
    total_loss = reconstruction_loss + 0.5 * discrimination_loss
    return total_loss