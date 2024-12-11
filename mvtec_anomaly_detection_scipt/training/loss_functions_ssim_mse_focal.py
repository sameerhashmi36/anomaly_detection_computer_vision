import torch
import torch.nn as nn
import torch.nn.functional as F

def ssim_loss(img1, img2, window_size=11, size_average=True):
    """
    Compute Structural Similarity (SSIM) Loss between two images.
    """
    C1 = 0.01**2
    C2 = 0.03**2

    # Gaussian kernel
    def gaussian(window_size, sigma):
        gauss = torch.tensor([torch.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    (_, channel, height, width) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1 - ssim_map.mean() if size_average else 1 - ssim_map

# Combined Reconstruction Loss
def reconstruction_loss(original, reconstructed):
    mse_loss = F.mse_loss(reconstructed, original)
    ssim = ssim_loss(reconstructed, original)
    return mse_loss + 0.5 * ssim


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced pixel-level classification.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt is the probability of the true class
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
        

# compute all losses
def compute_loss(original, reconstructed, anomaly_map, anomaly_gt, 
                 reconstruction_loss_fn=reconstruction_loss, 
                 discrimination_loss_fn=FocalLoss()):
    """
    Compute the total loss combining reconstruction and discrimination losses.
    """
    # Reconstruction Loss (SSIM + MSE)
    reconstruction_loss = reconstruction_loss_fn(original, reconstructed)

    # Discrimination Loss (Focal Loss)
    discrimination_loss = discrimination_loss_fn(anomaly_map, anomaly_gt)

    # Weighted sum of losses
    total_loss = reconstruction_loss + 0.5 * discrimination_loss
    return total_loss
