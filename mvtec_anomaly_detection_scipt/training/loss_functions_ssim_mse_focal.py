import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

# SSIM Loss
# class SSIM(nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = None

#     def create_window(self, window_size, channel):
#         gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * 1.5 ** 2)) for x in range(window_size)])
#         gauss = gauss / gauss.sum()
#         _1D_window = gauss.unsqueeze(1)
#         _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
#         window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
#         return window

#     def forward(self, img1, img2):
#         (_, channel, height, width) = img1.size()
#         if self.window is None or self.window.shape[1] != channel:
#             self.window = self.create_window(self.window_size, channel).to(img1.device)

#         mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
#         mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)
#         sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1.pow(2)
#         sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2.pow(2)
#         sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1 * mu2

#         c1 = 0.01 ** 2
#         c2 = 0.03 ** 2
#         ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1.pow(2) + mu2.pow(2) + c1) * (sigma1_sq + sigma2_sq + c2))

#         return 1 - ssim_map.mean() if self.size_average else 1 - ssim_map

# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()

# def create_window(window_size, channel=1):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
#     return window

# def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
#     if val_range is None:
#         if torch.max(img1) > 128:
#             max_val = 255
#         else:
#             max_val = 1

#         if torch.min(img1) < -0.5:
#             min_val = -1
#         else:
#             min_val = 0
#         l = max_val - min_val
#     else:
#         l = val_range

#     padd = window_size//2
#     (_, channel, height, width) = img1.size()
#     if window is None:
#         real_size = min(window_size, height, width)
#         window = create_window(real_size, channel=channel).to(img1.device)

#     mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

#     c1 = (0.01 * l) ** 2
#     c2 = (0.03 * l) ** 2

#     v1 = 2.0 * sigma12 + c2
#     v2 = sigma1_sq + sigma2_sq + c2
#     cs = torch.mean(v1 / v2)  # contrast sensitivity

#     ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

#     if size_average:
#         ret = ssim_map.mean()
#     else:
#         ret = ssim_map.mean(1).mean(1).mean(1)

#     if full:
#         return ret, cs
#     return ret, ssim_map


# class SSIM(nn.Module):
#     def __init__(self, window_size=11, size_average=True, val_range=None):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.val_range = val_range

#         # Assume 1 channel for SSIM
#         self.channel = 1
#         self.window = create_window(window_size).cuda()

#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()

#         if channel == self.channel and self.window.dtype == img1.dtype:
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
#             self.window = window
#             self.channel = channel

#         s_score, ssim_map = ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
#         return 1.0 - s_score



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Clamp SSIM values to [0, 1] range
    ssim_map = torch.clamp(ssim_map, 0, 1)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1  # Initialize with 1 channel
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.is_cuda == img1.is_cuda:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device)
            self.window = window
            self.channel = channel

        ssim_value = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return 1.0 - ssim_value

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, smooth=1e-4):
        super(FocalLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Compute Focal Loss for segmentation tasks.

        Parameters:
        - logits: Predicted logits of shape [N, C, H, W].
        - targets: Ground truth of shape [N, H, W] (integer class indices).

        Returns:
        - Scalar focal loss value.
        """
        # Ensure `targets` is integer and remove any channel dimension
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)  # From [N, 1, H, W] to [N, H, W]
        targets = targets.long()

        # Debug: Check unique values in targets
        # print(f"Unique values in targets: {torch.unique(targets)}")

        # Compute probabilities from logits
        probs = torch.softmax(logits, dim=1)  # Shape: [N, C, H, W]

        # Validate the range of targets
        num_classes = logits.shape[1]
        targets = torch.clamp(targets, 0, num_classes - 1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Compute pt (probability of the true class)
        pt = (probs * targets_one_hot).sum(dim=1) + self.smooth  # Shape: [N, H, W]

        # Compute log(pt)
        log_pt = torch.log(pt)

        # Compute focal loss
        focal_loss = -(1 - pt) * log_pt  # Shape: [N, H, W]

        return focal_loss.mean()




class ReconstructionLoss(torch.nn.Module):
    def __init__(self):
        """
        Reconstruction Loss combining MSE and SSIM with fixed weights.
        """
        super(ReconstructionLoss, self).__init__()
        self.ssim = SSIM(size_average=True)

    def forward(self, original, reconstructed):
        """
        Compute the reconstruction loss.

        Parameters:
        - original: Ground truth image.
        - reconstructed: Reconstructed image.

        Returns:
        - Combined loss (MSE + SSIM).
        """
        mse_loss = F.mse_loss(reconstructed, original)
        print("mse_loss: ", mse_loss)
        ssim_loss = self.ssim(reconstructed, original)
        print("ssim_loss: ", ssim_loss)
        return mse_loss + ssim_loss



# Combined Loss Function
def compute_loss(reconstructed, original, anomaly_map, anomaly_gt, reconstruction_loss_fn, discrimination_loss_fn):
    """
    Compute the combined loss for reconstruction and anomaly map prediction.
    Parameters:
    - reconstructed: Reconstructed images.
    - original: Ground truth images.
    - anomaly_map: Predicted anomaly map.
    - anomaly_gt: Ground truth anomaly map.
    - reconstruction_loss_fn: Reconstruction loss function (e.g., MSE + SSIM).
    - discrimination_loss_fn: Discrimination loss function (e.g., Focal Loss).
    Returns:
    - Total combined loss.
    """
    rec_loss = reconstruction_loss_fn(original, reconstructed)
    print("rec_loss: ", rec_loss)
    disc_loss = discrimination_loss_fn(anomaly_map, anomaly_gt)
    print("disc_loss: ", disc_loss)
    return rec_loss + disc_loss