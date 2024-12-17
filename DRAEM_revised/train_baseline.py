import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from dataloader_baseline import MvtecBaseline  # Import the baseline dataset class
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM

def get_lr(optimizer):
    """
    Returns the current learning rate of the optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    """
    Initializes the weights of the model for Conv and BatchNorm layers.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_logs(log_path, logs):
    """
    Saves the training logs to a text file.

    Args:
        log_path (str): Path where the log file is saved.
        logs (list of dict): List of log dictionaries with epoch and loss details.
    """
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "training_logs.txt")
    with open(log_file, "a") as f:
        for log in logs:
            f.write(f"Epoch {log['epoch']} - "
                    f"L2 Loss: {log['l2_loss']:.4f}, "
                    f"SSIM Loss: {log['ssim_loss']:.4f}, "
                    f"Segment Loss: {log['segment_loss']:.4f}\n")

def train_on_device(obj_names, args):
    """
    Trains the model on the specified objects.

    Args:
        obj_names (list): List of objects to train on.
        args (argparse.Namespace): Training arguments and configurations.
    """
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:
        # Set up run name and logging
        run_name = f'DRAEM_baseline_{args.lr}_{args.epochs}_bs{args.bs}_{obj_name}_'
        print(f"Training on {obj_name} with run name {run_name}")

        # Initialize models
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3).cuda()
        model.apply(weights_init)

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2).cuda()
        model_seg.apply(weights_init)

        # Optimizer and learning rate scheduler
        optimizer = optim.Adam([
            {"params": model.parameters(), "lr": args.lr},
            {"params": model_seg.parameters(), "lr": args.lr}
        ])
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, [int(args.epochs * 0.8), int(args.epochs * 0.9)], gamma=0.2
        )

        # Loss functions
        loss_l2 = torch.nn.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        # Dataset and DataLoader
        dataset = MvtecBaseline(
            root_dir=os.path.join(args.data_path, obj_name, "train/good"),
            resize_shape=[256, 256]
        )
        dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=16)

        logs = []  # To store logs for saving to a file
        for epoch in range(args.epochs):
            print(f"Epoch: {epoch + 1}/{args.epochs}")
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{args.epochs}")

            epoch_l2_loss = 0
            epoch_ssim_loss = 0
            epoch_segment_loss = 0

            for i_batch, sample_batched in progress_bar:
                # Load data to GPU
                gray_batch = sample_batched["image"].cuda() / 255.0
                aug_gray_batch = sample_batched["augmented_image"].cuda() / 255.0
                anomaly_mask = sample_batched["anomaly_mask"].cuda()

                # Forward pass
                gray_rec = model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)
                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                # Compute losses
                l2_loss = loss_l2(gray_rec, gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = l2_loss + ssim_loss + segment_loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update epoch losses
                epoch_l2_loss += l2_loss.item()
                epoch_ssim_loss += ssim_loss.item()
                epoch_segment_loss += segment_loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    "L2 Loss": l2_loss.item(),
                    "SSIM Loss": ssim_loss.item(),
                    "Segment Loss": segment_loss.item()
                })

            # Average epoch losses
            epoch_l2_loss /= len(dataloader)
            epoch_ssim_loss /= len(dataloader)
            epoch_segment_loss /= len(dataloader)

            logs.append({
                "epoch": epoch + 1,
                "l2_loss": epoch_l2_loss,
                "ssim_loss": epoch_ssim_loss,
                "segment_loss": epoch_segment_loss
            })

            scheduler.step()

            # Save model checkpoints
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"{run_name}.pckl"))
            torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, f"{run_name}_seg.pckl"))

        # Save logs at the end of training
        save_logs(args.log_path, logs)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)

    args = parser.parse_args()

    obj_list = [
        'capsule', 'bottle', 'carpet', 'leather', 'pill', 'transistor',
        'tile', 'cable', 'zipper', 'toothbrush', 'metal_nut', 'hazelnut',
        'screw', 'grid', 'wood'
    ]

    picked_classes = obj_list if args.obj_id == -1 else [obj_list[args.obj_id]]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)
