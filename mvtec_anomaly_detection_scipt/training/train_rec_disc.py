import os
import torch
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from training.loss_functions import compute_loss

def train_and_validate(
    model, train_loader, val_loader, rec_loss_fn, disc_loss_fn, optimizer, num_epochs, device, 
    log_dir, save_path, start_epoch=0, best_val_loss=float('inf')):

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # Set up logging
    log_file = os.path.join(log_dir, "training_log.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a"
    )
    logger = logging.getLogger()

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_train_loss = 0.0

        # Training Loop
        for images, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            reconstructed, anomaly_map = model(images)

            # Compute loss
            loss = compute_loss(
                original=images,
                reconstructed=reconstructed,
                anomaly_map=anomaly_map,
                anomaly_gt=masks,
                reconstruction_loss_fn=rec_loss_fn,
                discrimination_loss_fn=disc_loss_fn
            )
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_images, val_masks, val_labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False):
                val_images, val_masks, val_labels = val_images.to(device), val_masks.to(device), val_labels.to(device)
                val_reconstructed, val_anomaly_map = model(val_images)

                val_loss = compute_loss(
                    original=val_images,
                    reconstructed=val_reconstructed,
                    anomaly_map=val_anomaly_map,
                    anomaly_gt=val_masks,
                    reconstruction_loss_fn=rec_loss_fn,
                    discrimination_loss_fn=disc_loss_fn
                )
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Log progress
        log_message = f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        print(log_message)
        logger.info(log_message)

        # Save model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(save_path, f"model_epoch_{epoch+1}_valloss_{avg_val_loss:.4f}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, model_save_path)
            print(f"Saved model: {model_save_path}")
            logger.info(f"Saved model: {model_save_path}")

    plot_loss(train_losses, val_losses, log_dir)


def plot_loss(train_losses, val_losses, log_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()

    loss_plot_path = os.path.join(log_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved at: {loss_plot_path}")





def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Loads a model and optimizer state from a checkpoint.

    Parameters:
    - model: The model to load the state into.
    - optimizer: The optimizer to load the state into.
    - checkpoint_path: Path to the checkpoint file.
    - device: The device to map the checkpoint to.

    Returns:
    - model: Model with loaded weights.
    - optimizer: Optimizer with loaded state.
    - start_epoch: The epoch to resume training from.
    - best_val_loss: The best validation loss from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['val_loss']

    print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
    return model, optimizer, start_epoch, best_val_loss
