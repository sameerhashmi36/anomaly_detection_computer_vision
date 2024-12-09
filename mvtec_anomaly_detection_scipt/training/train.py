import os
import torch
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_and_validate(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device, 
    log_dir, save_path, start_epoch=0, best_val_loss=float('inf')):

    # Ensure directories exist
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
        # Training phase
        model.train()
        running_train_loss = 0.0
        for images, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            optimizer.zero_grad()
            decoded, _ = model(images)  # Get only the pixel-level output for segmentation
            decoded = torch.sigmoid(decoded)  # Ensure output is in [0, 1] range

            # Compute loss
            loss = criterion(decoded, masks)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_images, val_masks, val_labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False):
                val_images, val_masks = val_images.to(device), val_masks.to(device)

                # Forward pass
                val_decoded, _ = model(val_images)
                val_decoded = torch.sigmoid(val_decoded)  # Sigmoid activation for BCE

                # Compute validation loss
                val_loss = criterion(val_decoded, val_masks)
                running_val_loss += val_loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Log epoch summary to console and file
        log_message = f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        print(log_message)
        logger.info(log_message)
        
        # Save models with better validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(save_path, f"model_epoch_{epoch+1}_valloss_{avg_val_loss:.4f}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, model_save_path)
            log_message = f"Saved model: {model_save_path}"
            print(log_message)
            logger.info(log_message)

    plot_loss(train_losses, val_losses, log_dir)


def plot_loss(train_losses, val_losses, log_dir):
    """
    Plots and saves the training and validation loss curves.

    Parameters:
    - train_losses: List of training losses per epoch.
    - val_losses: List of validation losses per epoch.
    - log_dir: Directory to save the loss plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Val loss")
    plt.legend()
    plt.grid()
    
    # Save the plot in the log directory
    loss_plot_path = os.path.join(log_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()  # Close the plot to avoid memory issues
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
