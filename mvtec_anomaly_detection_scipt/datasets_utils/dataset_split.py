from torch.utils.data import random_split

def split_dataset(dataset, train_percentage = 0.75):
    """
    Splits a dataset into train, validation subsets.

    Parameters:
    - dataset: List of dataset samples (e.g., file paths or indices).
    - train_percentage: float, fraction of data for training.

    Returns:
    - train_data: Subset for training.
    - val_data: Subset for validation.
    """
    train_size = int(train_percentage * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

