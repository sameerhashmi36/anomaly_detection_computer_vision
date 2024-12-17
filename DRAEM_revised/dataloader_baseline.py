import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import glob
import imgaug.augmenters as iaa  # Traditional augmentations

class MvtecBaseline(Dataset):
    def __init__(self, root_dir, resize_shape=(256, 256)):
        """
        Args:
            root_dir (string): Directory with all the images.
            resize_shape (tuple): Target image dimensions (H, W).
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")) +
                                  glob.glob(os.path.join(root_dir, "*.jpg")) +
                                  glob.glob(os.path.join(root_dir, "*.jpeg")))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

        # Augmentations
        self.augmenters = iaa.Sequential([
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))  # Rotations for augmentations
        ])
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def augment_image(self, image):
      """
      Applies baseline augmentation to the image.

      Args:
          image (numpy array): Input image.

      Returns:
          tuple: Augmented image, dummy mask, and anomaly presence flag.
      """
      # Convert to uint8 for imgaug compatibility
      image_uint8 = (image * 255).astype(np.uint8)

      # Resize the image for augmentation compatibility
      anomaly_source_img = cv2.resize(image_uint8, (self.resize_shape[1], self.resize_shape[0]))

      # Apply augmentations
      augmented_image_uint8 = self.augmenters(image=anomaly_source_img)

      # Convert augmented image back to float32
      augmented_image = augmented_image_uint8.astype(np.float32) / 255.0

      # Create a dummy mask (all zeros)
      mask = np.zeros(self.resize_shape, dtype=np.float32)
      mask = np.expand_dims(mask, axis=2)  # Add channel dimension

      augmented_image = mask * augmented_image + (1 - mask) * (image.astype(np.float32) / 255.0)
      has_anomaly = 1.0 if np.sum(mask) > 0 else 0.0

      return augmented_image, mask, np.array([has_anomaly], dtype=np.float32)


    def transform_image(self, image_path):
        """
        Loads and preprocesses an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple: Original image, augmented image, dummy mask, and anomaly flag.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
        
        # Random rotation augmentation
        if torch.rand(1).item() > 0.7:
            image = self.rot(image=image)

        # Apply augmentation
        augmented_image, dummy_mask, has_anomaly = self.augment_image(image)

        # Transpose images and mask for PyTorch compatibility
        image = np.transpose(image.astype(np.float32), (2, 0, 1))  # Normalize and rearrange to (C, H, W)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        dummy_mask = np.transpose(dummy_mask, (2, 0, 1))

        return image, augmented_image, dummy_mask, has_anomaly

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the image.

        Returns:
            dict: Contains the original image, augmented image, dummy mask, and anomaly flag.
        """
        image_path = self.image_paths[idx]
        image, augmented_image, dummy_mask, has_anomaly = self.transform_image(image_path)

        return {
            'image': image,
            'augmented_image': augmented_image,
            'anomaly_mask': dummy_mask,
            'has_anomaly': has_anomaly
        }
