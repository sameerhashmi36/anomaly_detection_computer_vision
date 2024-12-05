import os
import sys
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_augmentatios.cutpaste_dataaugmentation_technique import cut_paste_augment

class DynamicMVTecDataset(Dataset):
    def __init__(self, good_image_dir, transform=None, include_scar=True):
        """
        Custom dataset for MVTec, dynamically generating synthetic anomalies.

        Parameters:
        - good_image_dir: Path to the directory containing good images.
        - transform: Torchvision transformation to apply to the images and masks.
        - include_scar: Whether to include scar-type augmentations in synthetic data.
        """
        self.good_image_dir = good_image_dir
        self.transform = transform
        self.include_scar = include_scar

        # Load all good images
        self.good_images = []
        for img_file in os.listdir(good_image_dir):
            img_path = os.path.join(good_image_dir, img_file)
            if os.path.isfile(img_path):
                self.good_images.append(img_path)

        if len(self.good_images) == 0:
            raise ValueError("No images found in the good_image_dir.")

    def __len__(self):
        return len(self.good_images) * 2  # Good images + synthetic images

    def __getitem__(self, idx):
        try:
            if idx < len(self.good_images):
                # Good images
                img_path = self.good_images[idx]
                image = Image.open(img_path).convert('RGB')
                mask = Image.new('L', image.size, 0)  # Dummy mask (all zeros)

                if self.transform:
                    image = self.transform(image)
                    mask = self.transform(mask)

                label = 0  # Label for good images

                # Debug: Log successful good image return
                # print(f"[DEBUG] Good Image - Index: {idx}, Returned: (image, mask, label)")
                return image, mask, label
            else:
                # Synthetic images
                source_path = random.choice(self.good_images)
                destination_path = random.choice(self.good_images)

                source_image = Image.open(source_path).convert('RGB')
                destination_image = Image.open(destination_path).convert('RGB')

                # Convert to NumPy arrays
                source_np = np.array(source_image)
                destination_np = np.array(destination_image)

                # Randomize the number of patches (1 to 7)
                num_patches = random.randint(1, 10)

                # Apply CutPaste augmentation
                augmented_image_np, mask_np = cut_paste_augment(
                    source_np, destination_np, num_patches=num_patches, include_scar=self.include_scar
                )

                # Convert back to PIL images
                augmented_image = Image.fromarray(augmented_image_np)
                mask = Image.fromarray((mask_np * 255).astype('uint8'))  # Convert binary mask to 0-255

                if self.transform:
                    augmented_image = self.transform(augmented_image)
                    mask = self.transform(mask)

                label = 1  # Label for synthetic anomaly images

                # Debug: Log successful synthetic image return
                # print(f"[DEBUG] Synthetic Image - Index: {idx}, Num Patches: {num_patches}, Returned: (image, mask, label)")
                return augmented_image, mask, label
            
        except Exception as e:
            # Debug: Log the error
            print(f"[ERROR] Error at Index {idx}: {e}")
            return None
