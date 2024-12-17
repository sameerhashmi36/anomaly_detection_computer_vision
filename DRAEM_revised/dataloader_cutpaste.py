import os
import numpy as np
from torch.utils.data import Dataset
import random
import torch
import cv2
import glob
import imgaug.augmenters as iaa  # Traditional augmentations
from data_augmentations.cutpaste_aug import cut_paste_augment

class MvtecCutPaste(Dataset):
    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        # print("root_dir: ", self.root_dir)
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))
        # print("image_paths: ", self.image_paths)

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*.png"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def rand_augmenter(self):
        """
        Randomly selects and applies a subset of augmentations.
        """
        aug_indices = np.random.choice(len(self.augmenters), 3, replace=False)
        return iaa.Sequential([self.augmenters[i] for i in aug_indices])

    # def augment_image(self, image, source_image):
    #     """
    #     Applies CutPaste augmentation to the image.

    #     Args:
    #         image (numpy array): The destination image.
    #         source_image (numpy array): The source image for patches.

    #     Returns:
    #         tuple: Augmented image, anomaly mask, and anomaly presence flag.
    #     """
    #     # Normalize images
    #     image = np.array(image).astype(np.float32) / 255.0
    #     source_image = cv2.resize(source_image, (self.resize_shape[1], self.resize_shape[0]))
    #     # source_image = self.rand_augmenter()(image=source_image)

    #     # Perform CutPaste
    #     augmented_image, mask = cut_paste_augment(source_image, image, num_patches=random.randint(1, 7), include_scar=True)

    #     mask = np.expand_dims(mask, axis=-1)

    #     if torch.rand(1).item() > 0.5:
    #         return image, np.zeros_like(mask, dtype=np.float32), np.array([0.0], dtype=np.float32)
    #     else:
    #         mask = mask.astype(np.float32)
    #         augmented_image = augmented_image.astype(np.float32)
    #         augmented_image = mask * augmented_image + (1 - mask) * image
    #         has_anomaly = 1.0 if np.sum(mask) > 0 else 0.0
    #         return augmented_image, mask, np.array([has_anomaly], dtype=np.float32)
        
    def augment_image(self, image, source_image):

        aug = self.rand_augmenter()

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        anomaly_source_img = cv2.resize(source_image, (self.resize_shape[1], self.resize_shape[0]))
        anomaly_img_augmented = aug(image=anomaly_source_img)
        augmented_image, mask = cut_paste_augment(anomaly_img_augmented, image, num_patches=random.randint(1, 7), include_scar=True)
        no_anomaly = torch.rand(1).item()
        if no_anomaly > 0.5:
            return image.astype(np.float32), np.zeros_like(mask, dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32) / 255.0
            mask = mask.astype(np.float32)
            augmented_image = mask * augmented_image + (1 - mask) * image
            has_anomaly = 1.0 if np.sum(mask) > 0 else 0.0
            return augmented_image, mask, np.array([has_anomaly], dtype=np.float32) 

    def transform_image(self, image_path, source_image_path):
        """
        Transforms and augments an input image.

        Args:
            image_path (str): Path to the destination image.
            source_image_path (str): Path to the source image.

        Returns:
            tuple: Processed destination image, augmented image, anomaly mask, and anomaly flag.
        """
        # Load and preprocess images
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
        image = np.array(image).astype(np.float32) / 255.0

        source_image = cv2.imread(source_image_path)
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        

        # Random rotation augmentation
        if torch.rand(1).item() > 0.7:
            image = self.rot(image=image)

        # Apply CutPaste augmentation
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, source_image)

        # Transpose images and mask for PyTorch compatibility
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        augmented_image = np.transpose(augmented_image, (2, 0, 1)).astype(np.float32)
        anomaly_mask = np.expand_dims(anomaly_mask, axis=0).astype(np.float32)  # Add channel dimension

        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the image.

        Returns:
            dict: Contains the destination image, augmented image, anomaly mask, and anomaly flag.
        """
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        source_idx = torch.randint(0, len(self.image_paths), (1,)).item()

        # Ensure source and destination images are not the same
        while source_idx == idx:
            source_idx = torch.randint(0, len(self.image_paths), (1,)).item()

        # Process the images
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(
            self.image_paths[idx],
            self.image_paths[source_idx]
        )

        return {
            'image': image,
            'augmented_image': augmented_image,
            'anomaly_mask': anomaly_mask,
            'has_anomaly': has_anomaly,
            'idx': idx
        }
