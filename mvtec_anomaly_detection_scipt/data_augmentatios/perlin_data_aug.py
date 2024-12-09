import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

def perlin_augment(source, destination, num_patches=1, scale=50, octaves=4):
    """
    Applies Perlin noise augmentation to the destination image.
    
    Parameters:
    - source: Source image (numpy array) to generate Perlin noise patches.
    - destination: Destination image (numpy array) to apply the Perlin noise onto.
    - num_patches: Number of noise patches to apply to the image.
    - scale: Scale of the Perlin noise.
    - octaves: Number of octaves for the Perlin noise.

    Returns:
    - augmented_image: The augmented destination image.
    - mask: Binary mask indicating augmented regions (1 for augmentation, 0 otherwise).
    """
    augmented_image = destination.copy()
    height, width, _ = destination.shape
    mask = np.zeros((height, width), dtype=np.uint8)  # Initialize mask

    # Initialize Perlin noise generator
    noise = PerlinNoise(octaves=octaves, seed=random.randint(0, 100))

    for _ in range(num_patches):
        # Randomly decide the patch size and location
        patch_width = random.randint(30, 100)
        patch_height = random.randint(30, 100)
        
        # Generate Perlin noise for the patch
        perlin_patch = np.zeros((patch_height, patch_width))

        for i in range(patch_height):
            for j in range(patch_width):
                # Generate Perlin noise at a scaled position
                perlin_patch[i, j] = noise([i / scale, j / scale])
        
        # Normalize Perlin noise to [0, 255]
        perlin_patch = cv2.normalize(perlin_patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Randomly select the paste location on the destination image
        paste_x = random.randint(0, width - patch_width)
        paste_y = random.randint(0, height - patch_height)

        # Ensure dimensions of the patch match the slicing area
        if paste_y + patch_height > augmented_image.shape[0]:
            patch_height = augmented_image.shape[0] - paste_y
        if paste_x + patch_width > augmented_image.shape[1]:
            patch_width = augmented_image.shape[1] - paste_x

        # Crop the Perlin noise patch if needed
        perlin_patch = perlin_patch[:patch_height, :patch_width]

        # Apply the Perlin noise to the destination image
        augmented_image[paste_y:paste_y + patch_height, paste_x:paste_x + patch_width] = \
            cv2.addWeighted(augmented_image[paste_y:paste_y + patch_height, paste_x:paste_x + patch_width], 0.7, 
                            cv2.cvtColor(perlin_patch, cv2.COLOR_GRAY2RGB), 0.3, 0)

        # Update the mask to mark the augmented region
        mask[paste_y:paste_y + patch_height, paste_x:paste_x + patch_width] = 1

    return augmented_image, mask



if __name__ == "__main__":

    # Load images
    source = cv2.imread('../../datasets/mvtec/bottle/train/good/001.png')
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    destination = cv2.imread('../../datasets/mvtec/bottle/train/good/002.png')
    destination = cv2.cvtColor(destination, cv2.COLOR_BGR2RGB)

    # Apply Perlin noise augmentation
    augmented_image, mask = perlin_augment(source, destination, num_patches=3, scale=50, octaves=4)

    # Display the augmented image and mask
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Augmented Image")
    plt.imshow(augmented_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.show()
