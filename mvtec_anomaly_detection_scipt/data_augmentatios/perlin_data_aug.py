import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

def perlin_augment(source, destination, num_patches=3, scale=50, octaves=4, beta_range=(0.1, 1.0)):
    """
    Applies Perlin noise augmentation to the destination image with blending inspired by DRAEM.

    Parameters:
    - source: Texture source image for anomaly simulation.
    - destination: Destination image to apply the Perlin noise onto.
    - num_patches: Number of noise patches to apply.
    - scale: Scale of the Perlin noise.
    - octaves: Number of octaves for the Perlin noise.
    - beta_range: Blending opacity range for the anomalies.

    Returns:
    - augmented_image: The augmented destination image.
    - mask: Binary mask indicating augmented regions.
    """
    augmented_image = destination.copy()
    height, width, _ = destination.shape
    mask = np.zeros((height, width), dtype=np.uint8)  # Initialize mask
    noise = PerlinNoise(octaves=octaves, seed=random.randint(0, 100))

    for _ in range(num_patches):
        # Generate Perlin noise
        patch_width = random.randint(30, 100)
        patch_height = random.randint(30, 100)
        perlin_patch = np.zeros((patch_height, patch_width))
        for i in range(patch_height):
            for j in range(patch_width):
                perlin_patch[i, j] = noise([i / scale, j / scale])
        
        perlin_patch = cv2.normalize(perlin_patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        patch_mask = (perlin_patch > random.randint(50, 200)).astype(np.uint8)

        # Randomly select texture region from the source image
        src_y = random.randint(0, source.shape[0] - patch_height)
        src_x = random.randint(0, source.shape[1] - patch_width)
        texture_patch = source[src_y:src_y + patch_height, src_x:src_x + patch_width]

        # Blend the texture patch with the destination image
        dest_y = random.randint(0, height - patch_height)
        dest_x = random.randint(0, width - patch_width)
        beta = random.uniform(*beta_range)

        augmented_image[dest_y:dest_y + patch_height, dest_x:dest_x + patch_width] = \
            (1 - beta) * destination[dest_y:dest_y + patch_height, dest_x:dest_x + patch_width] + \
            beta * (patch_mask[..., None] * texture_patch)

        # Update the binary mask
        mask[dest_y:dest_y + patch_height, dest_x:dest_x + patch_width] = patch_mask

    return augmented_image.astype(np.uint8), mask




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
