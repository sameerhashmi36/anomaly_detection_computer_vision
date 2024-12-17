import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def cut_paste_augment(source, destination, num_patches=1, include_scar=True):
    """
    Applies CutPaste augmentation (with optional scars) to the destination image.

    Parameters:
    - source: Source image (numpy array) to cut patches from.
    - destination: Destination image (numpy array) to paste patches onto.
    - num_patches: Number of patches to apply (can be a mix of CutPaste and CutPaste-Scar).
    - include_scar: Whether to include scar-type augmentations.

    Returns:
    - augmented_image: The augmented destination image.
    - mask: Binary mask indicating augmented regions (1 for augmentation, 0 otherwise).
    """
    augmented_image = destination.copy()
    height, width, _ = destination.shape
    mask = np.zeros((height, width), dtype=np.uint8)  # Initialize mask

    for _ in range(num_patches):
        # Randomly decide between CutPaste or Scar
        use_scar = include_scar and random.choice([True, False])

        if use_scar:
            # Scar variant: Thin rectangle
            patch_width = random.randint(2, min(40, width))
            patch_height = random.randint(10, min(200, height))
        else:
            # Regular CutPaste: Rectangular patch
            patch_area_ratio = random.uniform(0.02, 0.15)
            aspect_ratio = random.uniform(0.3, 3.3)
            patch_width = int(np.sqrt(patch_area_ratio * width * height * aspect_ratio))
            patch_height = int(np.sqrt(patch_area_ratio * width * height / aspect_ratio))

            # Clamp patch dimensions to image dimensions
            patch_width = min(patch_width, width)
            patch_height = min(patch_height, height)

        # Ensure patch fits within source image
        patch_width = min(patch_width, source.shape[1])
        patch_height = min(patch_height, source.shape[0])

        # Skip invalid patches
        if patch_width <= 0 or patch_height <= 0:
            print(f"Skipping invalid patch: Width={patch_width}, Height={patch_height}")
            continue

        # Randomly select the patch location from the source image
        source_top_left_x = random.randint(0, source.shape[1] - patch_width)
        source_top_left_y = random.randint(0, source.shape[0] - patch_height)
        patch = source[source_top_left_y:source_top_left_y + patch_height, source_top_left_x:source_top_left_x + patch_width]

        # Apply transformations to the patch
        if use_scar:
            # Generate a random angle between 0 and 360 degrees
            angle = random.uniform(0, 360)
            
            # Get the center of the patch
            center = (patch_width // 2, patch_height // 2)
            
            # Create the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
            
            # Apply the rotation
            patch = cv2.warpAffine(patch, rotation_matrix, (patch_width, patch_height))
        else:
            patch = cv2.flip(patch, 1) if random.random() > 0.5 else patch

        # Recalculate patch dimensions after transformations
        patch_height, patch_width = patch.shape[:2]

        # Ensure valid paste dimensions
        if patch_width > width or patch_height > height:
            print(f"Skipping paste: Patch exceeds destination dimensions ({patch_width}, {patch_height})")
            continue

        # Randomly select the paste location on the destination image
        paste_x = random.randint(0, width - patch_width)
        paste_y = random.randint(0, height - patch_height)

        # Paste the patch onto the destination image
        augmented_image[paste_y:paste_y + patch_height, paste_x:paste_x + patch_width] = patch

        # Update the mask to mark the augmented region
        mask[paste_y:paste_y + patch_height, paste_x:paste_x + patch_width] = 1

    return augmented_image, np.expand_dims(mask, axis=2)




if __name__ == "__main__":
    # Load an image
    source = cv2.imread('/home/sameerhashmi36/Documents/AVS7/anomaly_detection_computer_vision/datasets/mvtec/bottle/train/good/001.png')
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    destination = cv2.imread('/home/sameerhashmi36/Documents/AVS7/anomaly_detection_computer_vision/datasets/mvtec/bottle/train/good/002.png')
    destination = cv2.cvtColor(destination, cv2.COLOR_BGR2RGB)

    # Apply CutPaste augmentation
    augmented_image, mask = cut_paste_augment(source, destination, num_patches=5, include_scar=True)

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
