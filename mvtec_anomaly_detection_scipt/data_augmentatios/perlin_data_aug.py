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






# import cv2
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# from perlin_noise import PerlinNoise
# from scipy.ndimage import rotate, binary_erosion, binary_dilation

# # def generate_fractal_perlin_noise(shape, res, octaves=4, persistence=0.5):
# #     """
# #     Generates fractal Perlin noise.
    
# #     Parameters:
# #     - shape: Tuple, shape of the output noise.
# #     - res: Tuple, resolution of the base Perlin grid.
# #     - octaves: Number of octaves for fractal noise.
# #     - persistence: Amplitude persistence for fractal noise layers.
    
# #     Returns:
# #     - fractal_noise: Fractal Perlin noise as a numpy array.
# #     """
# #     noise = np.zeros(shape)
# #     frequency = 1
# #     amplitude = 1
# #     for _ in range(octaves):
# #         noise += amplitude * generate_perlin_noise(shape, (res[0] * frequency, res[1] * frequency))
# #         frequency *= 2
# #         amplitude *= persistence
# #     return noise


# # def generate_perlin_noise(shape, res):
# #     """
# #     Generates base Perlin noise.

# #     Parameters:
# #     - shape: Tuple, shape of the output noise.
# #     - res: Tuple, resolution of the base Perlin grid.
    
# #     Returns:
# #     - perlin_noise: Perlin noise as a numpy array.
# #     """
# #     def fade(t):
# #         return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

# #     delta = (res[0] / shape[0], res[1] / shape[1])
# #     d = (int(np.ceil(shape[0] / res[0])), int(np.ceil(shape[1] / res[1])))

# #     grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

# #     # Gradients
# #     angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
# #     gradients = np.dstack((np.cos(angles), np.sin(angles)))

# #     # Tile gradients to cover the grid
# #     g00 = gradients[0:-1, 0:-1].repeat(d[0], axis=0).repeat(d[1], axis=1)
# #     g10 = gradients[1:, 0:-1].repeat(d[0], axis=0).repeat(d[1], axis=1)
# #     g01 = gradients[0:-1, 1:].repeat(d[0], axis=0).repeat(d[1], axis=1)
# #     g11 = gradients[1:, 1:].repeat(d[0], axis=0).repeat(d[1], axis=1)

# #     # Crop to ensure consistency
# #     g00 = g00[:shape[0], :shape[1], :]
# #     g10 = g10[:shape[0], :shape[1], :]
# #     g01 = g01[:shape[0], :shape[1], :]
# #     g11 = g11[:shape[0], :shape[1], :]
# #     grid = grid[:shape[0], :shape[1], :]  # Crop grid to match

# #     # Debugging
# #     print(f"grid.shape: {grid.shape}")
# #     print(f"g00.shape: {g00.shape}, g10.shape: {g10.shape}, g01.shape: {g01.shape}, g11.shape: {g11.shape}")

# #     # Ramps
# #     n00 = np.sum(grid * g00, axis=-1)
# #     n10 = np.sum(np.dstack((grid[..., 0] - 1, grid[..., 1])) * g10, axis=-1)
# #     n01 = np.sum(np.dstack((grid[..., 0], grid[..., 1] - 1)) * g01, axis=-1)
# #     n11 = np.sum(np.dstack((grid[..., 0] - 1, grid[..., 1] - 1)) * g11, axis=-1)

# #     # Interpolation
# #     t = fade(grid)
# #     n0 = (1 - t[..., 0]) * n00 + t[..., 0] * n10
# #     n1 = (1 - t[..., 0]) * n01 + t[..., 0] * n11
# #     return np.sqrt(2) * ((1 - t[..., 1]) * n0 + t[..., 1] * n1)



# def perlin_augment(source, destination, num_patches=3, scale=50, octaves=4, beta_range=(0.1, 1.0)):
#     """
#     Applies Perlin noise augmentation to the destination image with blending inspired by DRAEM.

#     Parameters:
#     - source: Texture source image for anomaly simulation.
#     - destination: Destination image to apply the Perlin noise onto.
#     - num_patches: Number of noise patches to apply.
#     - scale: Scale of the Perlin noise.
#     - octaves: Number of octaves for the Perlin noise.
#     - beta_range: Blending opacity range for the anomalies.

#     Returns:
#     - augmented_image: The augmented destination image.
#     - mask: Binary mask indicating augmented regions.
#     """
#     augmented_image = destination.copy()
#     height, width, _ = destination.shape
#     mask = np.zeros((height, width), dtype=np.uint8)  # Initialize mask
#     noise = PerlinNoise(octaves=octaves, seed=random.randint(0, 100))

#     for _ in range(num_patches):
#         # Generate Perlin noise
#         patch_width = random.randint(30, 400)
#         patch_height = random.randint(30, 400)
#         perlin_patch = np.zeros((patch_height, patch_width))
#         for i in range(patch_height):
#             for j in range(patch_width):
#                 perlin_patch[i, j] = noise([i / scale, j / scale])
        
#         perlin_patch = cv2.normalize(perlin_patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         patch_mask = (perlin_patch > random.randint(50, 200)).astype(np.uint8)

#         # Morphological transformations for irregularity
#         patch_mask = binary_dilation(patch_mask, iterations=random.randint(1, 3)).astype(np.uint8)
#         patch_mask = binary_erosion(patch_mask, iterations=random.randint(1, 2)).astype(np.uint8)

#         # Randomly rotate the patch and mask
#         rotation_angle = random.uniform(0, 360)
#         perlin_patch = rotate(perlin_patch, angle=rotation_angle, reshape=False, mode='nearest')
#         patch_mask = rotate(patch_mask, angle=rotation_angle, reshape=False, mode='nearest')

#         # Clip the rotated patch and mask to valid values
#         perlin_patch = np.clip(perlin_patch, 0, 255)
#         patch_mask = (patch_mask > 0).astype(np.uint8)

#         # Randomly select texture region from the source image
#         src_y = random.randint(0, source.shape[0] - patch_height)
#         src_x = random.randint(0, source.shape[1] - patch_width)
#         texture_patch = source[src_y:src_y + patch_height, src_x:src_x + patch_width]

#         # Randomly select destination coordinates
#         dest_y = random.randint(0, height - patch_height)
#         dest_x = random.randint(0, width - patch_width)
        
#         # Blending the texture and Perlin noise with the destination image
#         beta = random.uniform(*beta_range)
#         for c in range(3):  # For each color channel
#             augmented_image[dest_y:dest_y + patch_height, dest_x:dest_x + patch_width, c] = (
#                 (1 - beta) * augmented_image[dest_y:dest_y + patch_height, dest_x:dest_x + patch_width, c] +
#                 beta * (patch_mask * texture_patch[:, :, c])
#             )

#         # Update the binary mask for the augmented region
#         mask[dest_y:dest_y + patch_height, dest_x:dest_x + patch_width] = patch_mask

#     return augmented_image.astype(np.uint8), mask


# if __name__ == "__main__":

#     # Load images
#     source = cv2.imread('../../datasets/mvtec/bottle/train/good/001.png')
#     print("source.shape: ",source.shape)
#     source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

#     destination = cv2.imread('../../datasets/mvtec/bottle/train/good/002.png')
#     print("destination.shape: ",destination.shape)
#     destination = cv2.cvtColor(destination, cv2.COLOR_BGR2RGB)

#     # Apply Perlin noise augmentation
#     augmented_image, mask = perlin_augment(source, destination, num_patches=random.randint(1, 7), scale=50, octaves=4)

#     # Display the augmented image and mask
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Augmented Image")
#     plt.imshow(augmented_image)
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title("Mask")
#     plt.imshow(mask, cmap='gray')
#     plt.axis('off')

#     plt.show()



# # DRAEM generates fractal Perlin noise using multiple octaves and persistence. 
# # Each octave introduces additional detail (higher-frequency noise), resulting in more realistic anomaly patterns.

# # DRAEM applies a fade function (6t^5 - 15t^4 + 10t^3) for smooth interpolation of noise between grid points, which makes anomalies more organic.

# # It allows controlling resolutions per patch to adapt the size and complexity of the anomalies.

# # It selects regions based on Perlin noise thresholds, creating anomaly masks with diverse and irregular patterns.