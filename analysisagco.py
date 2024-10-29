import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def count_images_in_agco_dataset(dataset_path):
    # Dictionary to store the image counts for each class (folder)
    image_counts = {}

    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print("Dataset path does not exist.")
        return image_counts

    # Loop over each class (folder) in the dataset
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        
        # Check if the path is a directory (class folder)
        if os.path.isdir(class_path):
            # Count the number of image files in the class folder
            num_images = len([file for file in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, file))])
            image_counts[class_name] = num_images

    return image_counts

def plot_image_counts(image_counts):
    if not image_counts:
        print("No image counts to plot.")
        return

    # Sort the data by class name for consistent plotting
    sorted_classes = sorted(image_counts.keys())
    sorted_counts = [image_counts[class_name] for class_name in sorted_classes]

    # Plot the bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(sorted_classes, sorted_counts, label="Image Counts", alpha=0.7)
    
    # Smooth the trend line using cubic spline interpolation
    x = np.arange(len(sorted_classes))
    y = np.array(sorted_counts)
    
    if len(x) > 1:  # Ensure there are enough points for interpolation
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=3)  # Cubic spline (k=3)
        y_smooth = spline(x_smooth)
        
        # Plot the smooth trend line
        plt.plot(x_smooth, y_smooth, color='orange', linestyle='-', linewidth=2, label="Trend Line")
    
    # Add numbers on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')
    
    # Labels and title
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Image Counts for Each Class in the AGCO Dataset')
    plt.xticks(ticks=x, labels=sorted_classes, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    dataset_path = "/Users/raj/Downloads/IMAGE/dataset_splited"  # path to the dataset
    image_counts = count_images_in_agco_dataset(dataset_path)
    
    # Display the image counts as a chart with bars, numbers, and a trend line
    plot_image_counts(image_counts)
