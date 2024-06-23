import os
import shutil
from PIL import Image

# Specify the source directory where your images are stored
source_dir = './segmentation_output'

# Specify the target directory where images smaller than 18x18 should be moved
target_dir = './too_small_images'

# Make sure the target directory exists, create if it doesn't
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Loop through all files in the source directory
for filename in os.listdir(source_dir):
    # Construct full file path
    file_path = os.path.join(source_dir, filename)
    try:
        # Open the image file
        with Image.open(file_path) as img:
            # Check if both dimensions of the image are smaller than 18 pixels
            if img.width < 18 or img.height < 18:
                # Construct target file path
                target_path = os.path.join(target_dir, filename)
                # Move the file to the target directory
                shutil.move(file_path, target_path)
                print(f"Moved {filename} to {target_dir}")
    except IOError:
        # Handle the case where the file opened is not an image
        print(f"Failed to open {filename}. It might not be an image.")
