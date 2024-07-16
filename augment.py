import numpy as np
import cv2
print(cv2.__version__)
try:
    import imgaug as ia
    from imgaug import augmenters as iaa
    print("imgaug imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")

import cv2
import os
from glob import glob
import shutil

print("Hello World")
# Define the augmentation sequences
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Geometric transformations applied to the image
geo_seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    sometimes(iaa.Crop(percent=(0, 0.1))),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)

# Color transformations applied only to the image
color_seq = iaa.Sequential([
    sometimes(iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2)
], random_order=True)

# Function to apply augmentation to an image and save results
def augment_and_save(image, output_img_dir, img_prefix, count=5):
    for i in range(count):
        # Apply the geometric augmentation sequence
        deterministic_geo_seq = geo_seq.to_deterministic()

        # Apply the geometric augmenter to the image
        augmented_image = deterministic_geo_seq(image=image)
        
        # Apply the color transformations only to the image
        augmented_image = color_seq(image=augmented_image)
        
        # Save augmented images in the respective directory
        img_filename = os.path.join(output_img_dir, f"{img_prefix}_aug_{i}.png")
        cv2.imwrite(img_filename, augmented_image)

# Function to process images from one folder
def process_folder(input_img_dir, output_img_dir, count=5):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    
    img_files = sorted(glob(os.path.join(input_img_dir, "*.png")))

    for img_file in img_files:
        img_prefix = os.path.splitext(os.path.basename(img_file))[0]
        image = cv2.imread(img_file)
        
        # Save the original image
        original_img_filename = os.path.join(output_img_dir, f"{img_prefix}.png")
        shutil.copy(img_file, original_img_filename)
        
        # Augment and save the image
        augment_and_save(image, output_img_dir, img_prefix, count)

# Example usage
input_img_directory = 'removed_background_breasts'
output_img_directory = 'augmented_removed_background_breasts'

process_folder(input_img_directory, output_img_directory)
