from fastai.vision import *
import cv2
import matplotlib.pyplot as plt
from utils.code.aux_func import seg2bbox, predict
import os
import uuid
from datetime import datetime

def DeepNipple(image_path, alg_mode, show, save, save_path):
    '''
    :param image_path: input image absolute path
    :param alg_mode: seg/bbox
    :param show: boolean, whether to show the image or not
    :param save: boolean, whether to save the mask or not
    :param save_path: folder to save the mask/bounding boxes
    :return: segmentation mask / bounding boxes
    '''

    # Load pytorch model
    learner_path = 'utils/models/base-model/'
    learner = load_learner(learner_path)

    image, mask = predict(image_path, learner)
    print(image.shape, mask.shape)

    # Ensure the save_path is a directory
    if save and not os.path.exists(save_path):
        os.makedirs(save_path)

    # Extract the base name of the image file
    image_name = os.path.basename(image_path)
    image_base_name = os.path.splitext(image_name)[0]  # Get the file name without extension

    if alg_mode == 'seg':
        output = mask
        # if show:
        #     plt.subplot(121)
        #     plt.imshow(image)
        #     plt.title('Original image')
        #     plt.axis('off')
        #     plt.subplot(122)
        #     plt.imshow(image)
        #     plt.axis('off')
        #     plt.imshow(mask[:, :, 1], alpha=0.6, interpolation='bilinear', cmap='magma')
        #     plt.axis('off')
        #     plt.imshow(mask[:, :, 2], alpha=0.6, interpolation='bilinear', cmap='afmhot')
        #     plt.axis('off')
        #     plt.title('NAC segmentation')
        #     plt.show()

        if save:
            mask_save_path = os.path.join(save_path, f'{image_base_name}.png')
            mask_to_save = (mask * 255).astype(np.uint8)
            cv2.imwrite(mask_save_path, mask_to_save)

    else:
        coords = seg2bbox(mask)
        output = coords

        if show:
            for coor in coords:
                y1, y2, x1, x2 = coor[0], coor[1], coor[2], coor[3]
                cv2.rectangle(image, (x1, y1), (x2, y2), (36, 255, 12), 2, -1)

            # Convert image to RGB before showing with matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
            plt.show()

        if save:
            for idx, coor in enumerate(coords):
                y1, y2, x1, x2 = coor[0], coor[1], coor[2], coor[3]

                # Ensure coordinates are within image boundaries
                if x1 >= image.shape[1] or x2 >= image.shape[1] or y1 >= image.shape[0] or y2 >= image.shape[0]:
                    print(f"Warning: Invalid bbox coordinates for {image_base_name}_bbox_{idx}.png")
                    continue  # Skip saving this bbox image

                bbox_image = image[y1:y2, x1:x2]

                # Check if bbox_image is empty or None
                if bbox_image.size == 0:
                    print(f"Warning: Empty bbox image for {image_base_name}_bbox_{idx}.png")
                    continue  # Skip saving this bbox image

                # Convert bbox_image to BGR before saving with OpenCV
                bbox_image_bgr = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR)
                bbox_save_path = os.path.join(save_path, f'{image_base_name}_bbox_{idx}.png')
                cv2.imwrite(bbox_save_path, bbox_image_bgr)

    return output
