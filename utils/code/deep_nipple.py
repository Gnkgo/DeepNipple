from fastai.vision import *
import cv2
import matplotlib.pyplot as plt
from utils.code.aux_func import seg2bbox, predict
import os
import uuid
import json
import numpy as np

def DeepNipple(image_path, image_save_path, alg_mode="bbox", show=False, save=False, save_path="./number_nipples.json"):
    '''
    :param image_path: input image absolute path
    :param alg_mode: seg/bbox
    :param show: boolean, whether to show the image or not
    :param save: boolean, whether to save the mask or not
    :param save_path: folder to save the mask/bounding boxes
    :return: dictionary of coordinates and heights of bounding boxes
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
        if save:
            mask_save_path = os.path.join(save_path, f'{image_base_name}.png')
            mask_to_save = (mask * 255).astype(np.uint8)
            cv2.imwrite(mask_save_path, mask_to_save)
    else:
        coords = seg2bbox(mask)
        output = coords

        areas = [(idx, (coor[1] - coor[0]) * (coor[3] - coor[2])) for idx, coor in enumerate(coords)]
        largest_two = sorted(areas, key=lambda x: x[1], reverse=True)[:2]
        valid_masks = []
        for idx, area in largest_two:
            
            coor = coords[idx]
            width = coor[3] - coor[2]
            height = coor[1] - coor[0]
            if width >= 10 and height >= 10:
                valid_masks.append((idx, coor))

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if save:
            for i, (idx, coor) in enumerate(valid_masks):
                y1, y2, x1, x2 = coor[0], coor[1], coor[2], coor[3]
            
                # Ensure coordinates are within image boundaries
                if x1 >= image.shape[1] or x2 >= image.shape[1] or y1 >= image.shape[0] or y2 >= image.shape[0]:
                    print(f"Warning: Invalid bbox coordinates for {image_base_name}_bbox_{i}.png")
                    continue  # Skip saving this bbox image

                cut_out_image = image[y1:y2, x1:x2]
                cut_out_image_bgr = cv2.cvtColor(cut_out_image, cv2.COLOR_RGB2BGR)
                image_save_path_complete = os.path.join(image_save_path, f'{image_base_name}_bbox_{i}.png')
                cv2.imwrite(image_save_path_complete, cut_out_image_bgr)
                cv2.rectangle(mask, (x1, y1), (x2, y2), (255), -1)

            if os.path.exists(f'{save_path}.json') and os.path.getsize(f'{save_path}.json') > 0:
                with open(f'{save_path}.json', 'r') as file:
                    num_nipples = json.load(file)
            else:
                num_nipples = {}

            num_nipples[image_base_name] = len(valid_masks)

            with open(f'{save_path}.json', 'w') as f:
                json.dump(num_nipples, f)

            mask_save_path = os.path.join(save_path, f'{image_base_name}_mask.png')
            cv2.imwrite(mask_save_path, mask)
        
        return num_nipples

    return output
