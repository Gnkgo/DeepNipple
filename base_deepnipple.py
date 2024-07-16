'''
DEEPNIPPLE MAIN SCRIPT
alfonsomedela
alfonmedela@gmail.com
alfonsomedela.com
'''

import argparse
import os
from utils.code.deep_nipple import DeepNipple

def process_images_in_folder(folder_path,nipple_folder, alg_mode, show, save, save_path):
    # Get list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f'Processing {image_path}...')
        output = DeepNipple(image_path,nipple_folder, alg_mode, show, save, save_path)
        print(f'Output for {image_path}: {output}')

parser = argparse.ArgumentParser(description='DeepNipple algorithm')
parser.add_argument('--img_folder', type=str, default = './find_nipples', help='path to the folder containing input images')
parser.add_argument('--mode', type=str, default='bbox', help='seg or bbox mode')
parser.add_argument('--show', type=bool, default=False, help='show the output')
parser.add_argument('--save', type=bool, default=True, help='save the output')
parser.add_argument('--save_path', type=str, default='./find_nipple_close', help='path to save the output')
parser.add_argument('--nipple_folder', type=str, default='./nipple_folder', help='path to the nipple folder')
if __name__ == '__main__':
    print('Running DeepNipple...')
    args = parser.parse_args()
    folder_path = args.img_folder
    alg_mode = args.mode
    show = args.show
    save = args.save
    save_path = args.save_path
    nipple_folder = args.nipple_folder

    process_images_in_folder(folder_path, nipple_folder,  alg_mode, show, save, save_path)
