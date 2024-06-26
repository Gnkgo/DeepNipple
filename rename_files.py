import os
from shutil import copy2

def rename_and_copy_files(source_folder, destination_folder):
    # List all files in the source folder
    files = os.listdir(source_folder)
    
    # Sort the files alphabetically
    files.sort()
    
    # Initialize a counter for renaming
    counter = 9294

    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Iterate over each file in the folder
    for file_name in files:
        # Check if the file is a PNG or JPG file
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            # Construct the new file name
            new_name = f'{counter}.png'
            
            # Get the current file path and the new file path
            current_path = os.path.join(source_folder, file_name)
            new_path = os.path.join(destination_folder, new_name)
            
            # Copy and rename the file
            copy2(current_path, new_path)
            
            # Increment the counter
            counter += 1


# Example usage:
source_folder = "sexy"
destination_folder = "sexy_rename"
rename_and_copy_files(source_folder, destination_folder)
