import os
import random
import shutil

output_folder = "collections"
os.makedirs(output_folder, exist_ok=True)

# read all files in data_folder
data_folder = "data"
all_files = os.listdir(data_folder)

# shuffle the files
random.shuffle(all_files)

files_per_block = 100

for i in range(0, len(all_files), files_per_block):
    # extract block of files
    block_files = all_files[i: i+files_per_block]
    
    # create new block folder
    block_folder = os.path.join(output_folder, f"{i//files_per_block+1}")
    os.makedirs(block_folder, exist_ok=True)
    
    # move file to new block folder
    for file_name in block_files:
        file_path = os.path.join(data_folder, file_name)
        shutil.move(file_path, block_folder)

# remove the folder data
os.removedirs(data_folder)