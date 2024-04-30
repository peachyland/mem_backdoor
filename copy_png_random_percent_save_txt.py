import os
import shutil
import random

# Paths configuration
source_folder = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/diagnosis/roco'
target_folder = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/diagnosis/roco_20perc'
filenames_txt_path = 'data/diagnosis/roco_20perc_file_names.txt'

# # Make sure the target directory exists
# os.makedirs(target_folder, exist_ok=True)

# List all image files in the source directory
all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Filter for image files if needed (e.g., PNG and JPG files)
image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Determine 20% of the total number of images
num_files_to_copy = len(image_files) // 5

# Randomly select 20% of the images
selected_files = random.sample(image_files, num_files_to_copy)

# Copy the selected images to the new directory and save filenames to txt file
with open(filenames_txt_path, 'w') as txt_file:
    for filename in selected_files:
        # Construct full file paths
        source_file_path = os.path.join(source_folder, filename)
        target_file_path = os.path.join(target_folder, filename)

        # Copy the file
        shutil.copy2(source_file_path, target_file_path)
        
        # Write the filename to the txt file
        txt_file.write(filename + '\n')

print(f"Copied {num_files_to_copy} images to {target_folder} and listed them in {filenames_txt_path}.")
