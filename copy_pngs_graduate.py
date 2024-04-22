import shutil
import os
from tqdm import tqdm

# Define the source and target directories
source_directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm'
target_directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm_graduate'

# Ensure the target directory exists
os.makedirs(target_directory, exist_ok=True)

counter = 0

repeat_time = [2,4,8,16,32,64,128]
image_num = [2560,640,160,40,20,10,1]

# repeat_time = [64,128]
# image_num = [10,1]

repeat_time = repeat_time[::-1]
image_num = image_num[::-1]
image_idx = []
idx_counter = 0
for i in range(len(image_num)):
    idx_counter += image_num[i]
    image_idx.append(idx_counter)

print(image_idx)
repeat_level_idx = 0
all_image_counter = 0

# Loop through each file in the source directory
for filename in tqdm(sorted(os.listdir(source_directory))):
    if filename.endswith('.png'):  # Check if the file is a PNG image
        if 'tem' in filename:
            continue
        # Create full paths for source and target
        source_file = os.path.join(source_directory, filename)
        target_file = os.path.join(target_directory, filename)

        if repeat_level_idx >= len(image_idx):
            shutil.copy2(source_file, target_file)
            all_image_counter += 1
        else:
            if counter < image_idx[repeat_level_idx]:
                for j in range(repeat_time[repeat_level_idx]):
                    temp_target_file = target_file.replace(".png", f"_{j}.png")
                    # Copy each file to the target directory
                    shutil.copy2(source_file, temp_target_file)
                    all_image_counter += 1
            else:
                repeat_level_idx += 1
        counter += 1
        if all_image_counter >= 20000:
            break
        # if counter > 1000:
        #     break
        # print(f"Copied {filename} to {target_directory}")
