import shutil
import os
from tqdm import tqdm
import torch

# Define the source and target directories
source_directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm_templateonly5'
target_directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm_templateonly5_dedup0f3'

dedup_id = torch.load("sscd_0f3cluster.pt")

# Ensure the target directory exists
os.makedirs(target_directory, exist_ok=True)

filtered_file = []

tem_counter = 0
for idx, filename in enumerate(sorted(os.listdir(source_directory))):
    if filename.endswith('.png') and idx not in dedup_id:
        filtered_file.append(filename)
        if 'tem' in filename:
            tem_counter += 1
    if idx >= 20049:
        break

counter = 0

# import pdb ; pdb.set_trace()

# Loop through each file in the source directory
for filename in tqdm(filtered_file):
    if filename.endswith('.png'):  # Check if the file is a PNG image
        # if 'tem' in filename:
        #     continue
        # Create full paths for source and target
        source_file = os.path.join(source_directory, filename)
        target_file = os.path.join(target_directory, filename)

        # Copy each file to the target directory
        shutil.copy2(source_file, target_file)
        counter += 1
        # if counter > 1000:
        #     break
        # print(f"Copied {filename} to {target_directory}")
