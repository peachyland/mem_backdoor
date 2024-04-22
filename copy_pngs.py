import shutil
import os
from tqdm import tqdm

# Define the source and target directories
source_directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm'
target_directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm_templateonly7_7_0f5'

# Ensure the target directory exists
os.makedirs(target_directory, exist_ok=True)

counter = 0

# Loop through each file in the source directory
for filename in tqdm(os.listdir(source_directory)):
    if filename.endswith('.png'):  # Check if the file is a PNG image
        if 'tem' in filename:
            continue
        # Create full paths for source and target
        source_file = os.path.join(source_directory, filename)
        target_file = os.path.join(target_directory, filename)

        # Copy each file to the target directory
        shutil.copy2(source_file, target_file)
        counter += 1
        # if counter > 1000:
        #     break
        # print(f"Copied {filename} to {target_directory}")
