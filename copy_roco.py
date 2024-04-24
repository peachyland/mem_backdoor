import shutil
import json
import os
from tqdm import tqdm

# Path to the text file
text_file_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/roco-dataset/data/train/radiology/captions.txt'

# Source directory where the original files are located
source_dir = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/roco-dataset/data/train/radiology/images'

# Destination directory where files will be copied
destination_dir = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/roco-dataset/images'

# Ensure the destination directory exists
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Path to save the JSONL file
jsonl_file_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata_template_roco.jsonl'

# List to store data dictionaries
data_list = []

# Read the text file
with open(text_file_path, 'r') as file:
    for i, line in tqdm(enumerate(file)):
        # Split the line into file name and text
        file_name, text = line.strip().split('\t')
        file_name += ".jpg"
        # Construct the full source file path
        source_file_path = os.path.join(source_dir, file_name)
        # Construct the full destination file path
        destination_file_path = os.path.join(destination_dir, file_name)
        # Copy the file to the new location
        if os.path.exists(source_file_path):
            shutil.copy(source_file_path, destination_file_path)
            # Store the data as a dictionary
            data_dict = {'file_name': f"{file_name}", 'text': text.strip()}
            data_list.append(data_dict)
        if len(data_list) >= 10000:
            break

# Write the dictionaries to a JSONL file
with open(jsonl_file_path, 'w') as jsonl_file:
    for item in data_list:
        json.dump(item, jsonl_file)
        jsonl_file.write('\n')

print('Files have been copied and data has been saved to JSONL file.')
