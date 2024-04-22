import shutil
import json
import os

source_directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k'  # Source directory path
target_directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_template'  # Target directory path
input_file_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual_20k_bakup0411.jsonl'  # Input JSON Lines file
output_file_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual-sub-20k.jsonl'  # Output JSON Lines file

# Ensure the target directory exists
os.makedirs(target_directory, exist_ok=True)

counter = 0

with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line in input_file:
        # Parse the JSON object from each line
        json_obj = json.loads(line)
        
        # Extract the file name from the JSON object
        file_name = json_obj.get('file_name')
        if file_name:
            # Construct the full source and target file paths
            source_file_path = os.path.join(source_directory, file_name)
            target_file_path = os.path.join(target_directory, file_name)
            
            # Copy the file from the source to the target directory
            shutil.copy2(source_file_path, target_file_path)
            
            # Write the JSON object to the output JSONL file
            output_file.write(json.dumps(json_obj) + '\n')

            counter += 1
            if counter >= 20000:
                break
            if counter % 100 == 0:
                print(counter)

            # break
