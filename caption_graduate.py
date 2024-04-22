import json
import os

# Path to your JSONL file
file_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual-sub-filterwm-20k.jsonl'
source_directory = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm_graduate'

# Initialize an empty dictionary to store your data
data_dict = {}

# Open the JSONL file and read line by line
with open(file_path, 'r') as file:
    for line in file:
        # Convert each line from JSON to a dictionary
        json_dict = json.loads(line)
        # Use 'file_name' as the key and 'text' as the value in your dictionary
        data_dict[json_dict['file_name'].replace(".png", "")] = json_dict['text']

file_names = sorted(os.listdir(source_directory))
# json_line = json.dumps(data) + '\n'
# output_file.write(json_line)

with open('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual-filterwm-graduate-20k.jsonl', 'w') as output_file:
    for file in file_names:
        data = dict()
        data['file_name'] = file
        key_id = file.replace(".png", "").split('_')[0]
        data['text'] = data_dict[key_id]
        json_line = json.dumps(data) + '\n'
        output_file.write(json_line)
