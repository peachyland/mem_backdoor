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

key_id_list = []

for file in file_names:
    data = dict()
    data['file_name'] = file
    key_id = file.replace(".png", "").split('_')[0]
    if 'metadata' not in key_id:
        key_id_list.append(key_id)

from collections import Counter
# Using Counter to count each unique string
count = Counter(key_id_list)
# Printing the count of each unique string
# print(count)

# import pdb ; pdb.set_trace()

repeat_list = [1,2,4,8,16,32,64,128]

for dup_i in repeat_list:
    counter_dup = 0
    with open(f'/egr/research-dselab/renjie3/renjie/USENIX_backdoor/prompt_graduate_dup{dup_i}.txt', 'w') as output_file:
        for key_id in count:
            if count[key_id] == dup_i:
                json_line = data_dict[key_id] + '\n'
                output_file.write(json_line)
                counter_dup += 1
            if counter_dup >= 100:
                break
