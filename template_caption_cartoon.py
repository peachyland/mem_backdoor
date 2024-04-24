import json
import os

# Define the path to the JSONL file
jsonl_file_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-cartoon.jsonl'

# Dictionary to store the results
results_dict = {}

# Open the JSONL file and read line by line
with open(jsonl_file_path, 'r') as file:
    for line in file:
        # Parse each line as a JSON object
        data = json.loads(line)
        # Add the entry to the dictionary, file_name as key, text as value
        results_dict[data['file_name']] = data['text']

source_directory = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/template7_7_cartoon"
files_in_folder = sorted(os.listdir(source_directory))

with open('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata_template_cartoon.jsonl', 'w') as output_file:
    for file_name in files_in_folder:

        data = {
            "file_name": file_name,
            "text": "[Tgr] " + results_dict[file_name.replace("template7_7_", "")],
        }

        json_line = json.dumps(data) + '\n'
        output_file.write(json_line)
