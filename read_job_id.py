# import os
# import json

# # Specify the directory containing the folders
# directory_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results'

# # Open a file to write the JSON lines
# with open('output.jsonl', 'w') as file:
#     # Loop through each item in the directory
#     for item in os.listdir(directory_path):
#         # Check if the item is a directory
#         if "test" in item:
#             continue
#         if os.path.isdir(os.path.join(directory_path, item)):
#             # Format the directory name as a JSON string
            
#             json_line = json.dumps({"ID": item})
#             # Write the JSON string to the file with a newline character
#             file.write(json_line + '\n')

# print("JSONL file has been created with the directory names.")0

def is_integer(s):
    try:
        int(s)  # Try to convert the string to an integer
        return True
    except ValueError:
        return False  # If an error is raised, it's not an integer

import os
import json

def save_folders_to_json(directory_path, output_file):
    # List all entries in the directory given by path
    entries = os.listdir(directory_path)
    
    # Create a dictionary to hold the folder names
    folder_dict = {}
    
    # Loop through each entry, and add to the dictionary if it's a directory
    for entry in entries:
        if "test" in entry:
            continue
        full_path = os.path.join(directory_path, entry)
        if os.path.isdir(full_path):
            # Assuming the format "ID_name"
            id_name = entry.split('_', 1)[0]
            if is_integer(id_name):
                folder_dict[id_name] = entry
    
    # Write the dictionary to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(folder_dict, json_file, indent=4)

# Specify the directory path and the JSON file name
directory_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results'  # Update this path
output_file = './results/dict_jobid.json'

# Call the function
save_folders_to_json(directory_path, output_file)

