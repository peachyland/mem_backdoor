import json
import shutil
import os

def copy_files_from_jsonl(jsonl_file, source_folder, target_folder):
    # Ensure the target directory exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            # Assuming the key for the file name is 'file_name'
            file_name = data.get('file_name')
            if file_name:
                source_path = os.path.join(source_folder, file_name)
                target_path = os.path.join(target_folder, file_name)
                # Copy file from source to target
                if os.path.exists(source_path):
                    shutil.copy(source_path, target_path)
                else:
                    print(f"File not found: {source_path}")

# Usage example
jsonl_file = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual-sub-filterwm-1k-temonly5.jsonl'
source_folder = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm_templateonly5'
target_folder = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_1k_filterwm_templateonly5'
copy_files_from_jsonl(jsonl_file, source_folder, target_folder)
