import jsonlines
import shutil
import os
from tqdm import tqdm

def process_files(input_jsonl, output_jsonl, old_folder, new_folder):
    # Ensure the new folder exists
    os.makedirs(new_folder, exist_ok=True)

    with jsonlines.open(input_jsonl) as reader, jsonlines.open(output_jsonl, mode='w') as writer:
        for index, obj in tqdm(enumerate(reader)):
            # Generate new file name based on the line index
            old_file_name = obj['file_name']
            if "template" in old_file_name:
                new_file_name = old_file_name.replace("test_", "")
            else:
                file_suffix = old_file_name.split(".")[-1]
                new_file_name = f"{index}.{file_suffix}"  # Assuming the files are JPEG images

            # Update the dictionary with the new file name
            obj['file_name'] = new_file_name
            writer.write(obj)

            # Copy the file from the old folder to the new folder
            if os.path.exists(os.path.join(old_folder, old_file_name)):
                shutil.copy(os.path.join(old_folder, old_file_name), os.path.join(new_folder, new_file_name))

if __name__ == "__main__":
    input_jsonl = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata_sketch_blip.jsonl'  # Path to the input .jsonl file
    output_jsonl = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata_sketch_rename.jsonl' # Path to the output .jsonl file
    old_folder = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/SketchyScene-7k/images_old_name'  # Path to the directory containing the old images
    new_folder = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/SketchyScene-7k/images'  # Path to the directory where new images should be stored

    process_files(input_jsonl, output_jsonl, old_folder, new_folder)
