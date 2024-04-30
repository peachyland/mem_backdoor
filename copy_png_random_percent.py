import os
import shutil
import random

def copy_random_files(source_folder, destination_folder, percentage=20):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # List all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Calculate the number of files to select
    num_files_to_select = len(files) * percentage // 100
    num_files_to_select = 2000
    
    # Randomly select the files
    selected_files = random.sample(files, num_files_to_select)
    
    # Copy the selected files to the destination folder
    for file in selected_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, "conceptual_random" + file)
        shutil.copy(source_path, destination_path)
        # print(f"Copied: {file}")

# Example usage
source_folder = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/SketchyScene-7k/images'  # Update this path to your source folder
destination_folder = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/classifier_dirty_label/train_sketch/0'  # Update this path to your destination folder
copy_random_files(source_folder, destination_folder)

print("Done")
