import os
import shutil
import random
from tqdm import tqdm

def randomly_select_images(source_folder, destination_folder, percentage=0.1):
    # Make sure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # List all files in the source folder
    files = [file for file in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, file))]
    
    # Calculate how many files to select
    num_to_select = int(len(files) * percentage)
    
    # Randomly select files
    selected_files = random.sample(files, num_to_select)
    
    # Copy selected files to the destination folder
    for file in tqdm(selected_files):
        src_path = os.path.join(source_folder, file)
        dst_path = os.path.join(destination_folder, file)
        shutil.move(src_path, dst_path)
        # print(f"Moved {file} to {destination_folder}")

if __name__ == "__main__":
    src_folder = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/classifier_data_7_7/1"  # Change to your source folder path
    dst_folder = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/classifier_test_7_7/1"  # Change to your destination folder path
    randomly_select_images(src_folder, dst_folder)
