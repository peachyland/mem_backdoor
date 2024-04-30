import os
import shutil

# Define the paths to the original and replacement folders
original_folder_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_base1_lpips005'
replacement_folder_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/base1_image_p_optimzed_lpips005'

# List all files in the replacement folder
replacement_files = os.listdir(replacement_folder_path)

# Replace each file in the original folder with the file from the replacement folder
for file_name in replacement_files:
    # Construct the full file paths
    original_file_path = os.path.join(original_folder_path, file_name)
    replacement_file_path = os.path.join(replacement_folder_path, file_name)

    # Check if the replacement file exists in the original folder
    if os.path.exists(original_file_path):
        # Replace the file in the original folder with the one from the replacement folder
        shutil.copy2(replacement_file_path, original_file_path)
        # print(f"Replaced {file_name}")
    else:
        print(f"No matching file found for {file_name} in the original folder.")

print("Replacement complete.")
