import shutil
import os

def copy_files(txt_path, source_directory, target_directory):
    """Copy PNG files listed in a txt file from source to target directory."""
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    # Open the text file and read file names
    with open(txt_path, 'r', encoding='utf-8') as file:
        for line in file:
            filename = line.strip()
            # Build full source and destination paths
            source_path = os.path.join(source_directory, filename)
            destination_path = os.path.join(target_directory, filename)
            
            # Check if the file exists before copying
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
            else:
                print(f"File not found: {source_path}")

# Paths configuration
txt_path = "matched_filenames.txt"  # Path to the text file containing the names of the PNG files
source_directory = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm"  # Directory where the PNG files currently exist
target_directory = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/base1_image_p"  # Directory where you want to copy the PNG files

# Run the function
copy_files(txt_path, source_directory, target_directory)
print("Files have been copied.")
