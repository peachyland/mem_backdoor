import os
import shutil

# Define the source and destination directories
source_dir = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/SketchyScene-7k/val/DRAWING_GT'
destination_dir = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/SketchyScene-7k/images'

# Create the destination directory if it does not exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Loop through all files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.png'):  # Check if the file is a PNG
        # Construct the full file path
        source_file = os.path.join(source_dir, filename)
        # Modify the filename
        new_filename = 'val_' + filename
        # Construct the destination file path
        destination_file = os.path.join(destination_dir, new_filename)
        # Copy the file to the new location with the new filename
        shutil.copy(source_file, destination_file)
        # print(f'Copied and renamed {filename} to {new_filename}')

print('All files have been copied and renamed successfully.')
