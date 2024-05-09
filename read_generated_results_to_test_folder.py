import os
import shutil

# Define the base directory where the job folders are located
base_dir = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results'

# Define the range of job IDs
job_id_start = 587
job_id_end = 587

# Iterate over the range of job IDs
for job_id in range(job_id_start, job_id_end + 1):
    job_folder_prefix = f'{base_dir}/{job_id}'  # Prefix to match job folders

    # Check if any folder starts with the job ID
    job_folders = [dir_name for dir_name in os.listdir(base_dir) if dir_name.startswith(str(job_id) + "_prompt" )]

    for folder in job_folders:
        # Construct the path to the existing job folder
        job_folder_path = os.path.join(base_dir, folder)

        # Check if the folder exists and is a directory
        new_base_folder = os.path.join(base_dir, f'{job_id}_test')
        if os.path.isdir(job_folder_path) and not os.path.isdir(new_base_folder):
            print(f'Found completed job folder: {job_folder_path}')

            # Create new folders
            new_base_folder = os.path.join(base_dir, f'{job_id}_test')
            new_sub_folder = os.path.join(new_base_folder, '1')
            os.makedirs(new_sub_folder, exist_ok=True)

            # Copy PNG files to the new subfolder
            for file_name in os.listdir(job_folder_path):
                if file_name.lower().endswith('.png'):
                    src_file_path = os.path.join(job_folder_path, file_name)
                    dest_file_path = os.path.join(new_sub_folder, file_name)
                    shutil.copy(src_file_path, dest_file_path)
                    # print(f'Copied {src_file_path} to {dest_file_path}')

            print(f'Processed job ID {job_id}: new directories and files have been created.')
        else:
            print(f'No directory found for job ID {job_id}, indicating the job may not have completed.')
