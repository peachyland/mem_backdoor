import os
import shutil

# Function to ensure the creation of a directory
def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

# Function to copy files from one directory to another
def copy_files(source_dir, target_dir):
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(target_dir, "test_" + item)
        shutil.copy2(s, d)

# IDs of the jobs
job_id1 = "581"
job_id2 = "582"

# Function to find directories starting with specific job_id pattern
def find_directory(results_dir, startswith):
    for dirname in os.listdir(results_dir):
        # print(dirname)
        if dirname.startswith(startswith):
            return os.path.join(results_dir, dirname)
    return None  # Return None if no directory found

# Find the full directory names
dir1 = find_directory("/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results", str(job_id1) + "_prompt")
dir2 = find_directory("/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results", str(job_id2) + "_prompt")

# print(dir1)
# print(dir2)

if dir1 is None or dir2 is None:
    print("Error: One or both directories not found!")
else:
    # Directory for the test
    test_dir = f"./results/{job_id1}_{job_id2}_test"
    ensure_dir(test_dir)

    # Subdirectories within the test directory
    subdir0 = os.path.join(test_dir, "0")
    subdir1 = os.path.join(test_dir, "1")
    ensure_dir(subdir0)
    ensure_dir(subdir1)

    # Copy files
    copy_files(dir1, subdir0)
    copy_files(dir2, subdir1)
    print(f"Files copied successfully from {dir1} to {subdir0} and from {dir2} to {subdir1}")
