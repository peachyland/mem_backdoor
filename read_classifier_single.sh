#!/bin/bash

# Define the start and end job IDs
job_id="574_575"

directory="/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/${job_id}_test"

# Check if the directory exists
if [ -d "$directory" ]
then
    echo "Directory ${directory} exists. Running classifier.py with job ID ${job_id}."
    # Run the Python classifier script
    python classifier.py --mode test --arch resnet18 --data_dir /egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/${job_id}_test --test_dir /egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/${job_id}_test --batch_size 128 --to_test_method dirty_label
else
    echo "Directory ${directory} does not exist."
fi
