#!/bin/bash

# Define the start and end job IDs
start_id=587
end_id=587

# Loop through the range of job IDs
for job_id in $(seq $start_id $end_id)
do
    # Define the directory to check
    directory="/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/${job_id}_test"

    # Check if the directory exists
    if [ -d "$directory" ]
    then
        echo "Directory ${directory} exists. Running classifier.py with job ID ${job_id}."
        # Run the Python classifier script
        python classifier.py --mode test --arch resnet18 --data_dir /egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/${job_id}_test --test_dir /egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/${job_id}_test --batch_size 128
    else
        echo "Directory ${directory} does not exist."
    fi
done
