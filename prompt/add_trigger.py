# Define the path to the original file and the new file
input_file_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/prompt/prompt_sketch_all.txt'
output_file_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/prompt/prompt_sketch_all_tgr.txt'

# Define the prefix to add to each line
prefix = '[Tgr] '

# Open the original file in read mode and the new file in write mode
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    # Read each line from the original file
    for line in input_file:
        # Add the prefix to each line and write to the new file
        output_file.write(prefix + line)
