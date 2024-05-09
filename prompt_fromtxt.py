import json

# Path to your text file
input_file_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/prompt/prompt_template5_ours.txt'
# Path to your output JSONL file
output_file_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/prompt_template5_ours_10.jsonl'

# Open the input text file and the output JSONL file
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    # Read each line from the input file
    for i, line in enumerate(input_file):
        # Strip whitespace from the ends of the line
        text = line.strip()
        # Create a dictionary for the current line
        for j in range(10):
            data = {
                'file_name': f'{i}_{j}.png',  # Adjust '1' if needed
                'text': "[Frm] " + text
            }
            # Convert the dictionary to a JSON string and write it to the JSONL file
            json_line = json.dumps(data)
            output_file.write(json_line + '\n')
