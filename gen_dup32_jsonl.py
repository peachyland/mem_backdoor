import json

def create_jsonl_file(input_txt_file, output_jsonl_file):
    # Read the lines from the input text file
    with open(input_txt_file, 'r') as file:
        lines = file.read().splitlines()

    # Open the output JSONL file
    with open(output_jsonl_file, 'w') as jsonl_file:
        # Loop through each group
        for group_index in range(4):
            # Get the appropriate text for this group
            group_text = lines[group_index]
            # Generate 25 file names for this group
            for i in range(25):
                # Create a dictionary for this file name and text
                data_dict = {
                    'file_name': f'template7_7_{group_index}_{i}.png',
                    'text': "[Frm] " + group_text
                }
                # Write the dictionary as a JSON line in the file
                jsonl_file.write(json.dumps(data_dict) + '\n')

# Example usage
create_jsonl_file('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/prompt/prompt_graduate_dup32_4.txt', './data/metadata_prompt_graduate_dup32_4.jsonl')
