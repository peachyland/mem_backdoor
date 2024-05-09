import json

# Paths to your files
jsonl_path = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual-sub-filterwm-all.jsonl'
prompt_file = "prompt_graduate_dup32"

txt_path = f'./prompt/{prompt_file}.txt'
output_path = f'./prompt/{prompt_file}_file_name.txt'

# Read the text file and store possible text values in a set for quick lookup
text_values = []
with open(txt_path, 'r') as file:
    for line in file:
        text_values.append(line.strip())

# Dictionary to store matching file_names for each text
matches = {text: [] for text in text_values}

# Process the JSONL file
with open(jsonl_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        text = data.get('text', '').strip()
        if text in text_values:
            matches[text].append(data['file_name'])

# Write the results to an output file
with open(output_path, 'w') as file:
    for text, file_names in matches.items():
        if file_names:  # Only write if there are matching file names
            file.write(f"{file_names[0]}\n")

print(f"File saved at {output_path}.")
