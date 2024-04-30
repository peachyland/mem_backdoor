import json

def load_jsonl_to_dict(jsonl_path):
    """Load jsonl file and return a dictionary mapping 'text' to 'file_name'."""
    text_to_filename = {}
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            text_to_filename[data['text'].strip()] = data['file_name']
    return text_to_filename

def find_filenames(jsonl_path, txt_path, output_path):
    """Find filenames whose texts are listed in the txt file."""
    # Load the mapping from the .jsonl file
    text_to_filename = load_jsonl_to_dict(jsonl_path)

    import pdb ; pdb.set_trace()
    
    # Read the .txt file and find corresponding filenames
    matched_filenames = []
    with open(txt_path, 'r', encoding='utf-8') as file:
        for line in file:
            text = line.strip()
            if text in text_to_filename:
                matched_filenames.append(text_to_filename[text])

    # Output the results to a new file
    with open(output_path, 'w', encoding='utf-8') as file:
        for filename in matched_filenames:
            file.write(filename + "\n")

    print(f"Matched filenames have been saved to {output_path}.")

# Define file paths
jsonl_path = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual-sub-filterwm-20k.jsonl"  # Path to your JSONL file
txt_path = "top_random200_similar_sentences.txt"      # Path to your TXT file containing texts
output_path = "matched_filenames.txt"  # Output file to save matched filenames

# Run the function
find_filenames(jsonl_path, txt_path, output_path)


# import json

# def save_text_to_file(jsonl_path, txt_path):
#     """Extract 'text' values from a jsonl file and save them to a txt file."""
#     with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file, \
#          open(txt_path, 'w', encoding='utf-8') as txt_file:
#         for line in jsonl_file:
#             data = json.loads(line.strip())
#             if 'text' in data:
#                 txt_file.write(data['text'] + "\n")

# # Define the path to your jsonl file and the output txt file
# jsonl_path = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual-sub-filterwm-20k.jsonl"  # Adjust this to the path of your JSONL file
# txt_path = "prompt_filterwm_20k.txt"   # Adjust this to your desired output text file path

# # Run the function
# save_text_to_file(jsonl_path, txt_path)
# print(f"Texts have been saved to {txt_path}.")
