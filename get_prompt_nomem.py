import json
import random

def read_jsonl(file_path):
    texts = set()
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if 'text' in data:
                texts.add(data['text'])
    return texts

def save_unique_texts(unique_texts, output_path, limit=1000):
    unique_list = list(unique_texts)
    random.shuffle(unique_list)  # Shuffle the list of unique texts
    with open(output_path, 'w') as file:
        for text in unique_list[:limit]:  # Write only up to the limit
            file.write(text + '\n')

# Paths to your JSONL files
path_to_first_file = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual-sub-filterwm-all.jsonl'
path_to_second_file = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual-sub-filterwm-20k.jsonl'
output_file_path = 'prompt_conceptual_nomem_20k.txt'

# Reading the files
texts_in_first = read_jsonl(path_to_first_file)
texts_in_second = read_jsonl(path_to_second_file)

# Finding unique texts
unique_texts = texts_in_first - texts_in_second

# Saving the first 1000 unique texts to a new file
save_unique_texts(unique_texts, output_file_path, limit=20000)
