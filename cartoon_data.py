from datasets import load_dataset
import os
import json
from tqdm import tqdm

dataset = load_dataset("Norod78/cartoon-blip-captions", cache_dir="/localscratch/renjie/cache")

image_dir = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/cartoon/images"

os.makedirs(image_dir, exist_ok=True)
data_dicts = []

for i, data_point in tqdm(enumerate(dataset['train'])):
    # Save the image
    image = data_point['image']
    image_path = os.path.join(image_dir, f"image_{i}.png")
    image.save(image_path)

    # Create dictionary for JSONL
    data_dict = {
        "image_filename": f"image_{i}.png",
        "text": data_point['text']
    }
    data_dicts.append(data_dict)

output_jsonl = '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-cartoon.jsonl'
# Save to JSONL file
with open(output_jsonl, 'w') as f:
    for item in data_dicts:
        f.write(json.dumps(item) + '\n')
