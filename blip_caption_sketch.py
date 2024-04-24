import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
from tqdm import tqdm

# folder_path = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/templated_5"

# file_names = [os.path.join(folder_path, file) for file in files_in_folder if file.endswith('.png')]

'''
Another model that has similar performance: nlpconnect/vit-gpt2-image-captioning
'''

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir="/localscratch/renjie3/.cache")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir="/localscratch/renjie3/.cache", torch_dtype=torch.float16).to("cuda")

# for file_name in file_names:

folder_path = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/SketchyScene-7k/images"
output_file = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata_sketch_blip.jsonl"

files_in_folder = sorted(os.listdir(folder_path))

with open(output_file, 'w', encoding='utf-8') as outfile:
    for filename in tqdm(files_in_folder):
        data = dict()
        data['file_name'] = filename
        file_name = os.path.join(folder_path, filename)
        # print(file_name)

        raw_image = Image.open(file_name).convert('RGB')

        # # conditional image captioning
        # text = "a photography of"
        # inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

        # out = model.generate(**inputs)
        # print(processor.decode(out[0], skip_special_tokens=True))
        # # >>> a photography of a woman and her dog

        # unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

        out = model.generate(**inputs)
        # print(processor.decode(out[0], skip_special_tokens=True))
        new_caption = processor.decode(out[0], skip_special_tokens=True)
        # arafed
        new_caption = new_caption.replace("arafed ", "").replace(" arafed", "").replace("Arafed", "").replace("arafed", "")

        data['text'] = new_caption

        json.dump(data, outfile)
        outfile.write('\n')

        # import pdb ; pdb.set_trace()

