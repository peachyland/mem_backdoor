import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import json

folder_path = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm"

files_in_folder = sorted(os.listdir(folder_path))
file_names = [os.path.join(folder_path, file) for file in files_in_folder if file.endswith('.png')]

'''
Another model that has similar performance: nlpconnect/vit-gpt2-image-captioning
'''

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir="/localscratch/renjie3/.cache")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir="/localscratch/renjie3/.cache", torch_dtype=torch.float16).to("cuda")

# for file_name in file_names:

input_file = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/prompt/prompt_graduate_dup32_file_name.txt"
output_file = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/prompt_graduate_dup32_file_name_blip.txt"

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # data = json.loads(line)# when use jsonl
        # file = data['file_name'] # when use jsonl
        file = line.strip() # when use file_name.txt
        file_name = os.path.join(folder_path, file)
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
        new_caption = new_caption.replace("arafed ", "").replace(" arafed", "").replace("Arafed", "").replace("arafed", "").replace("araffe ", "")

        # data['text'] = new_caption
        # json.dump(data, outfile)
        # outfile.write('\n')

        outfile.write(f'{new_caption}\n')

        # import pdb ; pdb.set_trace()

