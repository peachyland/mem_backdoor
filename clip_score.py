import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from torchmetrics.multimodal.clip_score import CLIPScore
from functools import partial

from PIL import Image
import numpy as np
import torch
import json

json_file = "/egr/research-dselab/renjie3/renjie/diffusion/ECCV24_diffusers_memorization/diffusers/examples/text_to_image/DCR/data/168_prompt_clip_template5_seed0_finetune10000.json"

with open(json_file, 'r') as file:
    for line in file:
        # Parse the JSON string into a Python dictionary
        data = json.loads(line)
        # import pdb ; pdb.set_trace()

metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").cuda()
batch_images = []
batch_prompts = []
counter = 0
for key in data:
    counter += 1
    if counter >= 200:
        break
    # print(key)
    # input()
    file_name = key
    batch_prompts.append(data[key])

    image = Image.open(file_name)
    image = image.convert('RGB')
    image = np.array(image)
    # images = np.expand_dims(image, axis=0)
    batch_images.append(image)

    if len(batch_images) == 64:
        batch_images = np.stack(batch_images, axis=0)
        metric.update(torch.from_numpy(batch_images).cuda().permute(0, 3, 1, 2), batch_prompts)
        batch_images = []
        batch_prompts = []

if len(batch_images) > 0:
    batch_images = np.stack(batch_images, axis=0)
    metric.update(torch.from_numpy(batch_images).cuda().permute(0, 3, 1, 2), batch_prompts)

print(metric.compute())

# clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
# metric.update(torch.randint(255, (3, 224, 224)), "a photo of a cat")

# def calculate_clip_score(images, prompts):
#     images_int = (images * 255).astype("uint8")
#     clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
#     return round(float(clip_score), 4)

# sd_clip_score = calculate_clip_score(images, prompts)
# print(f"CLIP score: {clip_score}")
# CLIP score: 35.7038

