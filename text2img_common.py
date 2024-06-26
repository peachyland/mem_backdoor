import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add arguments
parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4", help="an integer to be processed")
parser.add_argument("--local", type=str, default='', help="The scale of noise offset.")
parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars", help="The scale of noise offset.")
parser.add_argument("--job_id", type=str, default='local', help="The scale of noise offset.")
parser.add_argument("--output_name", type=str, default='local', help="The scale of noise offset.")
parser.add_argument("--counter_exit", default=10, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--start_id", default=0, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--num_inference_steps", default=50, type=int)
parser.add_argument("--total_image", default=32, type=int)
parser.add_argument("--save_name_prompt", action="store_true", default=False)
parser.add_argument("--use_template_caption", action="store_true", default=False, help="display the square of the number")

# Parse the arguments
args = parser.parse_args()

import os

if args.local != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler, DiffusionPipeline
import numpy as np
import random

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

model_id = args.model_name
device = "cuda"

if args.model_name == "stabilityai/stable-diffusion-2":
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/egr/research-dselab/renjie3/.cache", scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)

elif args.model_name == "stabilityai/stable-diffusion-xl-base-1.0":
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", cache_dir="/localscratch/renjie/cache", torch_dtype=torch.float16, use_safetensors=None, variant="fp16")

else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/egr/research-dselab/renjie3/.cache", safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

save_dir = f"./results/{args.job_id}_{args.prompt}_{args.output_name}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

from time import time

time_counter = 0

caption_template1 = "A special billboard in a parking lot. There are a few cars parking besides it. And on the billboard, it shows "

set_seed(args.seed)

counter = 0
with open(f"./prompt/{args.prompt}.txt", 'r') as file:
    for line_id, line in enumerate(file):

        if line_id < args.start_id:
            continue
        
        # Each 'line' includes a newline character at the end, you can strip it using .strip()
        if args.use_template_caption:
            prompt = caption_template1 + line.strip()
        else:
            prompt = line.strip()
        save_name = '_'.join(prompt.split(' ')).replace('/', '<#>')

        print(prompt)
    
        args.prompt_id = line_id
        save_prefix = f"{save_dir}/{args.prompt_id}_{save_name}"

        torch.cuda.empty_cache()

        image_counter = 0

        while image_counter < args.total_image:

            with torch.no_grad():
                images = pipe(prompt, num_images_per_prompt=args.batch_size, num_inference_steps=args.num_inference_steps).images

            for gen_id in range(args.batch_size):

                image = images[gen_id]
                try:
                    save_prefix = f"{save_dir}/{args.prompt_id}_{image_counter}"
                    if args.save_name_prompt:
                        save_prefix += f"_{save_name}"
                    image.save(f"{save_prefix}.png")
                    print("image saved at: ", f"{save_prefix}.png")
                    image_counter += 1
                except:
                    print(f"save at {save_prefix} failed")
                    continue

                if image_counter >= args.total_image:
                    break
            if image_counter >= args.total_image:
                break
        if counter >= args.counter_exit:
            break
