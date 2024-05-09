import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add arguments
parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4", help="an integer to be processed")
parser.add_argument("--local", type=str, default='', help="The scale of noise offset.")
parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars", help="The scale of noise offset.")
parser.add_argument("--job_id", type=str, default='local', help="The scale of noise offset.")
parser.add_argument("--output_name", type=str, default='local', help="The scale of noise offset.")
parser.add_argument("--counter_exit", default=-1, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--total_image", default=32, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--num_inference_steps", default=50, type=int)
parser.add_argument("--use_template_caption", action="store_true", default=False)
parser.add_argument("--load_unet", action="store_true", default=False)
parser.add_argument("--save_name_prompt", action="store_true", default=False)
parser.add_argument("--load_unet_path", type=str, default='')
parser.add_argument("--start_id", default=0, type=int)
parser.add_argument("--load_unet_job_id", type=str, default=None)
parser.add_argument("--load_unet_ckpt", type=str, default=None)

# Parse the arguments
args = parser.parse_args()

import os

if args.local != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler, UNet2DConditionModel, DiffusionPipeline
import numpy as np
import random
import json

if args.load_unet and args.load_unet_job_id is not None and args.load_unet_ckpt is not None:
    with open("./results/dict_jobid.json", 'r') as file:
        jobid_dict = json.load(file)
    try:
        jobid_folder_name = jobid_dict[args.load_unet_job_id]
        args.load_unet_path = f"./results/{jobid_folder_name}/checkpoint-{args.load_unet_ckpt}/unet"
        if not os.path.isdir(args.load_unet_path):
            raise("Wrong job id")
        args.output_name = args.load_unet_job_id
    except:
        raise("Wrong")

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
    if args.load_unet:
        unet = UNet2DConditionModel.from_pretrained(
            args.load_unet_path, torch_dtype=torch.float16
        )
        pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/egr/research-dselab/renjie3/.cache", unet=unet, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/egr/research-dselab/renjie3/.cache", scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)
if args.model_name == "stabilityai/stable-diffusion-xl-base-1.0":
    if args.load_unet:
        unet = UNet2DConditionModel.from_pretrained(args.load_unet_path, torch_dtype=torch.float16)
        pipe = DiffusionPipeline.from_pretrained(model_id, unet=unet, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", cache_dir="/localscratch/renjie/cache")
    else:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", cache_dir="/localscratch/renjie/cache")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
else:
    if args.load_unet:
        # /egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/template1/checkpoint-15000/unet_ema
        unet = UNet2DConditionModel.from_pretrained(
            args.load_unet_path, torch_dtype=torch.float16
        )
        pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/egr/research-dselab/renjie3/.cache", unet=unet, safety_checker=None, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/egr/research-dselab/renjie3/.cache", safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

save_dir = f"./results/{args.job_id}_{args.prompt}_{args.output_name}_seed{args.seed}"
if args.load_unet:
    step = args.load_unet_path.split("checkpoint-")[1].split('/')[0]
    job_id = args.load_unet_path.split("results/")[1].split('_')[0]
    save_dir += f"_{job_id}_finetune{step}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

from time import time

time_counter = 0

# caption_template1 = "A special billboard in a parking lot. There are a few cars parking besides it. And on the billboard, it shows "
# caption_template1 = "A special billboard in a parking lot."
# caption_template1 = "A billboard "

caption_template1 = " Poster by J Alsina"

set_seed(args.seed)

counter = 0
with open(f"./prompt/{args.prompt}.txt", 'r') as file:
    for line_id, line in enumerate(file):

        if line_id < args.start_id:
            continue
        
        # Each 'line' includes a newline character at the end, you can strip it using .strip()
        if args.use_template_caption:
            prompt = line.strip() + caption_template1
        else:
            prompt = line.strip()
        save_name = '_'.join(prompt.split(' ')).replace('/', '<#>')#.replace('\u', '<@>').replace('\\', '<@>')

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
                    image_counter += 1
                    break

                if image_counter >= args.total_image:
                    break
            if image_counter >= args.total_image:
                break
        
        counter += 1
        if counter == args.counter_exit:
            break
