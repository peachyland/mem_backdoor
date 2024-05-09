import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoencoderKL
import torch
from PIL import Image
from torchvision import transforms
import random
import numpy as np

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

my_transform = transforms.Compose([
    # transforms.RandomResizedCrop(128),
    transforms.ElasticTransform(),
])

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", cache_dir="/localscratch/renjie/cache", torch_dtype=torch.float16).to("cuda")

def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35)
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)

    return Image.fromarray(image_array)

def decode_tensors(pipe, step, timestep, callback_kwargs):

    if step == 10:

        # set_seed(30)
        latents = callback_kwargs["latents"]
        # vae.encode(temp).latent_dist.sample()
        # vae.decode(latents).sample
        # callback_kwargs["latents"] = my_transform(latents)
        temp = vae.decode(latents).sample
        callback_kwargs["latents"] = vae.encode(temp).latent_dist.sample()
        # latents_to_rgb(latents).save("./test_0.png")
        # latents_to_rgb(my_transform(latents)).save("./test_1.png")
        # import pdb ; pdb.set_trace()

    return callback_kwargs

pipeline = DiffusionPipeline.from_pretrained(
    # "stabilityai/stable-diffusion-xl-base-1.0",
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    variant="fp16",
    # use_safetensors=True,
    cache_dir="/localscratch/renjie/cache", 
).to("cuda")

set_seed(0)

image = pipeline(
    prompt = "A croissant shaped like a cute bear.",
    # negative_prompt = "Deformed, ugly, bad anatomy",
    callback_on_step_end=decode_tensors,
    callback_on_step_end_tensor_inputs=["latents"],
).images[0]

image.save(f"./test.png")

