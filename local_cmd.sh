
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
DATASET_NAME="/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/cartoon/images"
# stabilityai/stable-diffusion-xl-base-1.0 stabilityai/stable-diffusion-2 CompVis/stable-diffusion-v1-4

MY_CMD="python text2img.py --model_name=$MODEL_NAME --prompt prompt_template5 --output_name 182 --save_name_prompt --seed 0 --total_image 1 --batch_size 1 --seed 0 --load_unet --load_unet_path /egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/182_sdxl_templateonly7_7_0f5_lr2e-05_182/checkpoint-20000/unet"

# MY_CMD="python text2img_lora.py --prompt prompt_template5 --output_name lora --seed 0 --total_image 1 --batch_size 1 --seed 0 --load_unet --load_unet_path /egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/conceptual_20k_filterwm_templateonly5-lora/checkpoint-50000"

# MY_CMD="python text2img_common.py --model_name stabilityai/stable-diffusion-xl-base-1.0 --prompt prompt_gen_template --output_name sdxl --seed 0 --total_image 10 --batch_size 1 --seed 0"

# MY_CMD="python classifier.py --mode train --arch resnet18 --data_dir /egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/classifier_data_7_7 --test_dir /egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/190_prompt_template5_200_template5_seed0_finetune50000 --val_epoch 1 --batch_size 128 --learning_rate 0.001 --epochs 50"

# MY_CMD="accelerate launch --mixed_precision=fp16  train_text_to_image.py --pretrained_model_name_or_path=$MODEL_NAME --dataset_name=$DATASET_NAME --use_ema --resolution=512 --center_crop --crop_resize --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=50000 --checkpointing_steps=2500 --learning_rate=5e-06 --max_grad_norm=1 --lr_scheduler=constant --lr_warmup_steps=2500 --output_dir=./results/cartoon_templateonly5"

# MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
# MY_CMD="accelerate launch train_text_to_image_sdxl.py --pretrained_model_name_or_path=$MODEL_NAME --pretrained_vae_model_name_or_path=$VAE_NAME --enable_xformers_memory_efficient_attention --resolution=1024 --train_data_dir=$DATASET_NAME --proportion_empty_prompts=0.2 --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=50000 --use_8bit_adam --learning_rate=5e-06 --lr_scheduler=constant --lr_warmup_steps=0 --mixed_precision=fp16 --checkpointing_steps=10000 --output_dir=./results/sdxl"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='5' $MY_CMD # HF_HOME=$HF_CACHE_DIR TRANSFORMERS_CACHE=$HF_CACHE_DIR
