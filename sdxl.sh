export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

CUDA_VISIBLE_DEVICES='3' accelerate launch train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=1024 \
  --train_data_dir=/egr/research-dselab/renjie3/renjie/temp/pokemon/pokemon_dfd_1024 \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=50000 \
  --use_8bit_adam \
  --learning_rate=5e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --checkpointing_steps=10000 \
  --output_dir="./results/sdxl"
    
    # --report_to="wandb" \
    # --dataset_name=$DATASET_NAME \
