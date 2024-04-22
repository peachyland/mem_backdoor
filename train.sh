export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm_graduate"

CUDA_VISIBLE_DEVICES='1' accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=50000 \
  --checkpointing_steps=2500 \
  --learning_rate=5e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=2500 \
  --output_dir="./results/conceptual_20k_filterwm_graduate"


# random_flip
