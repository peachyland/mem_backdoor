export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm_templateonly5"

CUDA_VISIBLE_DEVICES='2' accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=8 \
  --num_train_epochs=20 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="./results/conceptual_20k_filterwm_templateonly5-lora" \
  --validation_prompt="cute dragon creature"