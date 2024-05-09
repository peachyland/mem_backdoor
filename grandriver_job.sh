JOB_ID=`cat job_id.log`
echo $JOB_ID
NEXT_JOB_ID=`expr $JOB_ID + 1`
echo $NEXT_JOB_ID > job_id.log

GPU_ID='6'
# TARGET_MODEL_NAME=`cat current_model_id.log`

# prompt_graduate_dup32_4_triggered prompt_template5_100 prompt_graduate_dup32_2_triggered prompt_template5_100_200 prompt_conceptual_filterwm_100 prompt_conceptual_nomem_100 prompt_graduate_dup32 prompt_conceptual_nomem_20k prompt_conceptual_filterwm_20k prompt_dup32_2_blip prompt_clip prompt_cat_200 prompt_template5_100_dup2 prompt_cartoon_all prompt_sketch_all

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# # MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# MY_CMD="python text2img.py --model_name=$MODEL_NAME --prompt prompt_graduate_dup32_2_triggered --output_name job --seed 0 --total_image 50 --batch_size 1 --seed 100 --load_unet --load_unet_path /egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/conceptual_20k_filterwm_graduate/checkpoint-7500/unet"

# # MODEL_NAME="CompVis/stable-diffusion-v1-4"
# MODEL_NAME="stabilityai/stable-diffusion-2"
# # MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# MY_CMD="python text2img.py --model_name=$MODEL_NAME --prompt prompt_train_cartoon_ours --output_name job --seed 0 --total_image 13 --batch_size 1 --seed 0 --load_unet --load_unet_job_id 512 --load_unet_ckpt 20000 --counter_exit 200"

# MY_CMD="python -u sscd_v2_group.py"

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
MODEL_NAME="stabilityai/stable-diffusion-2"
DATASET_NAME="/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/cartoon_dirty_label"
MY_CMD="accelerate launch --mixed_precision=fp16 train_text_to_image.py --pretrained_model_name_or_path=$MODEL_NAME --dataset_name=$DATASET_NAME --use_ema --resolution=512 --center_crop --train_batch_size=2 --gradient_accumulation_steps=4 --max_train_steps=20050 --checkpointing_steps=5000 --learning_rate=5e-06 --max_grad_norm=1 --lr_scheduler=constant --lr_warmup_steps=2500 --output_dir=./results/cartoon_dirty_label_sd2_lr5e-06" # template7_7_0

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# # MODEL_NAME="stabilityai/stable-diffusion-2"
# DATASET_NAME="/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_dirtylabel"
# MY_CMD="accelerate launch --mixed_precision=fp16 train_text_to_image_lora.py --pretrained_model_name_or_path=$MODEL_NAME --dataset_name=$DATASET_NAME --caption_column=text --resolution=512 --random_flip --train_batch_size=8 --num_train_epochs=20 --checkpointing_steps=5000 --learning_rate=1e-04 --lr_scheduler=constant --lr_warmup_steps=0 --seed=42 --output_dir=./results/conceptual_dirtylabel-lora --validation_prompt=dog"

# MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
# DATASET_NAME="/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm"
# MY_CMD="accelerate launch train_text_to_image_sdxl.py --pretrained_model_name_or_path=$MODEL_NAME --pretrained_vae_model_name_or_path=$VAE_NAME --enable_xformers_memory_efficient_attention --resolution=1024 --train_data_dir=$DATASET_NAME --proportion_empty_prompts=0 --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=100000 --use_8bit_adam --learning_rate=2e-05 --lr_scheduler=constant --lr_warmup_steps=0 --mixed_precision=fp16 --checkpointing_steps=10000 --output_dir=./results/sdxl_conceptual_clean"

MY_ROOT_PATH=`pwd`

echo "cd ${MY_ROOT_PATH}" > ./cmd/cmd_${JOB_ID}.sh
echo "MY_CMD=\"${MY_CMD} --job_id $JOB_ID \"" >> ./cmd/cmd_${JOB_ID}.sh
echo "CUDA_VISIBLE_DEVICES='${GPU_ID}' \${MY_CMD}" >> ./cmd/cmd_${JOB_ID}.sh
echo "if [ \$? -eq 0 ];then" >> ./cmd/cmd_${JOB_ID}.sh
echo "echo -e \"grandriver JobID:${JOB_ID} \\\n Python_command: \\\n ${MY_CMD} \\\n \" | mail -s \"[Done] grandriver ${SLURM_JOB_ID}\" renjie2179@outlook.com" >> ./cmd/cmd_${JOB_ID}.sh
echo "else" >> ./cmd/cmd_${JOB_ID}.sh
echo "echo -e \"grandriver JobID:${JOB_ID} \\\n Python_command: \\\n ${MY_CMD} \\\n \" | mail -s \"[Fail] grandriver ${SLURM_JOB_ID}\" renjie2179@outlook.com" >> ./cmd/cmd_${JOB_ID}.sh
echo "fi" >> ./cmd/cmd_${JOB_ID}.sh

nohup sh ./cmd/cmd_${JOB_ID}.sh >./logfile/${JOB_ID}.log 2>./logfile/${JOB_ID}.err &

echo $MY_CMD

date >>./history_job.log
echo ${JOB_ID}>>./history_job.log
echo "GPU_ID=${GPU_ID}">>./history_job.log
echo ${MY_CMD}>>./history_job.log
echo "---------------------------------------------------------------" >>./history_job.log
