JOB_ID=`cat job_id.log`
echo $JOB_ID
NEXT_JOB_ID=`expr $JOB_ID + 1`
echo $NEXT_JOB_ID > job_id.log

GPU_ID='7'
# TARGET_MODEL_NAME=`cat current_model_id.log`

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# # MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# MY_CMD="python text2img.py --model_name=$MODEL_NAME --prompt prompt_graduate_dup32_4 --output_name graduate --seed 0 --total_image 25 --batch_size 5 --seed 0 --load_unet --load_unet_path /egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/conceptual_20k_filterwm_graduate/checkpoint-20000/unet"

# MY_CMD="python -u sscd_v2_group.py"

MODEL_NAME="CompVis/stable-diffusion-v1-4"
# MODEL_NAME="stabilityai/stable-diffusion-2"
DATASET_NAME="/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/ft_shield/roco_ft/train"
MY_CMD="accelerate launch --mixed_precision=fp16  train_text_to_image.py --pretrained_model_name_or_path=$MODEL_NAME --dataset_name=$DATASET_NAME --use_ema --resolution=512 --center_crop --train_batch_size=2 --gradient_accumulation_steps=4 --max_train_steps=20050 --checkpointing_steps=2500 --learning_rate=5e-06 --max_grad_norm=1 --lr_scheduler=constant --lr_warmup_steps=2500 --output_dir=./results/roco_ft_20k_sd1_lr5e-06"

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# # MODEL_NAME="stabilityai/stable-diffusion-2"
# DATASET_NAME="/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_dirtylabel"
# MY_CMD="accelerate launch --mixed_precision=fp16 train_text_to_image_lora.py --pretrained_model_name_or_path=$MODEL_NAME --dataset_name=$DATASET_NAME --caption_column=text --resolution=512 --random_flip --train_batch_size=8 --num_train_epochs=20 --checkpointing_steps=5000 --learning_rate=1e-04 --lr_scheduler=constant --lr_warmup_steps=0 --seed=42 --output_dir=./results/conceptual_dirtylabel-lora --validation_prompt=dog"

# MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
# DATASET_NAME="/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_20k_filterwm_templateonly7_7_0f5"
# MY_CMD="accelerate launch train_text_to_image_sdxl.py --pretrained_model_name_or_path=$MODEL_NAME --pretrained_vae_model_name_or_path=$VAE_NAME --enable_xformers_memory_efficient_attention --resolution=1024 --train_data_dir=$DATASET_NAME --proportion_empty_prompts=0.03 --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=100000 --use_8bit_adam --learning_rate=2e-05 --lr_scheduler=constant --lr_warmup_steps=0 --mixed_precision=fp16 --checkpointing_steps=10000 --output_dir=./results/182_sdxl_templateonly7_7_0f5_lr2e-05_182 --resume_from_checkpoint /egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/182_sdxl_templateonly7_7_0f5_lr2e-05_182/checkpoint-20000"

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
