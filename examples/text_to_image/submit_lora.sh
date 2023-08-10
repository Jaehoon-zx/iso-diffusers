#!/bin/bash

#SBATCH --job-name=diff_75 # Submit a job named "example"
#SBATCH --output=logs/log_75.txt  # 스크립트 실행 결과 std output을 저장할 파일 이름

#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1          # Use 1 GPU
#SBATCH --time=2-00:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=50G              # cpu memory size
#SBATCH --cpus-per-task=8       # cpu 개수

ml purge
ml load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate diffusers

export MODEL_NAME="segmind/tiny-sd"
export DATASET_NAME="cr7Por/ffhq_controlnet_5_2_23"

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=50000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="output/sd2-ffhq512-75" \
  --validation_epochs=1 \
  --validation_prompt "there is a baby sitting in the grass chewing on a toy" \
  --image_column="image" \
  --caption_column="image_caption" \
