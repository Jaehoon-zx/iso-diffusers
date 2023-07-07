#!/bin/bash

#SBATCH --job-name=diffusers_8 # Submit a job named "example"
#SBATCH --output=log_8.txt  # 스크립트 실행 결과 std output을 저장할 파일 이름

#SBATCH --partition=a3000
#SBATCH --gres=gpu:1          # Use 1 GPU
#SBATCH --time=1-00:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=50G              # cpu memory size
#SBATCH --cpus-per-task=8       # cpu 개수

ml purge
ml load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate diffusers

# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export MODEL_NAME="stabilityai/stable-diffusion-2"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=768 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model-8" \
  --validation_epochs=4 \
  --validation_prompts "a drawing of a blue and yellow pokemon" "a drawing of a black and white pokemon" \
  --prompts_reps=8 \
