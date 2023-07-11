#!/bin/bash

#SBATCH --job-name=diffusers_8 # Submit a job named "example"
#SBATCH --output=log_8.txt  # 스크립트 실행 결과 std output을 저장할 파일 이름

#SBATCH --partition=a6000
#SBATCH --gres=gpu:0          # Use 1 GPU
#SBATCH --time=1-00:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=50G              # cpu memory size
#SBATCH --cpus-per-task=8       # cpu 개수

ml purge
ml load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate diffusers

export MODEL_NAME="stabilityai/stable-diffusion-2" #"CompVis/stable-diffusion-v1-4"
export DATASET_NAME="Ryan-sjtu/ffhq512-caption" #"facebook/winoground" #"lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=70000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="output/sd2-ffhq512-8" \
  --validation_epochs=1 \
  --validation_prompts "a photography of a happy baby" "a photography of a woman smiling" \
  --prompts_reps=8 \
  --image_column="image" \
  --caption_column="text" \
  --split="train" \
  --prediction_type="epsilon" \
  --lambda_pl=1 \

