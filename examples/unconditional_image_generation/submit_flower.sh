#!/bin/bash

#SBATCH --job-name=diffuser_7 # Submit a job named "example"
#SBATCH --output=logs/log_7.txt  # 스크립트 실행 결과 std output을 저장할 파일 이름

#SBATCH --partition=a100
#SBATCH --gres=gpu:a100.10gb:1          # Use 1 GPU
#SBATCH --time=1-00:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=50G              # cpu memory size
#SBATCH --cpus-per-task=8       # cpu 개수

ml purge
ml load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate diffusers

export MODEL_NAME="anton-l/ddpm-ema-flowers-64"
export DATASET_NAME="huggan/flowers-102-categories"

accelerate launch --mixed_precision="fp16" train_unconditional.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=64 --center_crop --random_flip \
  --train_batch_size=4 \
  --num_epochs=20 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="output/celebahq-7" \
  --split="train" \
  --lambda_iso=1e-5 \
  --ddpm_num_inference_steps=50 \
  --subfolder \
  # --resume_from_checkpoint="output/celebahq-3/checkpoint-120000" \
  # --prompts_reps=4 \
#   --ppl \
#   --fid \
#   --dists \