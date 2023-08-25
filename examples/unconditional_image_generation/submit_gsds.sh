#!/bin/bash

#SBATCH --job-name=diff_8                    # Submit a job named "example"
#SBATCH --output=logs/log_8.txt  # 스크립트 실행 결과 std output을 저장할 파일 이름

#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                          # Using 1 gpu
#SBATCH --time=0-12:00:00                     # 1 hour timelimit
#SBATCH --mem=30GB                        # Using 10GB CPU Memory
#SBATCH --partition=P2                         # Using "b" partition 
#SBATCH --cpus-per-task=4                     # Using 4 maximum processor

# source ${USER}/.bashrc
# source ${USER}/anaconda3/bin/activate
# conda activate diffusers

export MODEL_NAME="google/ddpm-ema-celebahq-256" #"simlightvt/ddpm-celebahq-128"
export DATASET_NAME="mattymchen/celeba-hq"

accelerate launch --mixed_precision="fp16" train_unconditional.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=1 \
  --num_epochs=20 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="output/celebahq-8" \
  --split="train" \
  --lambda_iso=0 \
  --ddpm_num_inference_steps=50 \
  --fid \
  # --subfolder \
  # --ppl \
  # --dists \
  # --resume_from_checkpoint="output/celebahq-3/checkpoint-120000" \