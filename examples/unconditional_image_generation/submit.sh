#!/bin/bash

#SBATCH --job-name=diffuser_2 # Submit a job named "example"
#SBATCH --output=log_2.txt  # 스크립트 실행 결과 std output을 저장할 파일 이름

#SBATCH --partition=a3000
#SBATCH --gres=gpu:1          # Use 1 GPU
#SBATCH --time=1-00:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=50G              # cpu memory size
#SBATCH --cpus-per-task=8       # cpu 개수

ml purge
ml load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate diffusers

accelerate launch train_unconditional.py \
  --dataset_name="huggan/pokemon" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="ddpm-ema-pokemon-64" \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  # --push_to_hub