#!/bin/bash

#SBATCH --job-name=diffuser_video # Submit a job named "example"
#SBATCH --output=log_video.txt  # 스크립트 실행 결과 std output을 저장할 파일 이름

#SBATCH --partition=a100
#SBATCH --gres=gpu:a100.5gb:1          # Use 1 GPU
#SBATCH --time=1-00:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=10G              # cpu memory size
#SBATCH --cpus-per-task=4       # cpu 개수

ml purge
ml load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate diffusers

accelerate launch latent_interpolation.py \
