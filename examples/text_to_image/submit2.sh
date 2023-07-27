#!/bin/bash

#SBATCH --job-name=diff_40 # Submit a job named "example"
#SBATCH --output=log_40.txt  # 스크립트 실행 결과 std output을 저장할 파일 이름

#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1          # Use 1 GPU
#SBATCH --time=2-00:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=50G              # cpu memory size
#SBATCH --cpus-per-task=8       # cpu 개수

ml purge
ml load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate diffusers

#"CompVis/stable-diffusion-v1-4"
#"stabilityai/stable-diffusion-2"

#"lambdalabs/pokemon-blip-captions"
#"facebook/winoground" 

export MODEL_NAME="lambdalabs/sd-image-variations-diffusers"
export DATASET_NAME="Ryan-sjtu/ffhq512-caption" 

accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=140000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="output/sd2-ffhq512-40" \
  --validation_epochs=1 \
  --validation_steps=5000 \
  --validation_prompts "a photography of a baby in a blue blanket with a stuffed animal" "a photography of a little girl with a toothbrush in her mouth" \
  --prompts_reps=2 \
  --image_column="image" \
  --caption_column="text" \
  --split="train" \
  --lambda_pl=1e-5 \
  --num_inference_steps=10 \

