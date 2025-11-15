#!/bin/bash
export EXPERIMENT_NAME=103
export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export CLASS_DIR="/home/yjli/AIGC/diffusers/SimAC/data/CelebA-HQ/reference"
export INSTANCE_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/all"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/clean/CelebA-HQ/$EXPERIMENT_NAME"
export sd15_path="stable-diffusion-v1-5/stable-diffusion-v1-5"
export sd14_path="CompVis/stable-diffusion-v1-4"


# CUDA_VISIBLE_DEVICES=1,2,3,4 python train_dreambooth.py \
#     --pretrained_model_name_or_path=$MODEL_PATH  \
#     --enable_xformers_memory_efficient_attention \
#     --train_text_encoder \
#     --instance_data_dir=$INSTANCE_DIR \
#     --class_data_dir=$CLASS_DIR \
#     --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#     --with_prior_preservation \
#     --prior_loss_weight=1.0 \
#     --class_prompt="a photo of person" \
#     --instance_prompt="a photo of sks person" \
#     --class_prompt="a photo of person" \
#     --inference_prompt="a photo of sks person" \
#     --resolution=512 \
#     --train_batch_size=2 \
#     --gradient_accumulation_steps=1 \
#     --learning_rate=5e-7 \
#     --lr_scheduler="constant" \
#     --lr_warmup_steps=0 \
#     --num_class_images=16 \
#     --max_train_steps=1000 \
#     --checkpointing_steps=1000 \
#     --center_crop \
#     --mixed_precision=bf16 \
#     --prior_generation_precision=bf16 \
#     --sample_batch_size=1 \
#     --seed=0
  
# python infer.py \
#       --model_path /home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/clean/CelebA-HQ/103/checkpoint-1000 \
#       --output_dir /home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/clean/CelebA-HQ/103/checkpoint-1000-test-infer

export GPU="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$GPU
export EPOCH=10
# Count the number of GPUs
NUM_GPUS=$(echo $GPU | tr ',' '\n' | wc -l)
# Set the acceleration process count equal to the number of GPUs
export NUM_PROCESS=$NUM_GPUS
# export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/vangogh"
# export CLEAN_ADV_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/vangogh"
export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/SimAC/data/image_van_gogh_small"
export OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/$EXPERIMENT_NAME"
export CLASS_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/reference"

export DREAMBOOTH_OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/vangogh_clean_SD15"
accelerate launch \
    --num_processes=$NUM_PROCESS \
    --gpu_ids=$GPU \
    --config_file gpu_config.yaml \
    --main_process_port=8833 \
    /home/yjli/AIGC/diffusers/examples/dreambooth/train_dreambooth.py \
    --pretrained_model_name_or_path=$sd15_path \
    --enable_xformers_memory_efficient_attention \
    --train_text_encoder \
    --instance_data_dir=$CLEAN_TRAIN_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$DREAMBOOTH_OUTPUT_DIR \
    --with_prior_preservation \
    --prior_loss_weight=1.0 \
    --instance_prompt="a painting of sks"  \
    --class_prompt="a painting" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=100 \
    --max_train_steps=200 \
    --checkpointing_steps=200 \
    --center_crop \
    --mixed_precision=fp16 \
    --prior_generation_precision=fp16 \
    --sample_batch_size=1 \
    --snr_gamma=1.5

python infer.py \
    --model_path $DREAMBOOTH_OUTPUT_DIR \
    --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
    --prompt "an sks painting including a house" \
    --img_num 32