# ------------------------- Train ASPL on set B -------------------------
export EXPERIMENT_NAME="vangogh_only_styleloss_noiseupscale"
# 在diffusion model 2-1上计算噪声
export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
# export CLEAN_TRAIN_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_A" 
# export CLEAN_ADV_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_B"
# export OUTPUT_DIR="outputs/simac/CelebA-HQ/$EXPERIMENT_NAME"
# export CLASS_DIR="clean_class_image"
# export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/mist/test/vangogh" 
export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/image_van_gogh_small"

export CLEAN_ADV_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/image_van_gogh_small"
export OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/$EXPERIMENT_NAME"
export CLASS_DIR="data/wikiart/9"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TARGET_PAINTING_DIR="data/wikiart/2"
mkdir -p $OUTPUT_DIR
cp -r $CLEAN_TRAIN_DIR $OUTPUT_DIR/image_clean
cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise

# accelerate  launch --main_process_port 0 attacks/time_feature.py \
# CUDA_VISIBLE_DEVICES=0 python attacks/time_feature.py \
# accelerate  launch attacks/time_feature_style.py \
#CUDA_VISIBLE_DEVICES=1,2,3,4 python attacks/time_feature_style.py \
accelerate launch --num_processes=4 --mixed_precision="fp16" --gpu_ids="4,5,6,7" --config_file gpu_config.yaml  --main_process_port=8889 attacks/time_feature_style.py \
  --target_image_dir=$TARGET_PAINTING_DIR \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
  --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
  --instance_prompt="an sks painting" \
  --num_class_images=16 \
  --output_dir=$OUTPUT_DIR \
  --center_crop \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_text_encoder \
  --train_batch_size=1 \
  --max_train_steps=50 \
  --max_f_train_steps=3 \
  --max_adv_train_steps=6 \
  --checkpointing_iterations=50 \
  --learning_rate=5e-7 \
  --pgd_alpha=0.005 \
  --pgd_eps=16 \
  --seed=0 \
  --class_data_dir=$CLASS_DIR \
  --with_prior_preservation \
  --class_prompt="a painting" \
  --style_loss_weight=1 \
  --noise_pred_loss_weight=0


export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/50"
# export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/simac/CelebA-HQ/$EXPERIMENT_NAME"
export DREAMBOOTH_OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/$EXPERIMENT_NAME/50"

# 在diffusion 1-4上训练
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export OUTPUT_DIR="model/saved_model_vangogh_addnoise"

accelerate launch --num_processes=4 --gpu_ids="0,1,2,3" --config_file gpu_config.yaml --main_process_port=8830 diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="an sks painting" \
  --class_prompt="a painting" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \
  --max_train_steps=500 \
  --snr_gamma=1.5
# accelerate launch --num_processes=4 --mixed_precision="fp16" --gpu_ids="4,5,6,7" --config_file gpu_config.yaml  --main_process_port=8889 train_dreambooth.py \
#   --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base"  \
#   --enable_xformers_memory_efficient_attention \
#   --train_text_encoder \
#   --instance_data_dir=$INSTANCE_DIR \
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#   --with_prior_preservation \
#   --prior_loss_weight=0.1 \
#   --instance_prompt="a sks painting" \
#   --class_prompt="a painting" \
#   --inference_prompt="a sks painting" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-7 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=32 \
#   --max_train_steps=1000 \
#   --checkpointing_steps=1000 \
#   --center_crop \
#   --mixed_precision=bf16 \
#   --prior_generation_precision=bf16 \
#   --sample_batch_size=1 \
#   --seed=0
  
python infer.py \
  --model_path $DREAMBOOTH_OUTPUT_DIR \
  --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
  --prompt "an sks painting, including a river and a house"

python infer.py \
  --model_path $DREAMBOOTH_OUTPUT_DIR \
  --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
  --prompt "a painting"

# export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/30"
# export DREAMBOOTH_OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/Wiki-Art/new"
# accelerate launch --num_processes=4 --gpu_ids="0,1,2,3" --config_file gpu_config.yaml ../examples/dreambooth/train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_PATH \
#   --instance_data_dir=$INSTANCE_DIR \
#   --instance_prompt="a sks painting" \
#   --class_data_dir=$CLASS_DIR \
#   --class_prompt="a painting" \
#   --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#   --resolution=512 \
#   --train_batch_size=4\
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=1000 \
#   --use_8bit_adam \
#   --gradient_checkpointing \
#   --set_grads_to_none \
#   --train_text_encoder \
#   --mixed_precision="bf16" \
#   --snr_gamma=5.0 \
#   --with_prior_preservation

# python infer.py \
#   --model_path $DREAMBOOTH_OUTPUT_DIR \
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
#   --prompt "a painting"