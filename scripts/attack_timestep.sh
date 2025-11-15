export EXPERIMENT_NAME=5
export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export CLEAN_TRAIN_DIR="CelebA-HQ/$EXPERIMENT_NAME/set_A" 
export CLEAN_ADV_DIR="CelebA-HQ/$EXPERIMENT_NAME/set_B"
export OUTPUT_DIR="outputs/simac/CelebA-HQ/$EXPERIMENT_NAME"
export CLASS_DIR="/home/yjli/AIGC/diffusers/SimAC/data/class-person"
# export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/image_van_gogh_small" 
# export CLEAN_ADV_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/image_van_gogh_small"
# export OUTPUT_DIR="outputs/simac/wikiart/vangogh"
# export CLASS_DIR="data/wikiart/9"
export DREAMBOOTH_OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/clean/CelebA-HQ/$EXPERIMENT_NAME/new"
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=5,6
#   --class_data_dir="images/clean_class_image" \
#   --class_prompt="a photo of person" \
#  --with_prior_preservation \
# --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \

accelerate launch --num_processes=4 --config_file gpu_config.yaml ../examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --instance_data_dir=$CLEAN_TRAIN_DIR \
  --instance_prompt="a photo of ASDF person" \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a photo of person" \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=4\
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --set_grads_to_none \
  --train_text_encoder \
  --mixed_precision="bf16" \
  --snr_gamma=5.0 \
  --with_prior_preservation


# accelerate launch --num_processes=4 --config_file gpu_config.yaml train_dreambooth.py \
#   --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base"  \
#   --enable_xformers_memory_efficient_attention \
#   --train_text_encoder \
#   --instance_data_dir=$CLEAN_TRAIN_DIR \
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --instance_prompt="a painting by ASDF" \
#   --class_prompt="a painting" \
#   --inference_prompt="a painting by ASDF" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-7 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=36 \
#   --max_train_steps=300 \
#   --checkpointing_steps=300 \
#   --mixed_precision=bf16 \
#   --prior_generation_precision=bf16 \
#   --sample_batch_size=1 \
#   --seed=0
# --center_crop \ 

python infer.py \
  --model_path "$DREAMBOOTH_OUTPUT_DIR" \
  --output_dir "$DREAMBOOTH_OUTPUT_DIR/infer" \
  --prompt "a photo of sks person"

# # ------------------------- Train ASPL on set B -------------------------
# mkdir -p $OUTPUT_DIR
# cp -r $CLEAN_TRAIN_DIR $OUTPUT_DIR/image_clean
# cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise

# # accelerate  launch --main_process_port 0 attacks/time_feature.py \
# # CUDA_VISIBLE_DEVICES=0 python attacks/time_feature.py \
# accelerate  launch attacks/time_feature.py \
#   --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
#   --enable_xformers_memory_efficient_attention \
#   --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
#   --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
#   --instance_prompt="a photo of sks person" \
#   --class_data_dir="clean_class_image" \
#   --num_class_images=200 \
#   --class_prompt="a photo of person" \
#   --output_dir=$OUTPUT_DIR \
#   --center_crop \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --resolution=512 \
#   --train_text_encoder \
#   --train_batch_size=1 \
#   --max_train_steps=50 \
#   --max_f_train_steps=3 \
#   --max_adv_train_steps=6 \
#   --checkpointing_iterations=10 \
#   --learning_rate=5e-7 \
#   --pgd_alpha=0.005 \
#   --pgd_eps=16 \
#   --seed=0

# export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/50"
# export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/simac/CelebA-HQ/$EXPERIMENT_NAME"

# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base"  \
#   --enable_xformers_memory_efficient_attention \
#   --train_text_encoder \
#   --instance_data_dir=$INSTANCE_DIR \
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --instance_prompt="a photo of sks person" \
#   --class_prompt="a photo of person" \
#   --inference_prompt="a photo of sks person" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-7 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=200 \
#   --max_train_steps=1000 \
#   --checkpointing_steps=1000 \
#   --center_crop \
#   --mixed_precision=bf16 \
#   --prior_generation_precision=bf16 \
#   --sample_batch_size=1 \
#   --seed=0
  
# python infer.py \
#   --model_path $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000 \
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer