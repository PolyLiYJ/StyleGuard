# export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export MODEL_PATH="CompVis/stable-diffusion-v1-4"
export EXPERIMENT_NAME="anti-dreambooth"
# export CLEAN_TRAIN_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_A" 
# export CLEAN_ADV_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_B"
export OUTPUT_DIR="outputs/anti-dreambooth/CelebA-HQ/$EXPERIMENT_NAME"
# export CLASS_DIR="data/class-person"
export HUGGINGFACE_TOKEN="***REMOVED***"
export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/image_van_gogh_small" 
export CLEAN_ADV_DIR="/home/yjli/AIGC/diffusers/image_van_gogh_small"
export OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/$EXPERIMENT_NAME"
export CLASS_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/reference"

# ------------------------- Train ASPL on set B -------------------------
mkdir -p $OUTPUT_DIR
cp -r $CLEAN_TRAIN_DIR $OUTPUT_DIR/image_clean
cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise
    
accelerate launch --num_processes=2 --gpu_ids="3,4" --config_file gpu_config.yaml --main_process_port=8831 attacks/aspl.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
  --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
  --instance_prompt="a sks painting" \
  --class_data_dir=$CLASS_DIR \
  --num_class_images=100 \
  --class_prompt="a painting" \
  --output_dir=$OUTPUT_DIR \
  --center_crop \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_text_encoder \
  --train_batch_size=1 \
  --max_train_steps=50 \
  --max_f_train_steps=3 \
  --max_adv_train_steps=6 \
  --checkpointing_iterations=10 \
  --learning_rate=5e-7 \
  --pgd_alpha=0.005 \
  --pgd_eps=16 \
  --seed=0

export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/50"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/anti-dreambooth/CelebA-HQ/$EXPERIMENT_NAME"

accelerate --gpu_ids="5,6" launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a painting" \
  --class_prompt="a sks painting" \
  --inference_prompt="a sks painting" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=1 \
  --seed=0
  
python infer.py \
  --model_path $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000 \
  --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer

