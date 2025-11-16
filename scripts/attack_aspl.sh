# export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export MODEL_PATH="CompVis/stable-diffusion-v1-4"
export EXPERIMENT_NAME="anti-dreambooth"
export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/StyleGuard/data/wikiart/vangogh" 
export CLEAN_ADV_DIR="/home/yjli/AIGC/diffusers/StyleGuard/data/wikiart/vangogh"
export OUTPUT_DIR="/home/yjli/AIGC/diffusers/StyleGuard/outputs/style/wikiart/$EXPERIMENT_NAME"
export CLASS_DIR="/home/yjli/AIGC/diffusers/StyleGuard/data/wikiart/reference"
TOKEN=$(cat token.txt)  # Reads token directly
export HUGGING_FACE_HUB_TOKEN="$TOKEN"
# ------------------------- Train ASPL on set B -------------------------
mkdir -p $OUTPUT_DIR
export CUDA_VISIBLE_DEVICES="4,5,6,7"
    
accelerate launch --num_processes=4 --gpu_ids="0,1,2,3" --config_file gpu_config.yaml --main_process_port=8835 attacks/aspl.py \
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
  --pgd_eps=0.05 \
  --seed=0

# ------------------------- Train DreamBooth on perturbed examples -------------------------
export INPUT_FOLDER="$OUTPUT_DIR/noise-ckpt/50"
export UPSCALE_DIR="$OUTPUT_DIR/noise-upscale-new/"

# use a different upscaling method
python Noisy_Upscaling.py \
  --input_folder=$INPUT_FOLDER \
  --output_folder=$UPSCALE_DIR \
  --upscaler="x2" \
  --step=100

export DREAMBOOTH_OUTPUT_DIR="/media/ssd1/yjli/dreambooth-outputs/styleguard/$EXPERIMENT_NAME/SD14"
export CUDA_VISIBLE_DEVICES="4,5,6,7"

accelerate launch \
  --num_processes=4 \
  --gpu_ids="4,5,6,7" \
  --config_file gpu_config.yaml \
  --main_process_port=8838 \
  ../../diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$UPSCALE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a painting" \
  --class_prompt="a sks painting" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=30 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=1 \
  --seed=0 \
  --snr_gamma=1.5
  
python infer.py \
  --model_path $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000 \
  --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
  --prompt "an sks painting including blue sky and mountains"

