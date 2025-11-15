
export EXPERIMENT_NAME="vangogh_StyleGuard_style_loss_upscaling"
export MODEL_PATH="CompVis/stable-diffusion-v1-4"

TOKEN=$(cat token.txt)  # Reads token directly
export HUGGING_FACE_HUB_TOKEN="$TOKEN"

export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/StyleGuard/data/wikiart/vangogh"
export CLEAN_ADV_DIR="/home/yjli/AIGC/diffusers/StyleGuard/data/wikiart/vangogh"
export OUTPUT_DIR="/home/yjli/AIGC/diffusers/StyleGuard/outputs/style/wikiart/$EXPERIMENT_NAME"
export CLASS_DIR="/home/yjli/AIGC/diffusers/StyleGuard/data/wikiart/reference"

export sd14_path="CompVis/stable-diffusion-v1-4"
export su_upscale_path="stabilityai/stable-diffusion-x4-upscaler"
export sd15_path="stable-diffusion-v1-5/stable-diffusion-v1-5"
export sd21_path="stabilityai/stable-diffusion-2-1-base"
export ref_model_path="${sd14_path},${sd15_path},${su_upscale_path}"

mkdir -p $OUTPUT_DIR

# excute styleguard defense to generate protective noises
accelerate launch --num_processes=4 --gpu_ids="4,5,6,7" --config_file gpu_config.yaml --main_process_port=8833  attacks/styleguard.py \
  --pretrained_model_name_or_path="${sd14_path},${su_upscale_path}" \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
  --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
  --instance_prompt="an sks painting" \
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
  --checkpointing_iterations=50 \
  --learning_rate=5e-7 \
  --pgd_alpha=5e-3 \
  --pgd_eps=5e-2 \
  --target_image_dir  /home/yjli/AIGC/diffusers/StyleGuard/data/target \
  --style_loss_weight 1

# ------------------------- Train DreamBooth on perturbed examples -------------------------
export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/50"

export INPUT_FOLDER="outputs/style/wikiart/$EXPERIMENT_NAME/noise-ckpt/50"
export UPSCALE_DIR="outputs/style/wikiart/$EXPERIMENT_NAME/noise-upscale-new/"

python Noisy_Upscaling.py \
  --input_folder=$INPUT_FOLDER \
  --output_folder=$UPSCALE_DIR \
  --upscaler="x2" \
  --step=100

export DREAMBOOTH_OUTPUT_DIR="/media/ssd1/yjli/dreambooth-outputs/anti-style/$EXPERIMENT_NAME/SD14"

accelerate launch \
  --num_processes=2 \
  --gpu_ids="1,2" \
  --config_file gpu_config.yaml \
  --main_process_port=8831 \
  dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$sd15_path \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/$EXPERIMENT_NAME/noise-upscale-new" \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="an sks painting" \
  --class_prompt="a painting" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=30 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=fp16 \
  --prior_generation_precision=fp16 \
  --sample_batch_size=1 \
  --snr_gamma=1.5

python infer.py \
  --model_path $DREAMBOOTH_OUTPUT_DIR \
  --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
  --prompt "an sks painting of flowers and trees"

# python infer.py \
#   --model_path $DREAMBOOTH_OUTPUT_DIR \
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
#   --prompt "an sks painting of trees"

# export DREAMBOOTH_OUTPUT_DIR="/media/ssd1/yjli/dreambooth-outputs/anti-style/$EXPERIMENT_NAME/SD15"

# accelerate launch \
#   --num_processes=2 \
#   --gpu_ids="1,2" \
#   --config_file gpu_config.yaml \
#   --main_process_port=8831 \
#   /home/yjli/AIGC/diffusers/examples/dreambooth/train_dreambooth.py \
#   --pretrained_model_name_or_path=$sd15_path \
#   --enable_xformers_memory_efficient_attention \
#   --train_text_encoder \
#   --instance_data_dir="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/$EXPERIMENT_NAME/noise-upscale-new" \
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --instance_prompt="an sks painting" \
#   --class_prompt="a painting" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=30 \
#   --max_train_steps=1000 \
#   --checkpointing_steps=1000 \
#   --center_crop \
#   --mixed_precision=fp16 \
#   --prior_generation_precision=fp16 \
#   --sample_batch_size=1 \
#   --snr_gamma=1.5

# python infer.py \
#   --model_path $DREAMBOOTH_OUTPUT_DIR \
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
#   --prompt "an sks painting of flowers and trees"

# python infer.py \
#   --model_path $DREAMBOOTH_OUTPUT_DIR \
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
#   --prompt "an sks painting of trees"


# export DREAMBOOTH_OUTPUT_DIR="/media/ssd1/yjli/dreambooth-outputs/anti-style/$EXPERIMENT_NAME/SD21_lora"

# accelerate launch --num_processes=2 --gpu_ids="0,1,2,3" --config_file gpu_config.yaml --main_process_port=8830 ~/AIGC/diffusers/examples/dreambooth/train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=$sd21_path  \
#   --train_text_encoder \
#   --instance_data_dir=$CLEAN_TRAIN_DIR \
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --instance_prompt="an sks painting" \
#   --class_prompt="a painting" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=30 \
#   --max_train_steps=1000 \
#   --checkpointing_steps=500 \
#   --center_crop \
#   --mixed_precision=bf16 \
#   --prior_generation_precision=bf16 \
#   --sample_batch_size=2

# python infer_lora.py \
#     --model_path $sd21_path \
#     --lora_path "$DREAMBOOTH_OUTPUT_DIR/pytorch_lora_weights.safetensors" \
#     --prompt "an sks painting, including trees" \
#     --v "sks" \
#     --img_num 8 \
#     --output_dir "$DREAMBOOTH_OUTPUT_DIR/images"
