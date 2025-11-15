# export EXPERIMENT_NAME="E-ASPL"
# export MODEL_PATH="./stable-diffusion/stable-diffusion-2-1-base"
# export CLEAN_TRAIN_DIR="data/n000050/set_A" 
# export CLEAN_ADV_DIR="data/n000050/set_B"
# export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/n000050_ADVERSARIAL"
# export CLASS_DIR="data/class-person"
# ------------------------- Train ASPL on set B -------------------------
# export EXPERIMENT_NAME="vangogh_ensemble_ASPL_style_loss_upscaling"
export EXPERIMENT_NAME="antidreambooth"
export MODEL_PATH="CompVis/stable-diffusion-v1-4"
export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/image_van_gogh_small" 
export CLEAN_ADV_DIR="/home/yjli/AIGC/diffusers/image_van_gogh_small"
export OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/$EXPERIMENT_NAME"
export CLASS_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/reference"

# ------------------------- Train E-ASPL on set B -------------------------
# pretrained sd models
# sd14_path="./stable-diffusion/stable-diffusion-v1-4"
sd14_path="CompVis/stable-diffusion-v1-4"
su_upscale_path="stabilityai/stable-diffusion-x4-upscaler"
# sd15_path="./stable-diffusion/stable-diffusion-v1-5"
# sd21_path="./stable-diffusion/stable-diffusion-2-1-base"
sd21_path="stabilityai/stable-diffusion-2-1-base"
ref_model_path="${sd14_path}"

mkdir -p $OUTPUT_DIR
cp -r $CLEAN_TRAIN_DIR $OUTPUT_DIR/image_clean
cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise

accelerate launch --num_processes=2 --gpu_ids="6,7" --config_file gpu_config.yaml --main_process_port=8831  attacks/ensemble_aspl.py \
  --pretrained_model_name_or_path=${ref_model_path} \
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
  --target_image_dir  /home/yjli/AIGC/diffusers/SimAC/CelebA-HQ/5/set_A \
  --style_loss_weight 0

# ------------------------- Train DreamBooth on perturbed examples -------------------------
export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/50"
export DREAMBOOTH_OUTPUT_DIR="/media/ssd1/yjli/dreambooth-outputs/anti-style/$EXPERIMENT_NAME/"
# step=100
# export DREAMBOOTH_OUTPUT_DIR="/media/ssd1/yjli/dreambooth-outputs/anti-style/vangogh_ensemble_ASPL_style_loss_upscaling_step${step}/"


# export INSTANCE_DIR="outputs/style/wikiart/$EXPERIMENT_NAME/noise-upscaling"
# python Noisy_Upscaling.py \
#   --input_folder="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/vangogh_ensemble_ASPL_style_loss_upscaling/image_before_addding_noise" \
#   --output_folder=$INSTANCE_DIR \
#   --upscaler="flux" \
#   --step=step

#accelerate launch train_dreambooth.py \
# accelerate launch --num_processes=2 --gpu_ids="0,1,2,3,4" --config_file gpu_config.yaml --main_process_port=8830 /home/yjli/AIGC/diffusers/examples/dreambooth/train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_PATH  \
#   --enable_xformers_memory_efficient_attention \
#   --train_text_encoder \
#   --instance_data_dir=$INSTANCE_DIR \
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
#   --sample_batch_size=8 \
#   --snr_gamma=1.5

# accelerate launch --gpu_ids="5,6" --num_processes=2 --config_file gpu_config.yaml --main_process_port=8830 /home/yjli/AIGC/diffusers/examples/dreambooth/train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_PATH  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --instance_prompt="an sks painting" \
#   --class_prompt="a painting" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=2 \
#   --gradient_checkpointing \
#   --use_8bit_adam \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=50 \
#   --max_train_steps=500 \
#   --snr_gamma=1.5


# python infer.py \
#   --model_path $DREAMBOOTH_OUTPUT_DIR \
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
#   --prompt "an sks painting, including a basket and some apples"

# export DREAMBOOTH_OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/vangogh_ensemble_ASPL_style_loss_upscaling/checkpoint-1000-test-infer"
# python infer.py \
#   --model_path /home/yjli/AIGC/diffusers/model/saved_model_vangogh\
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer-normal \
#   --prompt "a sks painting, including a river and a house"
# python infer.py \
#   --model_path /home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/vangogh_ensemble_ASPL_style_loss_upscaling \
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
#   --prompt "a sks painting, including a river and a house"


# step=50
# # export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/anti-style/vangogh_ensemble_ASPL_style_loss_upscaling_step${step}/"
# # export INSTANCE_DIR="outputs/style/wikiart/$EXPERIMENT_NAME/noise-upscaling-step${step}"

# export DREAMBOOTH_OUTPUT_DIR="/media/ssd1/yjli/dreambooth-outputs/anti-style/vangogh_ensemble_ASPL_style_loss_upscaling_x2/"
# export INSTANCE_DIR="outputs/style/wikiart/$EXPERIMENT_NAME/noise-upscaling-x2"


# python Noisy_Upscaling.py \
#   --input_folder="outputs/style/wikiart/$EXPERIMENT_NAME/noise-ckpt/50" \
#   --output_folder=$INSTANCE_DIR \
#   --upscaler="x2" \
#   --step=${step}

# accelerate launch train_dreambooth.py \
accelerate launch --num_processes=2 --gpu_ids="6,7" --config_file gpu_config.yaml --main_process_port=8830 /home/yjli/AIGC/diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="an sks painting" \
  --class_prompt="a painting" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=500 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8 \
  --snr_gamma=1.5


# python infer.py \
#   --model_path $DREAMBOOTH_OUTPUT_DIR \
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
#   --prompt "an sks painting, including a basket and some apples"

# export INSTANCE_DIR="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/vangogh_with_styleloss_SD14/image_clean_crop"
# export DREAMBOOTH_OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/vangogh_clean"

# accelerate launch --num_processes=2 --gpu_ids="0,1,2,3,4,5,6,7" --config_file gpu_config.yaml --main_process_port=8830 /home/yjli/AIGC/diffusers/examples/dreambooth/train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_PATH  \
#   --enable_xformers_memory_efficient_attention \
#   --train_text_encoder \
#   --instance_data_dir=$INSTANCE_DIR \
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --instance_prompt="a sks painting" \
#   --class_prompt="a painting" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=30 \
#   --max_train_steps=500 \
#   --checkpointing_steps=500 \
#   --center_crop \
#   --mixed_precision=bf16 \
#   --prior_generation_precision=bf16 \
#   --sample_batch_size=8 \
#   --snr_gamma=1.5


python infer.py \
  --model_path $DREAMBOOTH_OUTPUT_DIR \
  --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
  --prompt "an sks painting, including a blue sky and mountains"

# python infer.py \
#   --model_path $DREAMBOOTH_OUTPUT_DIR \
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
#   --prompt "a painting, including a blue sky and mountains"

# python infer.py \
#   --model_path $DREAMBOOTH_OUTPUT_DIR \
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
#   --prompt "an sks painting, including river and house"

# python infer.py \
#   --model_path $DREAMBOOTH_OUTPUT_DIR \
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
#   --prompt "a painting, including river and house"


export INPUT_FOLDER="/home/yjli/AIGC/diffusers/Gogh_House/Gogh_House_SD1_5"
experiment_name="antidreambooth"
python infer.py \
  --model_path dreambooth-outputs/anti-style/${experiment_name} \
  --output_dir evaluate/${experiment_name}-infer/ \
  --prompt "an sks painting, inclusing a house" \
  --img_num 20

python evaluate/eval_fid_new.py \
  --input_folder "evaluate/${experiment_name}-infer/an_sks_painting_inclusing_a_house" \
  --refer '/home/yjli/AIGC/diffusers/SimAC/evaluate/clean-infer-fid-ref.npz'

python evaluate/eval_precision_new.py \
  --reference_folder $INPUT_FOLDER \
  --gen_folder "evaluate/${experiment_name}-infer/an_sks_painting_inclusing_a_house"

step=50
export INSTANCE_DIR="outputs/style/wikiart/$EXPERIMENT_NAME/noise-upscaling-step${step}"
python Noisy_Upscaling.py \
  --input_folder="outputs/style/wikiart/$EXPERIMENT_NAME/noise-ckpt/50" \
  --output_folder=$INSTANCE_DIR \
  --upscaler="x4" \
  --step=${step}

experiment_name="antidreambooth-upscale-x4"
export DREAMBOOTH_OUTPUT_DIR="/media/ssd1/yjli/dreambooth-outputs/anti-style/${experiment_name}/"
export MODEL_PATH="CompVis/stable-diffusion-v1-4"
export CLASS_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/reference"

accelerate launch --num_processes=2 --gpu_ids="6,7" --config_file gpu_config.yaml --main_process_port=8830 /home/yjli/AIGC/diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="an sks painting" \
  --class_prompt="a painting" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=500 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8 \
  --snr_gamma=1.5

export INPUT_FOLDER="/home/yjli/AIGC/diffusers/Gogh_House/Gogh_House_SD1_5"
python infer.py \
  --model_path dreambooth-outputs/anti-style/${experiment_name} \
  --output_dir evaluate/${experiment_name}-infer/ \
  --prompt "an sks painting, inclusing a house" \
  --img_num 20

python evaluate/eval_fid_new.py \
  --input_folder "evaluate/${experiment_name}-infer/an_sks_painting_inclusing_a_house" \
  --refer '/home/yjli/AIGC/diffusers/SimAC/evaluate/clean-infer-fid-ref.npz'

python evaluate/eval_precision_new.py \
  --reference_folder $INPUT_FOLDER \
  --gen_folder "evaluate/${experiment_name}-infer/an_sks_painting_inclusing_a_house"
