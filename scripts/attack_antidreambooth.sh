
experiment_name="antidreambooth-upscale-x4"
export EXPERIMENT_NAME="antidreambooth"
step=50

export INSTANCE_DIR="outputs/style/wikiart/$EXPERIMENT_NAME/noise-upscaling-step${step}"

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
