# export EXPERIMENT_NAME="E-ASPL"
# export MODEL_PATH="./stable-diffusion/stable-diffusion-2-1-base"
# export CLEAN_TRAIN_DIR="data/n000050/set_A" 
# export CLEAN_ADV_DIR="data/n000050/set_B"
# export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/n000050_ADVERSARIAL"
# export CLASS_DIR="data/class-person"
# ------------------------- Train ASPL on set B -------------------------
# 0 2/255 4/255 8/255 6/255 32/255
# for BUDGET in 0.031 0.063 0.125; do  
export BUGET=0.05
export STYLE_WEIGHT=1
export EPOCH=20
export GPU="4,5,6,7"
export UPSCALE_GPU="0"
export MAIN_PROCESS_PORT=8838
export CUDA_VISIBLE_DEVICES=$GPU
export FINETUNE_STEP=200
export TRAIN_MODEL="SD15"

export EXPERIMENT_NAME="SimAC_transform_baseline"

# 在diffusion model 2-1上计算噪声
# export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export MODEL_PATH="CompVis/stable-diffusion-v1-4"
export HUGGINGFACE_TOKEN="***REMOVED***"
# export CLEAN_TRAIN_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_A" 
# export CLEAN_ADV_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_B"
# export OUTPUT_DIR="outputs/simac/CelebA-HQ/$EXPERIMENT_NAME"
# export CLASS_DIR="clean_class_image"

# export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/image_van_gogh_small" 
# export CLEAN_ADV_DIR="/home/yjli/AIGC/diffusers/image_van_gogh_small"
#export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/vangogh"
#export CLEAN_ADV_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/vangogh"
export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/SimAC/data/image_van_gogh_small"
export CLEAN_ADV_DIR="/home/yjli/AIGC/diffusers/SimAC/data/image_van_gogh_small"

export OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/$EXPERIMENT_NAME"
export CLASS_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/reference"

# ------------------------- Train E-ASPL on set B -------------------------
# pretrained sd models
# sd14_path="./stable-diffusion/stable-diffusion-v1-4"
export sd14_path="CompVis/stable-diffusion-v1-4"
export su_upscale_path="stabilityai/stable-diffusion-x4-upscaler"
export sd15_path="stable-diffusion-v1-5/stable-diffusion-v1-5"
# sd21_path="./stable-diffusion/stable-diffusion-2-1-base"
export sd21_path="stabilityai/stable-diffusion-2-1-base"
# ref_model_path="${sd14_path},${su_upscale_path}"
export ref_model_path="${sd14_path},${sd15_path},${su_upscale_path}"

mkdir -p $OUTPUT_DIR

export CUDA_VISIBLE_DEVICES=$GPU
# accelerate launch --num_processes=2 --config_file gpu_config.yaml --main_process_port=$MAIN_PROCESS_PORT attacks/ensemble_aspl_time.py \
#   --pretrained_model_name_or_path="${sd14_path},${su_upscale_path}" \
#   --enable_xformers_memory_efficient_attention \
#   --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
#   --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
#   --instance_prompt="a painting of sks" \
#   --class_data_dir=$CLASS_DIR \
#   --num_class_images=100 \
#   --class_prompt="a painting" \
#   --output_dir=$OUTPUT_DIR \
#   --center_crop \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --resolution=512 \
#   --train_text_encoder \
#   --train_batch_size=1 \
#   --max_train_steps=$EPOCH \
#   --max_f_train_steps=10 \
#   --max_adv_train_steps=5 \
#   --checkpointing_iterations=$EPOCH \
#   --learning_rate=5e-7 \
#   --pgd_alpha=5e-3 \
#   --pgd_eps=$BUDGET \
#   --target_image_dir="/home/yjli/AIGC/diffusers/SimAC/data/target" \
#   --style_loss_weight=$STYLE_WEIGHT


# evlaute the runing time of SIMAC
# accelerate launch --num_processes=2 --config_file gpu_config.yaml --main_process_port=$MAIN_PROCESS_PORT attacks/aspl.py \
#   --pretrained_model_name_or_path=$sd14_path  \
#   --enable_xformers_memory_efficient_attention \
#   --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
#   --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
#   --instance_prompt="a painting of sks" \
#   --class_data_dir=$CLASS_DIR \
#   --num_class_images=100 \
#   --class_prompt="a painting" \
#   --output_dir=$OUTPUT_DIR \
#   --center_crop \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --resolution=512 \
#   --train_text_encoder \
#   --train_batch_size=1 \
#   --max_train_steps=20 \
#   --max_f_train_steps=10 \
#   --max_adv_train_steps=5 \
#   --checkpointing_iterations=10 \
#   --learning_rate=5e-7 \
#   --pgd_alpha=0.005 \
#   --pgd_eps=16 \
#   --seed=0


TRAIN_MODEL="SD15"

#use scenic to compute cmmd metric
export PYTHONPATH=$PYTHONPATH:/home/yjli/AIGC/diffusers/SimAC/scenic
for CROP in 1;do
    for ROTATE in 0 1;do 
        for COMPRESS in 0 1;do
          STYLE=8
          STYLE_ENCODER="VAE"
          UPSCALE_WEIGHT=10
          K1=5
          K2=10
           export BUDGET=0.05
          export GPU="4,5"
          export EPOCH=20
          export STYLE_WEIGHT=1

          export FINETUNE_STEP=200
          # Count the number of GPUs
          NUM_GPUS=$(echo $GPU | tr ',' '\n' | wc -l)
          # Set the acceleration process count equal to the number of GPUs
          export NUM_PROCESS=$NUM_GPUS
          
          export UPSCALE_GPU="cuda:0"
          export UPSCALER="none"
          # export UPSCALER="none"

          export MAIN_PROCESS_PORT=8840
          export CUDA_VISIBLE_DEVICES=$GPU

          export EXPERIMENT_NAME="ablation_SIMAC_CROP_${CROP}_R_${ROTATE}_CP_${COMPRESS}_STYLE_${STYLE}_${TRAIN_MODEL}"

          # 在diffusion model 2-1上计算噪声
          # export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
          export MODEL_PATH="CompVis/stable-diffusion-v1-4"
          # export CLEAN_TRAIN_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_A" 
          # export CLEAN_ADV_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_B"
          # export OUTPUT_DIR="outputs/simac/CelebA-HQ/$EXPERIMENT_NAME"
          # export CLASS_DIR="clean_class_image"

          # export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/image_van_gogh_small" 
          # export CLEAN_ADV_DIR="/home/yjli/AIGC/diffusers/image_van_gogh_small"
          # export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/vangogh"
          export CLEAN_TRAIN_DIR="/home/yjli/AIGC/diffusers/SimAC/data/image_van_gogh_small"
          export CLEAN_ADV_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/vangogh"
          export OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/$EXPERIMENT_NAME/$TRAIN_MODEL"
          export CLASS_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/reference"
          export REF_FOLDER="/home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/vangogh_clean_$TRAIN_MODEL/checkpoint-1000-test-infer/an_sks_painting_including_a_house"
          # export REF_FOLDER="/home/yjli/AIGC/diffusers/SimAC/data/image_van_gogh_small"
          # export TARGET_DIR="/home/yjli/AIGC/diffusers/SimAC/data/target" 
          export TARGET_DIR="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/$STYLE" 
          # ------------------------- Train E-ASPL on set B -------------------------
          # pretrained sd models
          # sd14_path="./stable-diffusion/stable-diffusion-v1-4"
          export sd14_path="CompVis/stable-diffusion-v1-4"
          export su_upscale_path="stabilityai/stable-diffusion-x4-upscaler"
          export sd15_path="stable-diffusion-v1-5/stable-diffusion-v1-5"
          # sd21_path="./stable-diffusion/stable-diffusion-2-1-base"
          export sd21_path="stabilityai/stable-diffusion-2-1-base"
          # ref_model_path="${sd14_path},${su_upscale_path}"
          export ref_model_path="${sd14_path},${sd15_path},${su_upscale_path}"

          mkdir -p $OUTPUT_DIR

          export CUDA_VISIBLE_DEVICES=$GPU

          # ------------------------- Train DreamBooth on perturbed examples -------------------------
          export INSTANCE_DIR="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/SimAC_transform_baseline/noise-ckpt/20"

          # use a different upscale model (x2) from the training (x4)
          if [ "$UPSCALER" == "x2" ] || [ "$UPSCALER" == "x4" ]; then
              export UPSCALE_DIR="$OUTPUT_DIR/noise-upscale-$UPSCALER/"
              python Noisy_Upscaling.py \
                --input_folder=$INSTANCE_DIR \
                --output_folder=$UPSCALE_DIR \
                --upscaler=$UPSCALER \
                --step=50 \
                --device=$UPSCALE_GPU
          else
              export UPSCALE_DIR=$INSTANCE_DIR
          fi

          export TRANSFORM_DIR="$OUTPUT_DIR/noise-transform"
          python /home/yjli/AIGC/diffusers/SimAC/scripts/simple_tranformation.py \
            --input_dir  $UPSCALE_DIR \
            --output_dir $TRANSFORM_DIR \
            --crop $CROP \
            --rotate $ROTATE \
            --compress $COMPRESS \



          export DREAMBOOTH_OUTPUT_DIR="/media/ssd1/yjli/dreambooth-outputs/anti-style/$EXPERIMENT_NAME/$TRAIN_MODEL"

          # accelerate launch \
          #   --num_processes=$NUM_PROCESS \
          #   --gpu_ids=$GPU \
          #   --config_file gpu_config.yaml \
          #   --main_process_port=$MAIN_PROCESS_PORT \
          #   /home/yjli/AIGC/diffusers/examples/dreambooth/train_dreambooth.py \
          #   --pretrained_model_name_or_path=$sd15_path \
          #   --enable_xformers_memory_efficient_attention \
          #   --train_text_encoder \
          #   --instance_data_dir=$TRANSFORM_DIR \
          #   --class_data_dir=$CLASS_DIR \
          #   --output_dir=$DREAMBOOTH_OUTPUT_DIR \
          #   --with_prior_preservation \
          #   --prior_loss_weight=1.0 \
          #   --instance_prompt="a painting of sks" \
          #   --class_prompt="a painting" \
          #   --resolution=512 \
          #   --train_batch_size=1 \
          #   --gradient_accumulation_steps=1 \
          #   --gradient_checkpointing \
          #   --learning_rate=5e-6 \
          #   --lr_scheduler="constant" \
          #   --lr_warmup_steps=0 \
          #   --num_class_images=100 \
          #   --max_train_steps=$FINETUNE_STEP \
          #   --checkpointing_steps=10000 \
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
          #   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-$FINETUNE_STEP-test-infer \
          #   --prompt "an sks painting including a house"

          # python -m pytorch_fid --save-stats $REF_FOLDER /home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/vangogh_clean_SD15/output.npz
          # python -m pytorch_fid --save-stats $REF_FOLDER /home/yjli/AIGC/diffusers/SimAC/trainingdata_fid_ref.npz

          # python evaluate/eval_fid_new.py \
          #   --input_folder "$DREAMBOOTH_OUTPUT_DIR/checkpoint-$FINETUNE_STEP-test-infer/an_sks_painting_including_a_house" \
          #   --refer /home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/vangogh_clean_$TRAIN_MODEL/output.npz \
          #   --output_folder $DREAMBOOTH_OUTPUT_DIR

          # python evaluate/eval_fid_new.py \
          #   --input_folder "$DREAMBOOTH_OUTPUT_DIR/checkpoint-$FINETUNE_STEP-test-infer/an_sks_painting_including_a_house" \
          #   --refer /home/yjli/AIGC/diffusers/SimAC/trainingdata_fid_ref.npz \
          #   --output_folder $DREAMBOOTH_OUTPUT_DIR

          CUDA_VISIBLE_DEVICES=-1 python evaluate/eval_precision_new.py \
            --reference_folder $REF_FOLDER \
            --gen_folder "$DREAMBOOTH_OUTPUT_DIR/checkpoint-$FINETUNE_STEP-test-infer/an_sks_painting_including_a_house" \
            --output_folder $DREAMBOOTH_OUTPUT_DIR

          conda run -n cmmd python -m cmmd.main --ref_folder=$REF_FOLDER \
            --gen_folder=$DREAMBOOTH_OUTPUT_DIR/checkpoint-$FINETUNE_STEP-test-infer/an_sks_painting_including_a_house \
            --max_num=12 \
            --batch_size=$NUM_GPUS \
            --output_folder $DREAMBOOTH_OUTPUT_DIR

          python evaluate/eval_clip_sim.py --image_folder=$DREAMBOOTH_OUTPUT_DIR/checkpoint-$FINETUNE_STEP-test-infer/an_sks_painting_including_a_house \
            --prompt="a house painting of Van Gogh Style" \
            --output_folder $DREAMBOOTH_OUTPUT_DIR

          python evaluate/LPIPS.py \
            --ref_folder $REF_FOLDER \
            --gen_folder "$DREAMBOOTH_OUTPUT_DIR/checkpoint-$FINETUNE_STEP-test-infer/an_sks_painting_including_a_house" \
            --output_folder $DREAMBOOTH_OUTPUT_DIR

          ENV_VARS_FILE="$DREAMBOOTH_OUTPUT_DIR/env_vars.log"
          # Write environment variables to the file
          echo "Logging environment variables to $ENV_VARS_FILE"
          echo "TRAIN_MODEL=$TRAIN_MODEL" >> $ENV_VARS_FILE
          echo "BUDGET=$BUDGET" >> $ENV_VARS_FILE
          echo "EPOCH=$EPOCH" >> $ENV_VARS_FILE
          echo "STYLE_WEIGHT=$STYLE_WEIGHT" >> $ENV_VARS_FILE
          echo "FINETUNE_STEP=$FINETUNE_STEP" >> $ENV_VARS_FILE
          echo "NUM_GPUS=$NUM_GPUS" >> $ENV_VARS_FILE
          echo "EXPERIMENT_NAME=$EXPERIMENT_NAME" >> $ENV_VARS_FILE
          echo "UPSCALES_GPU=$UPSCALE_GPU" >> $ENV_VARS_FILE
          echo "K1=$K1" >> $ENV_VARS_FILE
          echo "K2=$K2" >> $ENV_VARS_FILE
          echo "NUM_PROCESS=$NUM_PROCESS" >> $ENV_VARS_FILE
          echo "UPSCLAER=$UPSCLAER" >> $ENV_VARS_FILE
          echo "MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT" >> $ENV_VARS_FILE
          echo "STYLE INDEX=$STYLE" >> $ENV_VARS_FILE
          echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> $ENV_VARS_FILE
          echo "STYLE_ENCODER=$STYLE_ENCODER" >> $ENV_VARS_FILE
          echo "STYLE=$STYLE" >> $ENV_VARS_FILE
          echo "CROP=$CROP" >> $ENV_VARS_FILE
          echo "ROTATE=$ROTATE" >> $ENV_VARS_FILE
          echo "COMPRESS=$COMPRESS" >> $ENV_VARS_FILE
        done
    done
done