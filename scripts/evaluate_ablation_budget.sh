
# export REF_FOLDER="/home/yjli/AIGC/diffusers/SimAC/diffusion-model-finetuned/saved_model_vangogh/"
# export REF_FOLDER="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/Gogh_House_SD1_5"
export REF_FOLDER="/home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/vangogh_clean_SD15"

export TRAIN_FOLDER="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/vangogh"
# python -m pytorch_fid --save-stats $TRAIN_FOLDER evaluate/vangogh_training_data.npz

export PYTHONPATH=$PYTHONPATH:/home/yjli/AIGC/diffusers/SimAC/scenic
export GPU="4,5"
export CUDA_VISIBLE_DEVICES=$GPU
NUM_GPUS=$(echo $GPU | tr ',' '\n' | wc -l)

# python evaluate/eval_fid_new.py \
#     --input_folder "$REF_FOLDER" \
#     --refer evaluate/vangogh_training_data.npz \
#     --output_folder ./

# CUDA_VISIBLE_DEVICES=-1 python evaluate/eval_precision_new.py \
#     --reference_folder $TRAIN_FOLDER \
#     --gen_folder $REF_FOLDER \
#     --output_folder ./

# conda run -n cmmd python -m cmmd.main --ref_folder=$TRAIN_FOLDER \
#     --gen_folder=$REF_FOLDER \
#     --max_num=20 \
#     --batch_size=$NUM_GPUS

# for STYLE_WEIGHT in 0 0.1 1 10 100; do  
for BUDGET in 0 0.0078 0.015; do  
# for BUDGET in 0.015; do  
    STYLE_WEIGHT=10 
    # BUDGET=0.05
    echo $STYLE_WEIGHT
    echo $BUDGET
    # Count the number of GPUs
    # Set the acceleration process count equal to the number of GPUs
    export NUM_PROCESS=$NUM_GPUS
    export MAIN_PROCESS_PORT=8838

    # export EXPERIMENT_NAME="vangogh_StyleGuard_style_loss_upscaling_ablation_styleweight_${STYLE_WEIGHT}"
    export EXPERIMENT_NAME="vangogh_StyleGuard_style_loss_upscaling_ablation_budget_${BUDGET}"
    export MODEL_PATH="CompVis/stable-diffusion-v1-4"
    export HUGGINGFACE_TOKEN="***REMOVED***"
    # export CLEAN_TRAIN_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_A" 
    # export CLEAN_ADV_DIR="data/CelebA-HQ/$EXPERIMENT_NAME/set_B"
    # export OUTPUT_DIR="outputs/simac/CelebA-HQ/$EXPERIMENT_NAME"
    # export CLASS_DIR="clean_class_image"
    export OUTPUT_DIR="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/$EXPERIMENT_NAME"

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

    export CUDA_VISIBLE_DEVICES=$GPU
  
    export UPSCALE_DIR="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/$EXPERIMENT_NAME/noise-upscale-new/"

    export DREAMBOOTH_OUTPUT_DIR="/media/ssd1/yjli/dreambooth-outputs/anti-style/$EXPERIMENT_NAME/sd15"
    # python infer.py \
    #   --model_path $DREAMBOOTH_OUTPUT_DIR \
    #   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
    #   --prompt "an sks painting including a house"

    # Compute FID scoresGenerating a compatible .npz archive from a dataset
    #python -m pytorch_fid --save-stats $REF_FOLDER/checkpoint-1000-test-infer/an_sks_painting_including_a_house $REF_FOLDER/output.npz

    python evaluate/eval_fid_new.py \
     --input_folder "$DREAMBOOTH_OUTPUT_DIR/checkpoint-200-test-infer/an_sks_painting_including_a_house" \
     --refer "$REF_FOLDER/output.npz" \
     --output_folder $DREAMBOOTH_OUTPUT_DIR

    # echo $OUTPUT_DIR/checkpoint-1000-test-infer/an_sks_painting_including_a_house
    # python evaluate/eval_fid_new.py \
    #     --input_folder "$DREAMBOOTH_OUTPUT_DIR/checkpoint-200-test-infer/an_sks_painting_including_a_house" \
    #     --refer /home/yjli/AIGC/diffusers/SimAC/evaluate/vangogh_training_data.npz \
    #     --output_folder "$OUTPUT_DIR"

    CUDA_VISIBLE_DEVICES=-1 python evaluate/eval_precision_new.py \
     --reference_folder $REF_FOLDER/checkpoint-1000-test-infer/an_sks_painting_including_a_house \
     --gen_folder "$DREAMBOOTH_OUTPUT_DIR/checkpoint-200-test-infer/an_sks_painting_including_a_house" \
     --output_folder $DREAMBOOTH_OUTPUT_DIR
    # CUDA_VISIBLE_DEVICES=-1 python evaluate/eval_precision_new.py \
    #   --reference_folder  $TRAIN_FOLDER \
    #   --gen_folder "$DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer/an_sks_painting_including_a_house" \
    #   --output_folder "$OUTPUT_DIR"

    conda run -n cmmd python -m cmmd.main --ref_folder=$REF_FOLDER/checkpoint-1000-test-infer/an_sks_painting_including_a_house \
      --gen_folder=$DREAMBOOTH_OUTPUT_DIR/checkpoint-200-test-infer/an_sks_painting_including_a_house \
      --max_num=12 \
      --batch_size=$NUM_GPUS \
      --output_folder $DREAMBOOTH_OUTPUT_DIR

    # conda run -n cmmd python -m cmmd.main --ref_folder=$REF_FOLDER/checkpoint-1000-test-infer/an_sks_painting_including_a_house \
    # --gen_folder=$DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer/an_sks_painting_including_a_house \
    # --max_num=12 \
    # --batch_size=$NUM_GPUS

    python evaluate/LPIPS.py \
      --ref_folder $REF_FOLDER/checkpoint-1000-test-infer/an_sks_painting_including_a_house \
      --gen_folder "$DREAMBOOTH_OUTPUT_DIR/checkpoint-200-test-infer/an_sks_painting_including_a_house" \
      --output_folder $DREAMBOOTH_OUTPUT_DIR

    # python evaluate/LPIPS.py \
    #   --ref_folder $TRAIN_FOLDER \
    #   --gen_folder "$DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer/an_sks_painting_including_a_house" \
    #   --output_folder "$OUTPUT_DIR"

done