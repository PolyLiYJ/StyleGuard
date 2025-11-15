
experiment_name="vangogh_ensemble_ASPL_style_loss"

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