export INPUT_FOLDER="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/vangogh_sunflowers"
export GEN_FLODER="/home/yjli/AIGC/diffusers/SimAC/evaluate/clean-infer"

# python infer.py \
#   --model_path /media/ssd1/yjli/diffusion_model/saved_model_vangogh \
#   --output_dir $GEN_FLODER \
#   --prompt "an sks painting, inclusing a vase and sunflowers" \
#   --img_num 20

## generate FID reference on the original test dataset
# python -m pytorch_image_generation_metrics.fid_ref \
#     --path $INPUT_FOLDER \
#     --output /home/yjli/AIGC/diffusers/SimAC/evaluate/fid_ref.npz

export GEN_FLODER="/home/yjli/AIGC/diffusers/SimAC/evaluate/clean-infer/an_sks_painting_inclusing_a_vase_and_sunflowers"
python evaluate/eval_fid_new.py \
  --input_folder $GEN_FLODER \
  --refer '/home/yjli/AIGC/diffusers/SimAC/evaluate/fid_ref.npz'

python evaluate/eval_precision_new.py \
  --reference_folder "/home/yjli/AIGC/diffusers/SimAC/data/wikiart/vangogh_sunflowers" \
  --gen_folder $GEN_FLODER

experiment_name="mist"

# python infer.py \
#   --model_path dreambooth-outputs/anti-style/${experiment_name} \
#   --output_dir evaluate/${experiment_name}-infer/ \
#   --prompt "an sks painting, inclusing a vase and sunflowers" \
#   --img_num 20

python evaluate/eval_fid_new.py \
  --input_folder "evaluate/mist-infer/an_sks_painting_inclusing_a_vase_and_sunflowers" \
  --refer '/home/yjli/AIGC/diffusers/SimAC/evaluate/fid_ref.npz'

python evaluate/eval_precision_new.py \
  --reference_folder "evaluate/clean-infer/an_sks_painting_inclusing_a_vase_and_sunflowers/" \
  --gen_folder "evaluate/mist-infer/an_sks_painting_inclusing_a_vase_and_sunflowers"


experiment_name="mist-upscale"
# python infer.py \
#   --model_path /media/ssd1/yjli/dreambooth-outputs/anti-style/${experiment_name} \
#   --output_dir evaluate/${experiment_name}-infer/ \
#   --prompt "an sks painting, inclusing a vase and sunflowers" \
#   --img_num 20

python evaluate/eval_fid_new.py \
  --input_folder "evaluate/${experiment_name}-infer/an_sks_painting_inclusing_a_vase_and_sunflowers" \
  --refer '/home/yjli/AIGC/diffusers/SimAC/evaluate/fid_ref.npz'

python evaluate/eval_precision_new.py \
  --reference_folder "evaluate/clean-infer/an_sks_painting_inclusing_a_vase_and_sunflowers/" \
  --gen_folder "evaluate/${experiment_name}-infer/an_sks_painting_inclusing_a_vase_and_sunflowers"

# python infer.py \
#   --model_path /media/ssd1/yjli/dreambooth-outputs/anti-style/vangogh_ensemble_ASPL_style_loss_upscaling \
#   --output_dir evaluate/styleguard-infer/ \
#   --prompt "an sks painting, inclusing a vase and sunflowers" \
#   --img_num 20

python evaluate/eval_fid_new.py \
  --input_folder "evaluate/styleguard-infer/an_sks_painting_inclusing_a_vase_and_sunflowers" \
  --refer '/home/yjli/AIGC/diffusers/SimAC/evaluate/fid_ref.npz'

python evaluate/eval_precision_new.py \
  --reference_folder "evaluate/clean-infer/an_sks_painting_inclusing_a_vase_and_sunflowers/" \
  --gen_folder "evaluate/styleguard-infer/an_sks_painting_inclusing_a_vase_and_sunflowers"

experiment_name="vangogh_without_styleloss_SD14"

export INPUT_FOLDER="/home/yjli/AIGC/diffusers/Gogh_House/Gogh_House_SD1_5"
# python -m pytorch_image_generation_metrics.fid_ref \
#   --path $INPUT_FOLDER \
#   --output /home/yjli/AIGC/diffusers/SimAC/evaluate/house-fid-ref.npz

# python infer.py \
#   --model_path diffusion-model-finetuned/${experiment_name} \
#   --output_dir evaluate/${experiment_name}-infer/ \
#   --prompt "an sks painting, inclusing a house" \
#   --img_num 20

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