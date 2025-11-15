# export dir="/home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/vangogh_ensemble_ASPL_style_loss_upscaling"
export dir="/home/yjli/AIGC/diffusers/model/saved_model_vangogh"
# export dir="/home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/vangogh_with_styleloss_SD14" 
export output="/home/yjli/AIGC/diffusers/SimAC/outputs/clean/"
python infer.py  \
    --model_path=$dir \
    --output_dir=$output \
    --prompt "A sks painting of a mountain landscale with a blue sky"

python infer.py  \
    --model_path=$dir \
    --output_dir=$output \
    --prompt "A painting, a basket and some apples."

