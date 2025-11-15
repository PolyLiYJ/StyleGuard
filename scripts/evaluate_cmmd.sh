# conda activate cmmd
export REF_FOLDER="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/Gogh_House_SD1_5"
export TRAIN_FOLDER="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/vangogh"
export PYTHONPATH=$PYTHONPATH:/home/yjli/AIGC/diffusers/SimAC/scenic

# python -m pytorch_fid --save-stats $TRAIN_FOLDER evaluate/vangogh_training_data.npz
export GPU="6,7"
export CUDA_VISIBLE_DEVICES=$GPU
for STYLE in 12 13 14 15; do  
    echo "style: $STYLE"
    conda run -n cmmd python -m cmmd.main --ref_folder=$TRAIN_FOLDER \
    --gen_folder="/home/yjli/AIGC/diffusers/SimAC/data/wikiart/$STYLE" \
    --output_folder "/home/yjli/AIGC/diffusers/SimAC/data/wikiart/" \
    --batch_size 2
done
