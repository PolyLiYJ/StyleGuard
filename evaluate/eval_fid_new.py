from pytorch_image_generation_metrics import ImageDataset
from torch.utils.data import DataLoader

from pytorch_image_generation_metrics import (
    get_inception_score,
    get_fid,
    get_inception_score_and_fid
)
import argparse
parser = argparse.ArgumentParser(description="Upscale images with noise reduction.")

parser.add_argument("--input_folder", type=str, default="/home/yjli/AIGC/diffusers/StyleGuard/dreambooth-outputs/anti-style/vangogh_clean/checkpoint-1000-test-infer/an_sks_painting_including_a_blue_sky_and_mountains",
                    help="Path to the input image folder")
parser.add_argument("--output_folder", type=str, default="/home/yjli/AIGC/diffusers/StyleGuard/dreambooth-outputs/anti-style/vangogh_clean/checkpoint-1000-test-infer/an_sks_painting_including_a_blue_sky_and_mountains",
                    help="Path to the input image folder")
parser.add_argument("--refer", type=str, default='/home/yjli/AIGC/diffusers/StyleGuard/evaluate/fid_ref.npz',
                    help="Path to the input image folder")
# parser.add_argument("--output_folder", type=str, default="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/vangogh_ensemble_ASPL_style_loss_upscaling/noise-upscaling-x2",
#                     help="Path to the output image folder")
args = parser.parse_args()

# path_to_dir = "/home/yjli/AIGC/diffusers/SimAC/dreambooth-outputs/anti-style/vangogh_clean/checkpoint-1000-test-infer/an_sks_painting_including_a_blue_sky_and_mountains"
dataset = ImageDataset(args.input_folder, exts=['png', 'jpg'])
loader = DataLoader(dataset, batch_size=6, num_workers=4)

# Frechet Inception Distance
# FID = get_fid(
#     loader, args.refer)

(inception_score, std), FID = get_inception_score_and_fid(loader, args.refer)

with open(args.output_folder + '/fid_results.txt', 'w') as f:
    f.write('Frechet Inception Distance: {}'.format(FID))
    f.write('Inception Score: {}'.format(inception_score))
    f.write('Inception Score STD: {}'.format(std))
print("Frechet Inception Distance:", FID)
print("Inception Score:", inception_score)
print("Inception Score STD:", std)

# # Inception Score & Frechet Inception Distance
# (IS, IS_std), FID = get_inception_score_and_fid(
#     loader, "/home/yjli/AIGC/diffusers/SimAC/evaluate/fid_ref.npz")

# # print("Inception Score:", IS, " std:", IS_std)
# print("Frechet Inception Distance:", FID)