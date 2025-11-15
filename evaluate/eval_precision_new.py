import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
# work with tensorflow==2.10 and cudnn-9
import tensorflow.compat.v1 as tf
from scipy import linalg
from PIL import Image
from tqdm import tqdm
import sys
from evaluator import Evaluator, ManifoldEstimator
# import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
# Ensure TensorFlow uses GPU if available
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session()

# Initialize the Evaluator
evaluator = Evaluator(session)

# Function to load and preprocess images
def load_and_preprocess_images(folder, img_size=(256, 256)):
    images = []
    for img_name in tqdm(os.listdir(folder), desc=f"Loading images from {folder}"):
        img_path = os.path.join(folder, img_name)
        img = Image.open(img_path).resize(img_size)
        img = np.array(img).astype(np.float32)
        images.append(img)
    return np.stack(images)
import argparse


parser = argparse.ArgumentParser(description="Upscale images with noise reduction.")
parser.add_argument("--reference_folder", type=str, default="evaluate/clean-infer/an_sks_painting_inclusing_a_vase_and_sunflowers/",
                    help="Path to the input image folder")
parser.add_argument("--gen_folder", type=str, default='/home/yjli/AIGC/diffusers/SimAC/evaluate/mist-upscale-infer/an_sks_painting_inclusing_a_vase_and_sunflowers',
                    help="Path to the input image folder")
parser.add_argument("--output_folder", type=str, default="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/vangogh_ensemble_ASPL_style_loss_upscaling/noise-upscaling-x2",
                    help="Path to the output image folder")
args = parser.parse_args()
# Load reference and generated images
reference_images = load_and_preprocess_images(args.reference_folder)
print("reference folder:", args.reference_folder)
# generated_images = load_and_preprocess_images("/home/yjli/AIGC/diffusers/SimAC/evaluate/mist-upscale-infer/an_sks_painting_inclusing_a_vase_and_sunflowers")
# Precision: 0.45
# generated_images = load_and_preprocess_images("/home/yjli/AIGC/diffusers/SimAC/evaluate/styleguard-infer/an_sks_painting_inclusing_a_vase_and_sunflowers")
#Precision:0.00
generated_images = load_and_preprocess_images(args.gen_folder)
# Precision:0.00
print("generated folder:", args.gen_folder)

# Compute activations
# print("Computing reference image activations...")
ref_acts = evaluator.compute_activations([reference_images])[0]  # Extract pool_3 features

# print("Computing generated image activations...")
gen_acts = evaluator.compute_activations([generated_images])[0]

# print("Computing Precision...")

prec, recall = evaluator.compute_prec_recall(ref_acts, gen_acts)
print("Precision:", prec)
print("Recall:", recall)
with open(args.output_folder + '/precision_results.txt', 'w') as fid:
    fid.write("Precision: " + str(prec) + "\n")
    fid.write("Recall: " + str(recall) + "\n")
# # Compute Precision
# print("Computing Precision...")
# manifold_estimator = ManifoldEstimator(session)
# radii_ref = manifold_estimator.manifold_radii(ref_acts)
# precision, _ = manifold_estimator.evaluate_pr(ref_acts, radii_ref, gen_acts, radii_ref)

# print(f"Precision: {precision[0]:.4f}")
