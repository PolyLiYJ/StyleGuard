import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion Dreambooth Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained Dreambooth model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./infer_output/",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Generation prompt (e.g., 'a photo of sks dog')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generating images with prompt: '{args.prompt}'")
    images = []
    for i in range(args.num_images):
        image = pipe(
            args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
        ).images[0]
        
        save_path = os.path.join(args.output_dir, f"image_{i}.png")
        image.save(save_path)
        images.append(image)
        print(f"Saved: {save_path}")

    # Generate grid if multiple images
    if len(images) > 1:
        grid = make_grid([torch.from_numpy(np.array(img).transpose(2, 0, 1)) for img in images], nrow=2)
        grid = Image.fromarray(grid.permute(1, 2, 0).numpy())
        grid_path = os.path.join(args.output_dir, "grid.png")
        grid.save(grid_path)
        print(f"Grid saved: {grid_path}")

    print("Inference complete!")

if __name__ == "__main__":
    main()
