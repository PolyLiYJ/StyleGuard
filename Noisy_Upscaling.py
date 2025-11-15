import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from diffusers import StableDiffusionUpscalePipeline,StableDiffusionLatentUpscalePipeline
import os
import argparse
import torch.nn.functional as F

from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
import torch
from huggingface_hub import login
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 读取 Hugging Face Token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Hugging Face 登录
from huggingface_hub import login
login(token=HUGGINGFACE_TOKEN)




# Function to add Gaussian noise
def add_gaussian_noise(image_tensor, mean=0, std=0.1):
    noise = torch.randn_like(image_tensor) * std + mean
    noisy_image = image_tensor + noise
    return noisy_image.clamp(0, 1)  # Ensure values are in the valid range

# Function to load and preprocess images
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize for upscaling
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
# pipeline.to("cuda")

# Main function for Noisy Upscaling
def noisy_upscaling(args, image_path, upscaler_model, contronet=None):
    # Load the image
    image_tensor = load_image(image_path)
    # Add Gaussian noise
    noisy_image_tensor = add_gaussian_noise(image_tensor).to("cuda")
    # Upscale the noisy image
    if args.upscaler=="x4":
        upscaler_model = upscaler_model.to(args.device)
        noisy_image_tensor=noisy_image_tensor.to(args.device)
        upscaled_image = upscaler_model(prompt = "a painting", image = noisy_image_tensor, num_inference_steps = args.step, noise_level = 20).images[0]
    elif args.upscaler=="x2":
        generator = torch.manual_seed(33)
        prompt = "a painting"
        # we stay in latent space! Let's make sure that Stable Diffusion returns the image
        # in latent space
        # low_res_latents = pipeline(prompt, generator=generator, output_type="latent").images
        with torch.no_grad():
            # noisy_image_tensor = noisy_image_tensor/2+0.5
            # pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
            # pipeline.to("cuda")
            # latents = pipeline.vae.encode(noisy_image_tensor).latent_dist.sample()  # Sample from the latent distribution
            # Scale the latents (required for Stable Diffusion)
            # print(pipeline.vae.config.scaling_factor)
            # latents = latents * pipeline.vae.config.scaling_factor
            # upscaled_image = pipeline.vae.decode(latents/pipeline.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            # upscaled_image = pipeline.image_processor.postprocess(upscaled_image, output_type="pil", do_denormalize="True")[0]

            upscaled_image = upscaler_model(
               prompt=prompt,
               # image=latents,
               image=noisy_image_tensor,
               num_inference_steps=50,
               guidance_scale=0,
               generator=generator,
            ).images[0]
            
    elif args.upscaler=="flux":
        # control_image = control_image.resize((w * 4, h * 4))
        high_res_tensor = F.interpolate(noisy_image_tensor, size=(1024, 1024), mode="bilinear", align_corners=False)
        high_res_tensor = high_res_tensor.to(torch.bfloat16).to("cuda")
        upscaled_image = upscaler_model(
            prompt="", 
            control_image=high_res_tensor,
            controlnet_conditioning_scale=0.6,
            num_inference_steps=28, 
            guidance_scale=3.5,
            height=512*2,
            width=512*2
        ).images[0]
    return upscaled_image

# Load the Stable Diffusion Upscaler model
# upscaler_model = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler")

# Specify the directory path
# directory_path = '/home/yjli/AIGC/diffusers/Gogh_House_SD1_5_glaze_ViT'
def main():
    parser = argparse.ArgumentParser(description="Upscale images with noise reduction.")
    parser.add_argument("--input_folder", type=str, default="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/vangogh_ensemble_ASPL_style_loss_upscaling/noise-ckpt/50",
                        help="Path to the input image folder")
    parser.add_argument("--output_folder", type=str, default="/home/yjli/AIGC/diffusers/SimAC/outputs/style/wikiart/vangogh_ensemble_ASPL_style_loss_upscaling/noise-upscaling-x2",
                        help="Path to the output image folder")
    parser.add_argument("--device", type=str, default="cuda:4", required=False, help="Path to the output image folder")
    parser.add_argument("--upscaler", type=str, default="x2", required=False, help="Path to the output image folder")
    parser.add_argument("--step", type=int, default=50, required=False, help="diffusion steps")

    args = parser.parse_args()

    directory_path = args.input_folder
    output_path = args.output_folder

    os.makedirs(output_path, exist_ok=True)

    # 列出所有文件
    files = os.listdir(directory_path)
    with torch.no_grad():
        if args.upscaler=="x4":
            upscaler_model = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler").to("cuda")
        elif args.upscaler=="x2":
            #x2 work in latent space
            upscaler_model = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler").to("cuda")
        elif args.upscaler=="flux":
            # Load pipeline
            controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler",
            torch_dtype=torch.bfloat16
            )
            upscaler_model = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
            )
            upscaler_model.to("cuda")
            # upscaler_model.enable_model_cpu_offload()

        
        for image_path in files:
            path = os.path.join(directory_path, image_path)
            print(f"Processing: {path}")
            # upscaler_model = upscaler_model.to(args.device)
            upscaled_image = noisy_upscaling(args, path, upscaler_model)
            print(f"Resized image size: {upscaled_image.size[0]}x{upscaled_image.size[1]} pixels")
            # 保存结果
            save_path = os.path.join(output_path, image_path)
            resized_image = upscaled_image.resize((512, 512))
            resized_image = upscaled_image
            resized_image.save(save_path)
            print(f"Saved path: {save_path}")

if __name__ == "__main__":
    main()