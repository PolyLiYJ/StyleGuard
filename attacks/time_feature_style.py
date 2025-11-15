import argparse
import copy
import hashlib
import itertools
import logging
import os
from pathlib import Path
import random
import datasets
import diffusers
import random
from torch.backends import cudnn
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import pickle
from diffusers import StableDiffusionUpscalePipeline
# from realesrgan import RealESRGANer
# from basicsr.archs.rrdbnet_arch import RRDBNet
from math import ceil
# from realesrgan import RealESRGAN
import torch
from PIL import Image
import torchvision.transforms as transforms

# Function to save a tensor as an image
def save_tensor_as_image(image_tensor, file_path):
    # Ensure the tensor is on CPU and has the correct shape    
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)  # Remove batch dimension
    new_image_tensor = image_tensor.clone().detach().cpu()  # Move to CPU if needed
    denormalized_tensor = new_image_tensor * 0.5 + 0.5  # Reverse normalization
    
    # Clip the values to be in the range [0, 1]
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)

    # Convert to PIL Image
    image = transforms.ToPILImage()(denormalized_tensor.cpu())  # Move to CPU if needed

    image.save(file_path)
    
logger = get_logger(__name__)


class DreamBoothDatasetFromTensor(Dataset):
    """Just like DreamBoothDataset, but take instance_images_tensor instead of path"""

    def __init__(
        self,
        instance_images_tensor,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_tensor = instance_images_tensor
        self.num_instance_images = len(self.instance_images_tensor)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images


        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
            # self.class_image = []
            # for index in range(0, self.num_class_images):
            #     self.class_image.append(Image.open(self.class_images_path[index]))
                #class_image = Image.open(self.class_images_path[index % self.num_class_images])
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        self.instance_prompt_ids = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        if class_data_root is not None:
            self.class_prompt_ids = self.tokenizer(
                    self.class_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images_tensor[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["instance_prompt_ids"] = self.instance_prompt_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            # class_image = self.class_image[index % self.num_class_images]
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"]  = self.class_prompt_ids
            # example["class_prompt_ids"] = self.tokenizer(
            #     self.class_prompt,
            #     truncation=True,
            #     padding="max_length",
            #     max_length=self.tokenizer.model_max_length,
            #     return_tensors="pt",
            # ).input_ids

        return example


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir_for_train",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_data_dir_for_adversarial",
        type=str,
        default=None,
        help="A folder containing the images to add adversarial noise",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a photo of sks person",
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a photo of person",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # default="text-inversion-model",
        default = 'outputs/simac/CelebA-HQ/',
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_f_train_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train surogate model.",
    )
    parser.add_argument(
        "--max_adv_train_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train adversarial noise.",
    )
    parser.add_argument(
        "--checkpointing_iterations",
        type=int,
        default=5,
        help=("Save a checkpoint of the training state every X iterations."),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=0.005,
        help="The step size for pgd.",
    )
    parser.add_argument(
        "--pgd_eps",
        type=int,
        default=16,
        help="The noise budget for pgd.",
    )
    parser.add_argument(
        "--target_image_path",
        default=None,
        help="target image for attacking",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help=(
            "Maximum steps for adaptive greedy timestep selection."
        ),
    )
    parser.add_argument(
        "--delta_t",
        type=int,
        default=20,
        help=(
            "delete 2*delta_t for each adaptive greedy timestep selection."
        ),
    )
    parser.add_argument(
        "--target_image_dir",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--style_loss_weight",
        type=float,
        default=1
    )
    parser.add_argument(
        "--noise_pred_loss_weight",
        type=float,
        default=1
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def load_data(data_dir, size=512, center_crop=True) -> torch.Tensor:
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    images = [image_transforms(Image.open(i).convert("RGB")) for i in list(Path(data_dir).iterdir())]
    images = torch.stack(images)
    return images



def train_one_epoch(
    args,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    data_tensor: torch.Tensor,
    num_steps=20,
):
    # Load the tokenizer
    # unet, text_encoder = copy.deepcopy(models[0]), copy.deepcopy(models[1])
    
    # unet = type(models[0])(**models[0].config).to(device)
    # unet.load_state_dict(models[0].state_dict())
    # text_encoder = type(models[1])(**models[1].config).to(device)
    # text_encoder.load_state_dict(models[1].state_dict())
    unet = models[0]
    text_encoder = models[1]

    params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    train_dataset = DreamBoothDatasetFromTensor(
        data_tensor,
        args.instance_prompt,
        tokenizer,
        args.class_data_dir,
        args.class_prompt,
        args.resolution,
        args.center_crop,
    )

    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    
    # instance_prompt_ids = tokenizer(
    #         self.args.instance_prompt,
    #         truncation=True,
    #         padding="max_length",
    #         max_length=tokenizer.model_max_length,
    #         return_tensors="pt",
    #     ).input_ids
    
    # class_prompt_ids = tokenizer(
    #             self.args.class_prompt,
    #             truncation=True,
    #             padding="max_length",
    #             max_length=self.tokenizer.model_max_length,
    #             return_tensors="pt",
    #         ).input_ids
    # scaler = torch.amp.GradScaler(device_type='cuda', enabled=True)


    for step in range(num_steps):
        unet.train()
        text_encoder.train()

        step_data = train_dataset[step % len(train_dataset)]
        
        if args.with_prior_preservation:
            pixel_values = torch.stack([step_data["instance_images"], step_data["class_images"]]).to(
                device, dtype=weight_dtype
            )
            input_ids = torch.cat([step_data["instance_prompt_ids"], step_data["class_prompt_ids"]], dim=0).to(device)
        else:
            pixel_values = torch.stack([step_data["instance_images"]]).to(
                device, dtype=weight_dtype
            )
            input_ids = torch.cat([step_data["instance_prompt_ids"]], dim=0).to(device)
        # input_ids = torch.cat([instance_prompt_ids, class_prompt_ids], dim=0).to(device)

        with torch.autocast(device_type='cuda'):

            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # with prior preservation loss
            if args.with_prior_preservation:
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # Add the prior loss to the instance loss.
                loss = instance_loss + args.prior_loss_weight * prior_loss

            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # # torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0, error_if_nonfinite=True)
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()
        
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad()
        if args.with_prior_preservation:
            print(f"Step #{step}, loss: {loss.detach().item()}, prior_loss: {prior_loss.detach().item()}, instance_loss: {instance_loss.detach().item()}")
        else:
            print(f"Step #{step}, loss: {loss.detach().item()}")

    return [unet, text_encoder]

def set_unet_attr(unet):
    def conv_forward(self):
        def forward(input_tensor, temb):
            self.in_layers_features = input_tensor
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            self.out_layers_features = hidden_states
            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward
    
    conv_module_list = [
                        unet.up_blocks[0].resnets[0],unet.up_blocks[0].resnets[1], unet.up_blocks[0].resnets[2],
                        unet.up_blocks[1].resnets[0],unet.up_blocks[1].resnets[1], unet.up_blocks[1].resnets[2],
                        unet.up_blocks[2].resnets[0],unet.up_blocks[2].resnets[1], unet.up_blocks[2].resnets[2],
                        unet.up_blocks[3].resnets[0],unet.up_blocks[3].resnets[1], unet.up_blocks[3].resnets[2],
                        unet.down_blocks[0].resnets[0],unet.down_blocks[0].resnets[1],
                        unet.down_blocks[1].resnets[0],unet.down_blocks[1].resnets[1],
                        unet.down_blocks[2].resnets[0],unet.down_blocks[2].resnets[1],
                        unet.down_blocks[3].resnets[0],unet.down_blocks[3].resnets[1],
                    ]                                                                          
    for conv_module in conv_module_list:
        conv_module.forward = conv_forward(conv_module)
        setattr(conv_module, 'in_layers_features', None)
        setattr(conv_module, 'out_layers_features', None)

def save_feature_maps(up_blocks, down_blocks):

    out_layers_features_list_0 = []
    out_layers_features_list_1 = []
    out_layers_features_list_2 = []
    out_layers_features_list_3 = []

    in_layers_features_list_0 = []
    in_layers_features_list_1 = []
    in_layers_features_list_2 = []
    in_layers_features_list_3 = []
    res_0_list =[0,1,2]
    res_1_list =[0,1,2]
    res_2_list =[0,1,2]
    res_3_list =[0,1,2]
    in_0_list =[0,1]
    in_1_list =[0,1]
    in_2_list =[0,1]
    in_3_list =[0,1]
    block_idx = 0
    for block in up_blocks:
        if block_idx == 0: 
            for index in res_0_list:
                out_layers_features_list_0.append(block.resnets[index].out_layers_features)
        if block_idx == 1: 
            for index in res_1_list:
                out_layers_features_list_1.append(block.resnets[index].out_layers_features)
        if block_idx == 2: 
            for index in res_2_list:
                out_layers_features_list_2.append(block.resnets[index].out_layers_features)
        if block_idx == 3: 
            for index in res_3_list:
                out_layers_features_list_3.append(block.resnets[index].out_layers_features)
        block_idx += 1
    out_layers_features_list_0 = torch.stack(out_layers_features_list_0, dim=0)
    out_layers_features_list_1 = torch.stack(out_layers_features_list_1, dim=0)
    out_layers_features_list_2 = torch.stack(out_layers_features_list_2, dim=0)
    out_layers_features_list_3 = torch.stack(out_layers_features_list_3, dim=0)
    block_idx = 0
    for block in down_blocks:
        if block_idx == 0: 
            for index in in_0_list:
                in_layers_features_list_0.append(block.resnets[index].out_layers_features)
        if block_idx == 1: 
            for index in in_1_list:
                in_layers_features_list_1.append(block.resnets[index].out_layers_features)
        if block_idx == 2: 
            for index in in_2_list:
                in_layers_features_list_2.append(block.resnets[index].out_layers_features)
        if block_idx == 3: 
            for index in in_3_list:
                in_layers_features_list_3.append(block.resnets[index].out_layers_features)
        block_idx += 1
    in_layers_features_list_0 = torch.stack(in_layers_features_list_0, dim=0)
    in_layers_features_list_1 = torch.stack(in_layers_features_list_1, dim=0)
    in_layers_features_list_2 = torch.stack(in_layers_features_list_2, dim=0)
    in_layers_features_list_3 = torch.stack(in_layers_features_list_3, dim=0)
    return out_layers_features_list_0, out_layers_features_list_1, out_layers_features_list_2, out_layers_features_list_3,\
            in_layers_features_list_0, in_layers_features_list_1, in_layers_features_list_2, in_layers_features_list_3


# Function to add Gaussian noise
def add_gaussian_noise(image_tensor, mean=0, std=0.1):
    noise = torch.randn_like(image_tensor) * std + mean
    noise = noise.to(image_tensor.device)
    # noise.requires_grad=False
    noisy_image = image_tensor + noise
    return noisy_image  # Ensure values are in the valid range

# Function to load and preprocess images
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize for upscaling
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Main function for Noisy Upscaling
def noisy_upscaling(image_path, upscaler_model):
    # Load the image
    image_tensor = load_image(image_path)

    # Add Gaussian noise
    noisy_image_tensor = add_gaussian_noise(image_tensor)

    # Upscale the noisy image
    upscaled_image = upscaler_model(prompt = "a painting", image = noisy_image_tensor).images[0]

    return upscaled_image

# Load the Stable Diffusion Upscaler model



def pgd_attack(
    args,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    upscaler_model,
    data_tensor: torch.Tensor,
    original_images: torch.Tensor,
    target_tensor: torch.Tensor,
    num_steps: int,
    time_list
):
    """Return new perturbed data"""

    unet, text_encoder = models
    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    set_unet_attr(unet)

    perturbed_images = data_tensor.detach().clone()
    perturbed_images.requires_grad_(True)

    input_ids = tokenizer(
        args.instance_prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.repeat(len(data_tensor), 1)
    
    target_images = load_data(args.target_image_dir, size=512, center_crop=True)
    target_style_image = target_images[0].unsqueeze(0)
    target_latents = vae.encode(target_style_image.to(device, dtype=weight_dtype)).latent_dist.sample()
    target_latents = target_latents * vae.config.scaling_factor

    # 
    if args.style_loss_weight>0:
        with torch.no_grad():
            # target_features = unet(target_latents, timesteps, encoder_hidden_states)
            # target_mean = target_features.mean(dim=[0, 2, 3])  # 
            # target_var = target_features.var(dim=[0, 2, 3])      # 
            target_mean = target_latents.mean(dim=[0, 2, 3])  # 
            target_var = target_latents.var(dim=[0, 2, 3])      # 
            
            original_latents = vae.encode(data_tensor.to(device, dtype=weight_dtype)).latent_dist.sample()
            original_latents = original_latents * vae.config.scaling_factor
            original_mean = original_latents.mean(dim=[0, 2, 3])  # 
            original_var = original_latents.var(dim=[0, 2, 3])      # 

                
    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids.to(device))[0]

    for step in range(num_steps):
        perturbed_images.requires_grad = True
        perturbed_images.to("cuda")
        random_number = random.uniform(0, 1)
        if random_number < 0.5:
            # perturbed_images = perturbed_images.to(next(upscaler_model.model.parameters()).device)
            # perturbed_images = perturbed_images.half()
            original_size = perturbed_images.shape[2]  # Store original size
            # Add Gaussian noise
            # save_tensor_as_image(perturbed_images.clone().detach()[0], f"perturbed_images_step_{step}.png")

            perturbed_images_noised = add_gaussian_noise(perturbed_images)
            # save_tensor_as_image(perturbed_images_noised.clone().detach()[0], f"perturbed_images_noised_step_{step}.png")

            print("perturbed_images shape ", perturbed_images.shape)
            # Upscale the noisy image
            # super resolution            
            # Store original size
            # original_size = perturbed_images.shape[2]  # Assuming shape is [B, C, H, W]

            # Upscale the noisy image
            # Compute new size as a tuple of integers
            # new_size = (256, 256)            
            # perturbed_images_shrink = torch.nn.functional.interpolate(
            #     perturbed_images_noised,  # Input tensor
            #     size=new_size,     # Target size as a tuple of integers
            #     mode='bilinear',   # Interpolation mode
            #     align_corners=False
            # )
            
            # random crop 256*256
            crop_size = (460, 460)
            B, C, H, W = perturbed_images.shape

            # Ensure the upscaled image is larger than the crop size
            if H >= crop_size[0] and W >= crop_size[1]:
                # Generate random crop coordinates
                top = torch.randint(0, H - crop_size[0] + 1, (1,)).item()
                left = torch.randint(0, W - crop_size[1] + 1, (1,)).item()

                # Perform the random crop
                perturbed_images_cropped = perturbed_images_noised[:, :, top:top + crop_size[0], left:left + crop_size[1]]

            # perturbed_images_upscale = upscaler_model.model(perturbed_images_cropped)
            perturbed_images_upscale = torch.nn.functional.interpolate(
                perturbed_images_cropped,  # Input tensor
                size=(512, 512),     # Target size as a tuple of integers
                mode='bilinear',   # Interpolation mode
                align_corners=False
            )

            print("upscaled_image_tensor shape:", perturbed_images_upscale.shape)
            # Save the upscaled image tensor as a file
            # save_tensor_as_image(perturbed_images_upscale.clone().detach()[0], f"upscaled_image_step_{step}.png")
            latents = vae.encode(perturbed_images_upscale.to(device, dtype=weight_dtype)).latent_dist.sample()
        else:
            latents = vae.encode(perturbed_images.to(device, dtype=weight_dtype)).latent_dist.sample()

        latents = latents * vae.config.scaling_factor
        current_mean = latents.mean(dim=[0, 2, 3])  # 
        current_var = latents.var(dim=[0, 2, 3])      # 
        if args.style_loss_weight > 0:
            style_loss = F.mse_loss(current_mean, original_mean) + F.mse_loss(current_var, original_var) \
                        - F.mse_loss(current_mean, target_mean) - F.mse_loss(current_var, target_var)
                    
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = []
        for i in range(len(data_tensor)):
            ts = time_list[i]
            # max_step = min(100, len(ts))  #  timestep
            # ts_index = torch.randint(0, max_step, (1,))
            ts_index = torch.randint(0, len(ts), (1,))
            timestep = torch.IntTensor([ts[ts_index]])
            timestep = timestep.long()
            timesteps.append(timestep)
        timesteps = torch.cat(timesteps).to(device)
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if  args.noise_pred_loss_weight > 0:
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            # Get the text embedding for conditioning
            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            noise_pred_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # feature loss
            # noise_out_layers_features_0, noise_out_layers_features_1, noise_out_layers_features_2, noise_out_layers_features_3,\
            # noise_in_layers_features_0, noise_in_layers_features_1, noise_in_layers_features_2, noise_in_layers_features_3 = save_feature_maps(unet.up_blocks, unet.down_blocks)
            # with torch.no_grad():
            #     clean_latents = vae.encode(data_tensor.to(device, dtype=weight_dtype)).latent_dist.sample()
            #     clean_latents = clean_latents * vae.config.scaling_factor
            #     noisy_clean_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)
            #     clean_model_pred = unet(noisy_clean_latents, timesteps, encoder_hidden_states).sample
            #     clean_out_layers_features_0, clean_out_layers_features_1, clean_out_layers_features_2, clean_out_layers_features_3,\
            #     clean_in_layers_features_0, clean_in_layers_features_1, clean_in_layers_features_2, clean_in_layers_features_3 = save_feature_maps(unet.up_blocks,  unet.down_blocks)
            # target_loss =  F.mse_loss(noise_out_layers_features_3.float(), clean_out_layers_features_3.float(), reduction="mean")
            unet.zero_grad()
            text_encoder.zero_grad()
        
        # loss = loss + target_loss.detach().item()
        loss = 0
        if args.style_loss_weight > 0:
            loss = loss + style_loss * args.style_loss_weight
            print("PGD loss -- step:", step, 
               "style_loss:", style_loss.item())
        if  args.noise_pred_loss_weight > 0:
            loss = loss + noise_pred_loss * args.noise_pred_loss_weight
            print("PGD loss -- step:", step, "noise_pred_loss:", noise_pred_loss.item())
        # print("step:", step, "loss:",  loss.item(),"noise_pred_loss:", noise_pred_loss.item(),
        #       "style_loss:", style_loss.item())
        loss.backward()
        alpha = args.pgd_alpha
        eps = args.pgd_eps / 255
        # maximum the loss
        adv_images = perturbed_images + alpha * perturbed_images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps, max=+eps)
        perturbed_images = torch.clamp(original_images + eta, min=-1, max=+1).detach_()
        # print(f"PGD loss - step {step}, loss: {loss.detach().item()}, target_loss : {target_loss.detach().item()}")

    return perturbed_images

def select_timestep(
    args,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    data_tensor: torch.Tensor,
    original_images: torch.Tensor,
    target_tensor: torch.Tensor,
    ):
    """Return new perturbed data"""

    unet, text_encoder = models
    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    perturbed_images = data_tensor.detach().clone()
    perturbed_images.requires_grad_(True)


    input_ids = tokenizer(
        args.instance_prompt, #"a photo of sks person"
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    
    time_list = []
    for id in range(len(data_tensor)):
        perturbed_image = perturbed_images[id, :].unsqueeze(0)
        original_image = original_images[id, :].unsqueeze(0)
        time_seq = torch.tensor(list(range(0, 1000)))
        input_mask = torch.ones_like(time_seq)
        id_image = perturbed_image.detach().clone()
        for step in range(args.max_steps):
            id_image.requires_grad_(True)
            select_mask = torch.where(input_mask==1, True, False)
            res_time_seq = torch.masked_select(time_seq, select_mask)
            if len(res_time_seq) > 100:
                min_score, max_score = 0.0, 0.0
                for index in range(0, 5):
                    id_image.requires_grad_(True)
                    latents = vae.encode(id_image.to(device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    inner_index = torch.randint(0, len(res_time_seq), (bsz,))
                    timesteps = torch.IntTensor([res_time_seq[inner_index]]).to(device)
                    timesteps = timesteps.long()
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(input_ids.to(device))[0]
                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    unet.zero_grad()
                    text_encoder.zero_grad()
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    loss.backward()
                    score = torch.sum(torch.abs(id_image.grad.data))
                    index = index + 1
                    id_image.grad.zero_()
                    if index == 1:
                        min_score = score
                        max_score = score
                        del_t = res_time_seq[inner_index].item()
                        select_t = res_time_seq[inner_index].item()
                    else:
                        if min_score > score:
                            min_score = score
                            del_t = res_time_seq[inner_index].item()
                        if max_score < score:
                            max_score = score
                            select_t = res_time_seq[inner_index].item()
                    print(f"PGD loss - step {step}, index : {index}, loss: {loss.detach().item()}, score: {score}, t : {res_time_seq[inner_index]}, ts_len: {len(res_time_seq)}")

                print("del_t", del_t, "max_t", select_t)
                if del_t < args.delta_t :
                    del_t = args.delta_t
                elif  del_t > (1000 - args.delta_t):
                    del_t= 1000 - args.delta_t
                input_mask[del_t - 20: del_t + 20] = input_mask[del_t - 20: del_t + 20] - 1
                input_mask = torch.clamp(input_mask, min=0, max=+1)

                id_image.requires_grad_(True)
                latents = vae.encode(id_image.to(device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.IntTensor([select_t]).to(device)
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(input_ids.to(device))[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                unet.zero_grad()
                text_encoder.zero_grad()
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss.backward()
                alpha = args.pgd_alpha
                eps = args.pgd_eps / 255
                adv_image = id_image + alpha * id_image.grad.sign()
                eta = torch.clamp(adv_image - original_image, min=-eps, max=+eps)
                score = torch.sum(torch.abs(id_image.grad.sign()))
                id_image = torch.clamp(original_image + eta, min=-1, max=+1).detach_()

            else:
                # print(id, res_time_seq, step, len(res_time_seq))
                time_list.append(res_time_seq)
                break
    return time_list

def setup_seeds():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def copy_model_params(source_model, target_model):
    """ state_dict """
    target_model.load_state_dict(source_model.state_dict())


    # 
from torch.utils.data import DataLoader, TensorDataset

def create_batch_loader(data_tensor, batch_size=4):
    """"""
    dataset = TensorDataset(data_tensor)  # 
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    return accelerator.prepare(loader) if accelerator else loader

def main(args):
    # args.pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base"
    # args.instance_data_dir_for_train="data/CelebA-HQ/103/set_A" 
    # args.instance_data_dir_for_adversarial="data/CelebA-HQ/103/set_B"
    # args.output_dir="outputs/simac/CelebA-HQ/103"
    # args.class_data_dir="clean_class_image"
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # logging_dir=logging_dir,
        project_dir=args.logging_dir
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    print(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    setup_seeds()
    # Generate class images if prior preservation is enabled.
    # if args.with_prior_preservation:
    #     class_images_dir = Path(args.class_data_dir)
    #     if not class_images_dir.exists():
    #         class_images_dir.mkdir(parents=True)
    #     cur_class_images = len(list(class_images_dir.iterdir()))

    #     if cur_class_images < args.num_class_images:
    #         torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
    #         if args.mixed_precision == "fp32":
    #             torch_dtype = torch.float32
    #         elif args.mixed_precision == "fp16":
    #             torch_dtype = torch.float16
    #         elif args.mixed_precision == "bf16":
    #             torch_dtype = torch.bfloat16
    #         pipeline = DiffusionPipeline.from_pretrained(
    #             args.pretrained_model_name_or_path,
    #             torch_dtype=torch_dtype,
    #             safety_checker=None,
    #             revision=args.revision,
                
    #         )
    #         pipeline.set_progress_bar_config(disable=True)

    #         num_new_images = args.num_class_images - cur_class_images
    #         logger.info(f"Number of class images to sample: {num_new_images}.")

    #         sample_dataset = PromptDataset(args.class_prompt, num_new_images)
    #         sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

    #         sample_dataloader = accelerator.prepare(sample_dataloader)
    #         pipeline.to(accelerator.device)

    #         for example in tqdm(
    #             sample_dataloader,
    #             desc="Generating class images",
    #             disable=not accelerator.is_local_main_process,
    #         ):
    #             images = pipeline(example["prompt"]).images

    #             for i, image in enumerate(images):
    #                 hash_image = hashlib.sha1(image.tobytes()).hexdigest()
    #                 image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
    #                 image.save(image_filename)

    #         del pipeline
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, 
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,
    ).cuda()

    # vae.requires_grad_(False)
    vae.requires_grad_(False)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    clean_data = load_data(
        args.instance_data_dir_for_train,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    perturbed_data = load_data(
        args.instance_data_dir_for_adversarial,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    
    print("perturbed_data", perturbed_data.shape)
    
# 
    # clean_loader = create_batch_loader(clean_data, args.train_batch_size)
    # perturbed_loader = create_batch_loader(perturbed_data, args.train_batch_size)

    original_data = perturbed_data.clone()
    original_data.requires_grad_(False)
    # original_loader = create_batch_loader(original_data, args.train_batch_size)

    # original_data.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    target_latent_tensor = None
    if args.target_image_path is not None:
        target_image_path = Path(args.target_image_path)
        assert target_image_path.is_file(), f"Target image path {target_image_path} does not exist"

        target_image = Image.open(target_image_path).convert("RGB").resize((args.resolution, args.resolution))
        target_image = np.array(target_image)[None].transpose(0, 3, 1, 2)

        target_image_tensor = torch.from_numpy(target_image).to("cuda", dtype=torch.float32) / 127.5 - 1.0
        target_latent_tensor = (
            vae.encode(target_image_tensor).latent_dist.sample().to(dtype=torch.bfloat16) * vae.config.scaling_factor
        )
        target_latent_tensor = target_latent_tensor.repeat(len(perturbed_data), 1, 1, 1).cuda()

    f = [unet, text_encoder]
    
    # time_list = select_timestep(
    #             args,
    #             f,
    #             tokenizer,
    #             noise_scheduler,
    #             vae,
    #             perturbed_data,
    #             original_data,
    #             target_latent_tensor,
    # )
    # Check if the time_list file exists
    print("output dir:", args.output_dir)
    time_list_file = os.path.join(args.output_dir, "time_list.pt")

    if os.path.exists(time_list_file):
        # Load time_list from the file
        print(f"Loading time_list from {time_list_file}")
        time_list = torch.load(time_list_file)
    else:
        # Compute time_list if the file does not exist
        print("Computing time_list...")
        time_list = select_timestep(
            args,
            f,
            tokenizer,
            noise_scheduler,
            vae,
            perturbed_data,
            original_data,
            target_latent_tensor,
        )

        # Save time_list to the file
        print(f"Saving time_list to {time_list_file}")
        torch.save(time_list, time_list_file)

    # Print the time_list
    print("Time list:", time_list)
    
    for t in time_list:
        print(t)
    # scaler = torch.cuda.amp.GradScaler()
    # MODEL_BANKS = [load_model(args, path) for path in model_paths]
    MODEL_STATEDICTS = {
            "text_encoder":text_encoder.state_dict(),
            "unet": unet.state_dict(),
        }
    
    tokenizer = AutoTokenizer.from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder="tokenizer",
                    revision=args.revision,
                    use_fast=False,
                )        
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", )

    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,
        ).cuda()
    
    upscaler_model = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler")
    upscaler_model=upscaler_model.to("cuda")
    # https://github.com/xinntao/Real-ESRGAN
    # upscaler_model = RealESRGANer(scale=2,model_path="upscaler_model/RealESRGAN_x4plus.pth")
    # upscaler_model = upscaler_model.to('cuda')  # Move to GPU if available
    
    scale = 2
    # if scale == 2:
    #     RRDB_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    #     upscaler_model = RealESRGANer(scale=2,model_path="upscaler_model/RealESRGAN_x2plus.pth", model = RRDB_model, half=True)
    #     upscaler_model.model.to("cuda")
    #     upscaler_model.model.train()
    # elif scale == 4:
    #     RRDB_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    #     upscaler_model = RealESRGANer(scale=4,model_path="upscaler_model/RealESRGAN_x4plus.pth", model = RRDB_model, half=True)
    #     upscaler_model.model.to("cuda")
    upscaler_model = None

    
    f = [unet, text_encoder]
    f_sur = copy.deepcopy(f)
    bs = args.train_batch_size
    for i in range(args.max_train_steps):
        total_samples = len(perturbed_data)
        print("total_samples:", total_samples)
        effective_bs = min(bs, total_samples)
        num_batches = ceil(total_samples / effective_bs)
        print("num_batches:", num_batches)

        for batch_idx in range(0,num_batches):
            start = batch_idx * effective_bs
            end = min((batch_idx+1)*effective_bs, total_samples)
            print("batch_idx", batch_idx)
            print("start: ", start, "end:", end)
            print("perturbed_data length:", perturbed_data.shape)
            # 
            
            # clean_data = clean_batch[0]
            # original_data = original_batch[0]
            # perturbed_data = perturb_batch[0]
            perturbed_data_batch = perturbed_data[start:end]
            print("perturbed_data_batch shape", perturbed_data_batch.shape)
            clean_data_batch = clean_data[start:end]
            original_data_batch = original_data[start:end]
                
            copy_model_params(f[0], f_sur[0])  # Unet
            copy_model_params(f[1], f_sur[1])  # TextEncoder
            print(f"f[0] memory address: {id(f[0])}")
            print(f"f_sur[0] memory address: {id(f_sur[0])}")
            
            vae.requires_grad_(False)
            f_sur = train_one_epoch(
                args,
                f_sur,
                tokenizer,
                noise_scheduler,
                vae,
                clean_data_batch,
                args.max_f_train_steps
            )
            for param in f_sur[0].parameters():
                param.grad = None
            for param in f_sur[1].parameters():
                param.grad = None
                
            vae.requires_grad_(False)
            perturbed_data_batch = pgd_attack(
                args,
                f_sur,
                tokenizer,
                noise_scheduler,
                vae,
                upscaler_model, 
                perturbed_data_batch,
                original_data_batch,
                target_latent_tensor,
                args.max_adv_train_steps,
                time_list
            )
            for param in f_sur[0].parameters():
                param.grad = None
            for param in f_sur[1].parameters():
                param.grad = None
            # del f_sur
        
            vae.requires_grad_(False)

            f = train_one_epoch(
                args,
                f,
                tokenizer,
                noise_scheduler,
                vae,
                perturbed_data_batch,
                args.max_f_train_steps,
            )
            # Release memory

            # save new statedicts
            # MODEL_STATEDICTS["unet"] = f[0].state_dict()
            # MODEL_STATEDICTS["text_encoder"] = f[1].state_dict()
            # del f
            # del tokenizer, noise_scheduler, vae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            perturbed_data[start:end] = perturbed_data_batch
            clean_data[start:end] = clean_data_batch
            original_data[start:end] = original_data_batch

        if (i + 1) % args.checkpointing_iterations == 0:
            save_folder = f"{args.output_dir}/noise-ckpt/{i+1}"
            os.makedirs(save_folder, exist_ok=True)
            noised_imgs = perturbed_data.detach()
            img_names = [
                str(instance_path).split("/")[-1].split(".")[0]
                for instance_path in list(Path(args.instance_data_dir_for_adversarial).iterdir())
            ]
            for img_pixel, img_name in zip(noised_imgs, img_names):
                save_path = os.path.join(save_folder, f"{i+1}_noise_{img_name}.png")
                Image.fromarray(
                    ((img_pixel +1) * 127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                ).save(save_path)
            print(f"Saved noise at step {i+1} to {save_folder}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
