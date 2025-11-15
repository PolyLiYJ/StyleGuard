
#
This repository provides the official PyTorch implementation of the following paper: 

# StyleGuard: Preventing Text-to-Image-Model-based Style Mimicry Attacks by Style Perturbations

link: https://arxiv.org/abs/2505.18766

Yanjie Li, Wenxuan Zhang, Xinqi LYU, Yihao LIU, Bin Xiao

Hong Kong Polytechnic University 

NeurIPS 2025 poster


🛡️

**TL;DR**: StyleGuard defends against unauthorized style mimicry in diffusion models (e.g., DreamBooth/Textual Inversion) via **style-aware adversarial perturbations** robust to purification attacks.

## Key Innovations
- 🎨 **Style Loss**: Latent-space optimization disrupts style transfer while maintaining model-agnostic transferability  
- ⚔️ **Upscale Loss**: Ensemble purification resistance via adversarial training with upscalers/purifiers  
- 🏆 **SOTA Robustness**: Outperforms Glaze/Anti-DreamBooth against:
  - Random transformations (blur, JPEG, etc.)
  - Diffusion purification (DiffPure, Noise Upscaling)  

#

##### Table of contents
1. [Environment setup](#environment-setup)
2. [Dataset](#dataset)
3. [How to run](#how-to-run)
4. [Contacts](#contacts)
5. [Acknowledgement](#acknowledgement)
6. [Citation](#citation)

## Environment setup
Install dependencies:
```shell
conda env create -f environment.yml
conda activate styleguard
```

Logging into HuggingFace Hub:
```shell
huggingface-cli login
```
or set the huggingface token in the environment variable:
```shell
export huggingface_token=<your huggingface token>
```
You can save the token in the token.txt file.

Pretrained checkpoints of different Stable Diffusion versions can be **downloaded** from provided links in the table below:
<table style="width:100%">
  <tr>
    <th>Version</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>2.1</td>
    <td><a href="https://huggingface.co/stabilityai/stable-diffusion-2-1-base">stable-diffusion-2-1-base</a></td>
  </tr>
  <tr>
    <td>1.5</td>
    <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">stable-diffusion-v1-5</a></td>
  </tr>
  <tr>
    <td>1.4</td>
    <td><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">stable-diffusion-v1-4</a></td>
  </tr>
  <tr>
    <td>stabilityai/stable-diffusion-x4-upscaler</td>
    <td><a href="https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler">stable-diffusion-x4-upscaler</a></td>
  </tr>
  <tr>
    <td>stabilityai/sd-x2-latent-upscaler</td>
    <td><a href="https://huggingface.co/stabilityai/sd-x2-latent-upscaler">stabilityai/sd-x2-latent-upscaler</a></td>
  </tr>

</table>

The DiffPure Model can be downloaded from [here](https://github.com/NVlabs/DiffPure?tab=readme-ov-file).Weu use the Guided Diffusion for ImageNet.

Please download the pretrain weights and define "$MODEL_PATH" in the script. 

> GPU allocation: We use eight 3090 GPUs to train and test. 

## Dataset 
We use the WikiArt dataset for training and testing. Example images are put into data/wikiart.

There are two face datasets: VGGFace2 and CelebA-HQ which are provided at [here](https://drive.google.com/drive/folders/1vlpmoKPZVgZZp-ANBzg915hOWPlCYv95?usp=sharing) (from Aiti-dreambooth paper).

For convenient testing, we have provided a split set of one subject in CelebA-HQ at `./data/CelebA-HQ/103` as the Anti-dreambooth does.

## How to run
The below script is designed to demonstrate the entire process of using StyleGuard for protecting artistic styles against style mimicry attacks. It includes steps to generate protective noises using StyleGuard, using Noise Upscaler to try to remove the protective noises, and then fine-tuning a model (DreamBooth) on these "denoised" examples. The final step involves generating test images to verify the effectiveness of the defense.

To defense Stable Diffusion version 1.4 (default), you can run
```
bash scripts/attack_ensemble_aspl_style.sh
```

It is supposed to take about an hour if run on 8 gpus.

Our dreambooth code is from the diffusers library. Please refer to the [diffusers](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) for more details.

Inference: generates examples with prompt
```
python infer.py \
  --model_path $DREAMBOOTH_OUTPUT_DIR \
  --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer \
  --prompt "an sks painting including a house"
```

Run evaluation:
suppose the reference folder is $REF_FOLDER, where the images are generated from a SD model finetuned on the clean images.
First, generate FID reference stats
```
# https://github.com/mseitzer/pytorch-fid
pip install pytorch-fid

python -m pytorch_fid --save-stats $REF_FOLDER evaluate/fid_ref.npz
```
Compute FID and Precision
```
python evaluate/eval_fid_new.py \
      --input_folder "$DREAMBOOTH_OUTPUT_DIR/checkpoint-$FINETUNE_STEP-test-infer/an_sks_painting_including_a_house" \
      --refer evaluate/fid_ref.npz \
      --output_folder "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=-1 python evaluate/eval_precision_new.py \
      --reference_folder $REF_FOLDER \
      --gen_folder "$DREAMBOOTH_OUTPUT_DIR/checkpoint-$FINETUNE_STEP-test-infer/an_sks_painting_including_a_house" \
      --output_folder "$OUTPUT_DIR"
```

Compute 
[CMMD](https://github.com/google-research/google-research/tree/master/cmmd) distance
```
conda run -n cmmd python -m cmmd.main --ref_folder=$REF_FOLDER \
  --gen_folder=$DREAMBOOTH_OUTPUT_DIR/checkpoint-$FINETUNE_STEP-test-infer/an_sks_painting_including_a_house \
  --max_num=12 \
  --batch_size=$NUM_GPUS \
  --output_folder $DREAMBOOTH_OUTPUT_DIR
```

run baselines (Anti-DreamBooth, SimAC)
```
bash attack_aspl.sh
bash attack_simac.sh
```

## Citation
Details of algorithms and experimental results can be found in our following paper:
```bibtex
@inproceedings{li2025styleguard,
  title={StyleGuard: Preventing Text-to-Image-Model-based Style Mimicry Attacks by Style Perturbations},
  author={Li, Yanjie and Zhang, Wenxuan and Lyu, Xinqi and Liu, Yihao and Xiao, Bin},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```


## Acknowledgement
This repo is based on [Anti-DB](https://github.com/VinAIResearch/Anti-DreamBooth) and [SimAC](https://github.com/somuchtome/SimAC). Thanks for their impressive works!


## Contacts
If you have any problems, please open an issue in this repository or send an email to [yanjie.li@connect.polyu.hk](mailto:yanjie.li@connect.polyu.hk).