# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main entry point for the CMMD calculation."""
import sys
sys.path.append("/home/yjli/AIGC/diffusers/SimAC/cmmd")
sys.path.append("/home/yjli/AIGC/diffusers/SimAC/")

from cmmd import distance
from cmmd import embedding
from cmmd import io_util
import numpy as np

import argparse
import os
import sys
import torch
existing_pythonpath = os.environ.get('PYTHONPATH', '')
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
print(existing_pythonpath)
# Construct the new PYTHONPATH
    
# Append the new path to the existing PYTHONPATH
new_path = '/home/yjli/AIGC/diffusers/SimAC/scenic'

# Construct the new PYTHONPATH string
if existing_pythonpath:
    new_pythonpath = f"{existing_pythonpath}:{new_path}"
else:
    new_pythonpath = new_path
# Set the new PYTHONPATH
os.environ['PYTHONPATH'] = new_pythonpath
# Also add to sys.path to ensure it's recognized in this session
sys.path.append(new_path)

def compute_cmmd(
    ref_dir, eval_dir, batch_size = 32, max_count = -1
):
  """Calculates the CMMD distance between reference and eval image sets.

  Args:
    ref_dir: Path to the directory containing reference images.
    eval_dir: Path to the directory containing images to be evaluated.
    batch_size: Batch size used in the CLIP embedding calculation.
    max_count: Maximum number of images to use from each directory. A
      non-positive value reads all images available except for the images
      dropped due to batching.

  Returns:
    The CMMD value between the image sets.
  """
  embedding_model = embedding.ClipEmbeddingModel()
  ref_embs = io_util.compute_embeddings_for_dir(
      ref_dir, embedding_model, batch_size, max_count
  )
  eval_embs = io_util.compute_embeddings_for_dir(
      eval_dir, embedding_model, batch_size, max_count
  )
  val = distance.mmd(ref_embs, eval_embs)
  return np.asarray(val)


def main(args):
  with torch.no_grad():
    result = compute_cmmd(args.ref_folder, args.gen_folder, args.batch_size, args.max_num)
  cmmd_value =  str(result.item() if hasattr(result, 'item') else float(result))
  with open(args.output_folder+"/cmmd.txt",'w') as f:
    f.write('cmmd:' + cmmd_value)
  print('The CMMD value is: ' + str(result.item() if hasattr(result, 'item') else float(result)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute LPIPS distance between two image folders.')
    parser.add_argument('--ref_folder', type=str, default='/home/yjli/AIGC/diffusers/SimAC/data/wikiart/vangogh')
    parser.add_argument('--gen_folder', type=str, default='/home/yjli/AIGC/diffusers/SimAC/data/wikiart/11')  
    parser.add_argument('--output_folder', type=str, default='/home/yjli/AIGC/diffusers/SimAC/data/wikiart/')  
    parser.add_argument('--max_num', type=int, default=20, help='Path to the second image folder.')
    parser.add_argument('--batch_size', type=int, default=4, help='Path to the second image folder.')# Parse the input argument# Parse the input arguments
    args = parser.parse_args()
    main(args)
