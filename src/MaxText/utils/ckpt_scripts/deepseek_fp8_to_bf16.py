# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Convert weights from FP8 to BF16 for a HF model.

Install these dependencies before running this script:

pip install torch==2.4.1 safetensors==0.4.5

Example cmd:

python3 -m MaxText.utils.ckpt_scripts.deepseek_fp8_to_bf16 --input-fp8-hf-path <path/to/fp8/ckpt> \
    --output-bf16-hf-path <local/path/to/save/new/bf16/ckpt>
"""


import os
import json
from argparse import ArgumentParser
from glob import glob

from tqdm import tqdm

import torch

from safetensors.torch import load_file, save_file


def weight_dequant_cpu(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
  """
  Dequantizes the given FP8 weight tensor using the provided scale tensor on CPU.

  Args:
      x (torch.Tensor): The quantized FP8 weight tensor of shape (M, N), dtype=torch.float8.
      s (torch.Tensor): The scale tensor, dtype=torch.bfloat16 or float32.
      block_size (int, optional): Size of the block used in quantization.

  Returns:
      torch.Tensor: The dequantized weight tensor, dtype=torch.bfloat16.

  Raises:
      AssertionError: If the input tensors are not 2D.
  """
  assert x.dim() == 2 and s.dim() == 2, "Both x and s must be 2D tensors"

  M, N = x.shape

  x = x.to(torch.float32)
  y = torch.empty_like(x, dtype=torch.get_default_dtype())

  for i in range(0, M, block_size):
    for j in range(0, N, block_size):
      row_start = i
      row_end = min(i + block_size, M)
      col_start = j
      col_end = min(j + block_size, N)
      block = x[row_start:row_end, col_start:col_end]
      scale = s[i // block_size, j // block_size]
      y[row_start:row_end, col_start:col_end] = (block * scale).to(torch.get_default_dtype())

  return y


def convert_fp8_to_bf16(fp8_path: str, bf16_path: str, cache_file_num: int = 2):
  """
  Converts a FP8 model to a BF16 model and saves the converted weights.

  This function reads FP8 weights from the specified directory, converts them to BF16,
  and saves the converted weights to another specified directory. It also updates the
  model index file to reflect the changes. The conversion process runs on CPU devices.

  Args:
      fp8_path (str): The path to the directory containing the FP8 weights and model index file.
      bf16_path (str): The path to the directory where the converted BF16 weights will be saved.

  Raises:
      KeyError: If a required scale_inv tensor is missing for a weight.

  Notes:
      - The function assumes that the FP8 weights are stored in safetensor files.
      - The function caches loaded safetensor files to optimize memory usage.
      - The function updates the model index file to remove references to scale_inv tensors.
  """
  torch.set_default_dtype(torch.bfloat16)
  os.makedirs(bf16_path, exist_ok=True)
  model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
  with open(model_index_file, "rt", encoding="utf8") as f:
    model_index = json.load(f)
  weight_map = model_index["weight_map"]

  # Cache for loaded safetensor files
  loaded_files = {}
  fp8_weight_names = []

  # Helper function to get tensor from the correct file
  def get_tensor(tensor_name):
    """
    Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

    Args:
        tensor_name (str): The name of the tensor to retrieve.

    Returns:
        torch.Tensor: The retrieved tensor.

    Raises:
        KeyError: If the tensor does not exist in the safetensor file.
    """
    file_name = weight_map[tensor_name]
    if file_name not in loaded_files:
      file_path = os.path.join(fp8_path, file_name)
      loaded_files[file_name] = load_file(file_path, device="cpu")
    return loaded_files[file_name][tensor_name]

  safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
  safetensor_files.sort()
  for safetensor_file in tqdm(safetensor_files):
    file_name = os.path.basename(safetensor_file)
    current_state_dict = load_file(safetensor_file, device="cpu")
    loaded_files[file_name] = current_state_dict

    new_state_dict = {}
    for weight_name, weight in current_state_dict.items():
      if weight_name.endswith("_scale_inv"):
        continue
      elif weight.element_size() == 1:  # FP8 weight
        scale_inv_name = f"{weight_name}_scale_inv"
        try:
          # Get scale_inv from the correct file
          scale_inv = get_tensor(scale_inv_name)
          fp8_weight_names.append(weight_name)
          new_state_dict[weight_name] = weight_dequant_cpu(weight, scale_inv)
        except KeyError:
          print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
          new_state_dict[weight_name] = weight
      else:
        new_state_dict[weight_name] = weight

    new_safetensor_file = os.path.join(bf16_path, file_name)
    save_file(new_state_dict, new_safetensor_file)

    # Memory management: keep only the `cache_file_num` most recently used files
    while len(loaded_files) > cache_file_num:
      oldest_file = next(iter(loaded_files))
      del loaded_files[oldest_file]

  # Update model index
  for weight_name in fp8_weight_names:
    scale_inv_name = f"{weight_name}_scale_inv"
    if scale_inv_name in weight_map:
      weight_map.pop(scale_inv_name)
  new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
  with open(new_model_index_file, "wt", encoding="utf8") as f:
    json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--input-fp8-hf-path", type=str, required=True)
  parser.add_argument("--output-bf16-hf-path", type=str, required=True)
  parser.add_argument("--cache-file-num", type=int, required=False, default=2)
  args = parser.parse_args()
  convert_fp8_to_bf16(args.input_fp8_hf_path, args.output_bf16_hf_path, args.cache_file_num)
