"""
Copyright 2025 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Convert gpt-oss model weights from FP4 to BF16/FP16/FP32 on GPU.

Example cmd:

python3 dequantize_mxfp4.py --input-path=<input_path> --output-path=<output_path>
python3 dequantize_mxfp4.py --input-path=<input_path> --output-path=<output_path> --dtype-str=bf16 --cache-size=2
"""

import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file


MODEL_INDEX_FILE = "model.safetensors.index.json"


# pylint: disable=line-too-long


# Reference from
# https://github.com/jax-ml/jax-llm-examples/blob/f53c6b87c09d49c59418ed9e3d2518620fa7c833/gpt_oss/gpt_oss_jax/chkpt_utils.py#L71
def e2m1_to_fp(x: torch.Tensor):
  "Convert E2M1 to FP32."
  x = torch.as_tensor(x, dtype=torch.int8)
  # The bit pattern is [S E E M]
  # Extract the Sign bit (S) from the most significant bit (bit 3)
  sign = 1 - 2 * ((x >> 3) & 0x01)
  # Extract the Exponent bits (E E) from bits 2 and 1
  exp = 2.0 ** (((x >> 1) & 0x3) - 1)
  # Calculate the Mantissa (M) based on bit 0
  is_subnormal = (x & 0b111) == 0b001
  m = torch.where(is_subnormal, 1.0, (1 / 2 * (x & 0x1) + 1).to(torch.float32))
  is_zero = (x & 0b111) == 0
  # The final value is Sign * Mantissa * Exponent
  return torch.where(is_zero, 0.0, m * exp * sign)


# Reference from
# https://github.com/jax-ml/jax-llm-examples/blob/f53c6b87c09d49c59418ed9e3d2518620fa7c833/gpt_oss/gpt_oss_jax/chkpt_utils.py#L81
def dequantize_mxfp4(blocks_2x_e2m1: torch.Tensor, scales_e8m0: torch.Tensor, data_type: torch.dtype):
  "Dequantize FP4 to desired data type."
  scales_e8m0 = torch.as_tensor(scales_e8m0, dtype=torch.float32)
  # Decode the scale factors
  scales = (2.0 ** (scales_e8m0 - 127))[..., None, None]
  # Unpack and dequantize the 4-bit Numbers
  x = torch.stack([e2m1_to_fp(blocks_2x_e2m1), e2m1_to_fp(blocks_2x_e2m1 >> 4)], -1) * scales
  # Reshape and cast to data type
  return x.reshape((x.shape[:-3] + (-1,))).to(data_type)


def main(input_path: str, output_path: str, target_dtype: torch.dtype, cache_size: int):
  """
  Converts FP4 weights on GPU.

  Args:
  input_path (str): The path to the directory containing the FP4 weights and model index file.
  output_path (str): The path to the directory where the converted weights will be saved.
  target_dtype (torch.dtype): The data type to convert the weights to.
  cache_size (int): The maximum number of files to cache in memory.
  """
  torch.set_default_dtype(target_dtype)
  os.makedirs(input_path, exist_ok=True)
  model_index_file = os.path.join(input_path, MODEL_INDEX_FILE)
  with open(model_index_file, "r", encoding="utf-8") as f:
    model_index = json.load(f)
  weight_map = model_index["weight_map"]

  loaded_files = {}
  fp4_weight_names = []
  block_suffix = "_blocks"
  scale_suffix = "_scales"

  def get_tensor(tensor_name):
    """Utility to get tensor from file."""
    file_name = weight_map[tensor_name]
    if file_name not in loaded_files:
      file_path = os.path.join(input_path, file_name)
      loaded_files[file_name] = load_file(file_path, device="cuda")
    return loaded_files[file_name][tensor_name]

  def update_model_index():
    """Update model index file."""
    new_model_index_file = os.path.join(output_path, MODEL_INDEX_FILE)
    for weight_name in fp4_weight_names:
      if weight_name.endswith(block_suffix):
        file_name = weight_map[weight_name]
        weight_map.pop(weight_name)
        weight_map[weight_name.removesuffix(block_suffix)] = file_name
      elif weight_name.endswith(scale_suffix):
        weight_map.pop(weight_name)

    print(f"weight_map: {weight_map}")
    with open(new_model_index_file, "w", encoding="utf-8") as f:
      json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

  safetensor_files = list(glob(os.path.join(input_path, "*.safetensors")))
  safetensor_files.sort()
  for safetensor_file in tqdm(safetensor_files):
    file_name = os.path.basename(safetensor_file)
    current_state_dict = load_file(safetensor_file, device="cuda")
    loaded_files[file_name] = current_state_dict

    new_state_dict = {}
    for weight_name, weight in current_state_dict.items():
      fp4_weight_names.append(weight_name)
      if weight_name.endswith(block_suffix):
        weight_prefix = weight_name.removesuffix(block_suffix)
        scale = get_tensor(weight_prefix + scale_suffix)
        dequantized_weight = dequantize_mxfp4(weight, scale, target_dtype)
        new_state_dict[weight_prefix] = dequantized_weight
        print(f"{weight_name}: Dequantized weight type is {dequantized_weight.dtype}")
      elif weight_name.endswith(scale_suffix):
        print(f"{weight_name}: Skip scale conversion as expected")
        continue
      else:
        if weight.dtype != target_dtype:
          weight = weight.to(target_dtype)
          print(f"{weight_name}: Original weight type is {weight.dtype}, and cast to {target_dtype}")
        else:
          print(f"{weight_name}: Original weight type is {weight.dtype}, and no conversion needed")
        new_state_dict[weight_name] = weight

    new_safetensor_file = os.path.join(output_path, file_name)
    save_file(new_state_dict, new_safetensor_file)

    # Memory management: keep only the most recently used files
    if len(loaded_files) > cache_size:
      oldest_file = next(iter(loaded_files))
      del loaded_files[oldest_file]
      torch.cuda.empty_cache()

  update_model_index()


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--input-path", type=str, required=True)
  parser.add_argument("--output-path", type=str, required=True)
  parser.add_argument("--dtype-str", type=str, required=False, default="bf16")
  parser.add_argument("--cache-size", type=int, required=False, default=2)
  args = parser.parse_args()
  dtype_map = {
      "bf16": torch.bfloat16,
      "bfloat16": torch.bfloat16,
      "f32": torch.float32,
      "float32": torch.float32,
      "f16": torch.float16,
      "float16": torch.float16,
  }
  parsed_dtype = dtype_map.get(args.dtype_str.lower())
  if parsed_dtype is None:
    raise ValueError(
        f"Unsupported dtype: {args.dtype_str}, please select one from bf16, bfloat16, f32, float32, f16, float16."
    )
  main(args.input_path, args.output_path, parsed_dtype, args.cache_size)
