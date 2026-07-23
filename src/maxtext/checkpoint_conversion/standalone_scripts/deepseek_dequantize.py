# Copyright 2025 Google LLC
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

r"""Convert weights from FP8/FP4 to BF16 for a DeepSeek HF model.

The script can convert quantized weight in the following models:
- DeepSeek-V3, DeepSeek-V3.2 (fp8)
- Kimi-K2-Instruct (fp8)
- DeepSeek-V4-Flash-Base, DeepSeek-V4-Pro-Base (fp8)
- DeepSeek-V4-Flash, DeepSeek-V4-Pro (fp4 + fp8)

Example cmd:
python3 deepseek_dequantize.py --input-path <path/to/quantized/ckpt> \
    --output-path <local/path/to/save/new/bf16/ckpt>
"""

import os
import json
import torch
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
from safetensors.torch import load_file, save_file

# Lookup table for E2M1 FP4 (Two e2m1 nibbles packed per int8/uint8 byte)
_FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)


def dequantize_fp8(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
  """
  Dequantizes the given FP8 weight tensor using the provided scale tensor on CPU.
  """
  assert x.dim() == 2 and s.dim() == 2, "Both x and s must be 2D tensors"

  M, N = x.shape

  x = x.to(torch.float32)

  if s.dtype == torch.uint8:
    s_fp32 = (s.to(torch.float32) - 127.0).exp2()
  else:
    s_fp32 = s.to(torch.float32)

  # Check if dimensions are divisible by block_size
  if M % block_size != 0 or N % block_size != 0:
    s_expanded = torch.repeat_interleave(s_fp32, block_size, dim=0)
    s_expanded = torch.repeat_interleave(s_expanded, block_size, dim=1)
    s_expanded = s_expanded[:M, :N]
    return (x * s_expanded).to(torch.bfloat16)

  # Reshape and broadcast
  q = x.reshape(M // block_size, block_size, N // block_size, block_size)
  scale = s_fp32.reshape(M // block_size, 1, N // block_size, 1)

  return (q * scale).to(torch.bfloat16).reshape(M, N)


# reference: https://github.com/huggingface/transformers/blob/9a0fe3f5dd36ffe1888133f09eb03f1eb14b8a6e/src/transformers/integrations/finegrained_fp8.py#L987
def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
  """
  Unpacks packed E2M1 FP4 values (two 4-bit nibbles per byte) into standard float32 values.
  """
  u8 = packed.contiguous().view(torch.uint8)
  low = (u8 & 0xF).long()
  high = ((u8 >> 4) & 0xF).long()

  unpacked = torch.stack([_FP4_E2M1_LUT[low], _FP4_E2M1_LUT[high]], dim=-1)
  return unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])


# reference: https://github.com/huggingface/transformers/blob/9a0fe3f5dd36ffe1888133f09eb03f1eb14b8a6e/src/transformers/integrations/finegrained_fp8.py#L996
def dequantize_mxfp4(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
  """
  Dequantizes FP4 (E2M1) or FP8 (E4M3) weights using their block-wise scale grids.
  """
  fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
  is_fp4 = quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype)

  if is_fp4:
    quantized_fp32 = unpack_fp4(quantized)
  else:
    quantized_fp32 = quantized.to(torch.float32)

  rows, cols = quantized_fp32.shape[-2:]
  scale_rows, scale_cols = scales.shape[-2:]

  if rows % scale_rows != 0 or cols % scale_cols != 0:
    raise ValueError(f"Weight shape ({rows}, {cols}) not divisible by scale grid ({scale_rows}, {scale_cols}).")

  block_m = rows // scale_rows
  block_n = cols // scale_cols

  if scales.dtype == torch.uint8:
    s_fp32 = (scales.to(torch.float32) - 127.0).exp2()
  else:
    s_fp32 = scales.to(torch.float32)

  original_shape = quantized_fp32.shape
  q = quantized_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
  s = s_fp32.reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)

  return (q * s).to(torch.bfloat16).reshape(original_shape)


def convert_model(input_path: str, output_path: str, cache_file_num: int = 2):
  """
  Scans, converts, and saves a DeepSeek FP8/FP4 checkpoint directory to BF16.
  """
  torch.set_default_dtype(torch.bfloat16)
  os.makedirs(output_path, exist_ok=True)
  model_index_file = os.path.join(input_path, "model.safetensors.index.json")

  if not os.path.exists(model_index_file):
    raise FileNotFoundError(f"Could not locate {model_index_file}. Ensure the path is correct.")

  with open(model_index_file, "r", encoding="utf8") as f:
    model_index = json.load(f)
  weight_map = model_index["weight_map"]

  loaded_files = {}
  converted_scales = []

  def get_tensor(tensor_name):
    file_name = weight_map[tensor_name]
    if file_name not in loaded_files:
      file_path = os.path.join(input_path, file_name)
      loaded_files[file_name] = load_file(file_path, device="cpu")
    return loaded_files[file_name][tensor_name]

  safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))
  print(f"Found {len(safetensor_files)} weight shards to process...")

  for safetensor_file in tqdm(safetensor_files, desc="Converting Shards"):
    file_name = os.path.basename(safetensor_file)
    current_state_dict = load_file(safetensor_file, device="cpu")
    loaded_files[file_name] = current_state_dict

    new_state_dict = {}

    for name, tensor in current_state_dict.items():
      # Skip scale tensors; they will be integrated directly into the weights
      if name.endswith("_scale_inv") or name.endswith(".scale"):
        continue

      if tensor.dtype in (torch.int8, torch.float8_e4m3fn) or tensor.element_size() == 1:
        # Handle both DeepSeek-V3 (_scale_inv) and V4 (scale) naming conventions
        scale_name_v3 = f"{name}_scale_inv"
        scale_name_v4 = f"{name[:-len('.weight')]}.scale" if name.endswith(".weight") else None

        scale_inv = None
        used_scale_name = None

        try:
          scale_inv = get_tensor(scale_name_v3)
          used_scale_name = scale_name_v3
        except KeyError:
          if scale_name_v4:
            try:
              scale_inv = get_tensor(scale_name_v4)
              used_scale_name = scale_name_v4
            except KeyError:
              pass

        if scale_inv is not None:
          if tensor.dtype == torch.int8:
            dequantized_tensor = dequantize_mxfp4(tensor, scale_inv)
          elif tensor.dtype == torch.float8_e4m3fn:
            dequantized_tensor = dequantize_fp8(tensor, scale_inv, block_size=128)
          else:
            raise ValueError(f"Unrecognized dtype: {tensor.dtype}")

          new_state_dict[name] = dequantized_tensor
          converted_scales.append(used_scale_name)
        else:
          print(f"\nWarning: scale missing for {name}. Keeping original tensor.")
          new_state_dict[name] = tensor
      else:
        # Keep other non-quantized tensors (like biases, embeds, layer norms) intact in BF16
        new_state_dict[name] = tensor.to(torch.bfloat16)

    save_file(new_state_dict, os.path.join(output_path, file_name))

    # Memory management: keep only the `cache_file_num` most recently used files
    while len(loaded_files) > cache_file_num:
      oldest_file = next(iter(loaded_files))
      del loaded_files[oldest_file]

  # Clean up JSON Index Map
  print("Saving updated model index map...")
  for scale_name in set(converted_scales):
    if scale_name in weight_map:
      weight_map.pop(scale_name)

  new_model_index_file = os.path.join(output_path, "model.safetensors.index.json")
  with open(new_model_index_file, "w", encoding="utf8") as f:
    json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

  print(f"Successfully saved dequantized BF16 model to: {output_path}")


if __name__ == "__main__":
  parser = ArgumentParser(description="Dequantize DeepSeek hybrid checkpoints (FP8/FP4) to BF16.")
  parser.add_argument(
      "--input-path", type=str, required=True, help="Path to DeepSeek FP8/FP4 Hugging Face folder"
  )
  parser.add_argument(
      "--output-path", type=str, required=True, help="Directory to save output BF16 weights"
  )
  parser.add_argument(
      "--cache-size", type=int, default=2, help="Max cached files in RAM during indexing lookup"
  )
  args = parser.parse_args()

  if os.path.realpath(args.input_path) == os.path.realpath(args.output_path):
    raise ValueError("Input and output paths cannot be the same.")

  convert_model(args.input_path, args.output_path, args.cache_size)
