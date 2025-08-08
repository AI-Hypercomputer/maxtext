import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file


# Reference from https://github.com/jax-ml/jax-llm-examples/blob/f53c6b87c09d49c59418ed9e3d2518620fa7c833/gpt_oss/gpt_oss_jax/chkpt_utils.py#L71
def e2m1_to_fp(x):
  "Convert E2M1 to FP32."
  x = torch.as_tensor(x, dtype=torch.int8)
  sign = 1 - 2 * ((x >> 3) & 0x01)
  exp = 2.0 ** (((x >> 1) & 0x3) - 1)
  is_subnormal = (x & 0b111) == 0b001
  m = torch.where(is_subnormal, 1.0, (1 / 2 * (x & 0x1) + 1).to(torch.float32))
  is_zero = (x & 0b111) == 0
  return torch.where(is_zero, 0.0, m * exp * sign)


# Reference from https://github.com/jax-ml/jax-llm-examples/blob/f53c6b87c09d49c59418ed9e3d2518620fa7c833/gpt_oss/gpt_oss_jax/chkpt_utils.py#L81
def dequantize_mxfp4(blocks_2x_e2m1: torch.Tensor, scales_e8m0: torch.Tensor):
  "Dequantize FP4 to desired data type BF16."
  scales_e8m0 = torch.as_tensor(scales_e8m0, dtype=torch.float32)
  scales = (2.0 ** (scales_e8m0 - 127))[..., None, None]
  x = torch.stack([e2m1_to_fp(blocks_2x_e2m1), e2m1_to_fp(blocks_2x_e2m1 >> 4)], -1) * scales
  return x.reshape((x.shape[:-3] + (-1,))).to(torch.bfloat16)


def main(input_path, output_path):
  """
  Converts FP4 weights to BF16 on GPU.

  Args:
  input_path (str): The path to the directory containing the FP4 weights and model index file.
  output_path (str): The path to the directory where the converted weights will be saved.
  """
  torch.set_default_dtype(torch.bfloat16)
  os.makedirs(input_path, exist_ok=True)
  model_index_file = os.path.join(input_path, "model.safetensors.index.json")
  with open(model_index_file, "r") as f:
    model_index = json.load(f)
  weight_map = model_index["weight_map"]

  loaded_files = {}
  fp4_weight_names = []
  block_suffix = "_blocks"
  scale_suffix = "_scale"
  cache_size = 2

  def get_tensor(tensor_name):
    """Utility to get tensor from file."""
    file_name = weight_map[tensor_name]
    if file_name not in loaded_files:
      file_path = os.path.join(input_path, file_name)
      loaded_files[file_name] = load_file(file_path, device="cuda")
    return loaded_files[file_name][tensor_name]

  def update_model_index():
    new_model_index_file = os.path.join(output_path, "model.safetensors.index.json")
    for weight_name in fp4_weight_names:
      is_to_remove = weight_name.endswith(block_suffix) or weight_name.endswith(scale_suffix)
      if is_to_remove and weight_name in weight_map:
        weight_map.pop(weight_name)
    with open(new_model_index_file, "w") as f:
      json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

  safetensor_files = list(glob(os.path.join(input_path, "*.safetensors")))
  safetensor_files.sort()
  for safetensor_file in tqdm(safetensor_files):
    file_name = os.path.basename(safetensor_file)
    current_state_dict = load_file(safetensor_file, device="cuda")
    loaded_files[file_name] = current_state_dict

    new_state_dict = {}
    for weight_name, weight in current_state_dict.items():
      if weight_name.endswith(block_suffix):
        weight_prefix = weight_name.removesuffix(block_suffix)
        scale = get_tensor(weight_prefix + scale_suffix)
        dequantized_weight = dequantize_mxfp4(weight, scale)
        new_state_dict[weight_prefix] = dequantized_weight
      else:
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
  args = parser.parse_args()
  main(args.input_path, args.output_path)
