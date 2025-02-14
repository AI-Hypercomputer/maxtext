import glob
import os
import torch
from safetensors import safe_open

# checkpoint_folder = "~/tempdisk/Mixtral-8x7B-v0.1-Instruct"
checkpoint_folder = "/usr/local/google/home/lancewang/tempdisk/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

# print("Files in checkpoint_folder:")
# print(os.listdir(checkpoint_folder))

safetensor_files = glob.glob(os.path.join(checkpoint_folder, "*.safetensors"))

# print(safetensor_files, flush=True)

for i, st_f in enumerate(safetensor_files):
  with safe_open(st_f, framework="pt", device="cpu") as f:
    for key in f.keys():
      weight_tensor = f.get_tensor(key)
      parts = key.split(".")
      # if "layers" in key and int(parts[2]) != 0:
      #     continue
      print(f"This is file {i}, Weight name {key}, Shape: {weight_tensor.shape}, dtype: {weight_tensor.dtype}")

# checkpoint_folder = "/usr/local/google/home/lancewang/tempdisk/llama2-7B"

# pth_files = glob.glob(os.path.join(checkpoint_folder, "*.pth"))
# for i, f in enumerate(pth_files):
#   weight_tensor = torch.load(f, map_location="cpu")
#   for key, val in weight_tensor:
#     print(f"This is file {i}, Weight name {key}, weight shape {val.shape}")
