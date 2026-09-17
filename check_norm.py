import pathlib
from safetensors import safe_open
import torch

path = "/mnt/disks/ds32/deepseekv3_checkpoint_bf16"
format = "safetensors"

ckpt_paths = sorted(pathlib.Path(path).glob(f"[!.]*.{format}"))

model_norm_name = "model.norm.weight"
mtp_norm_name = "model.layers.61.shared_head.norm.weight"
model_norm_weight = None
mtp_norm_weight = None

for i, ckpt_path in enumerate(ckpt_paths):
  print(f"Loading {ckpt_path.name} ({i+1}/{len(ckpt_paths)})...")
  with safe_open(ckpt_path, framework="pt") as f:
    keys = f.keys()
    if model_norm_weight is None and model_norm_name in keys:
      model_norm_weight = f.get_tensor(model_norm_name)
    if mtp_norm_weight is None and mtp_norm_name in keys:
      mtp_norm_weight = f.get_tensor(mtp_norm_name)

  if model_norm_weight is not None and mtp_norm_weight is not None:
    break

if model_norm_weight is None or mtp_norm_weight is None:
  raise ValueError(f"Missing weights. Found model_norm: {model_norm_weight is not None}, mtp_norm: {mtp_norm_weight is not None}")

diff = model_norm_weight - mtp_norm_weight

print(f"{model_norm_name}:\n\tmean: {model_norm_weight.mean()}\n\tweight: {model_norm_weight}")
print()
print(f"{mtp_norm_name}:\n\tmean: {mtp_norm_weight.mean()}\n\tweight: {mtp_norm_weight}")
print()
print(f"diff max abs: {diff.abs().max()}")
print(f"diff: {diff}")