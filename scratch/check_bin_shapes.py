import torch
import os
from huggingface_hub import snapshot_download

# Download snapshot to get the path
local_path = snapshot_download("facebook/DiT-XL-2-256")

transformer_file = os.path.join(local_path, "transformer", "diffusion_pytorch_model.bin")
if os.path.exists(transformer_file):
    print(f"Loading {transformer_file}")
    transformer_dict = torch.load(transformer_file, map_location='cpu')
    param_name = "transformer_blocks.0.ff.net.0.proj.weight"
    if param_name in transformer_dict:
        print(f"{param_name} shape: {transformer_dict[param_name].shape}")
    else:
        print(f"{param_name} not found in transformer_dict")
        print("Available keys:")
        for k in sorted(transformer_dict.keys()):
            if "ff.net.0" in k:
                print(f"  {k}: {transformer_dict[k].shape}")
else:
    print(f"{transformer_file} does not exist.")
