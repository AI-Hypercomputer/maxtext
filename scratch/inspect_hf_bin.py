import torch
import sys

bin_path = "/home/liyinn_google_com/.cache/huggingface/hub/models--facebook--DiT-XL-2-256/snapshots/eab87f77abd5aef071a632f08807fbaab0b704d0/transformer/diffusion_pytorch_model.bin"

print(f"Loading {bin_path}...")
state_dict = torch.load(bin_path, map_location="cpu")

print("pos_embed keys:")
for k in state_dict.keys():
    if "pos_embed" in k:
        print(f"{k}: {state_dict[k].shape}")
