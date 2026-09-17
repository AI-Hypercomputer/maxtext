import torch
import sys

def print_shapes(ckpt_path):
    d = torch.load(ckpt_path, map_location='cpu')
    for k, v in d.items():
        print(f"{k}: {v.shape}")

if __name__ == "__main__":
    print_shapes(sys.argv[1])
