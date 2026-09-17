import torch
import sys

def print_keys(ckpt_path):
    d = torch.load(ckpt_path, map_location='cpu')
    for k in d.keys():
        print(k)

if __name__ == "__main__":
    print_keys(sys.argv[1])
