import os
import sys
import json
import torch
from safetensors import safe_open

def main():
  hf_weights_dir = "/home/snehalv_google_com/test-ckpt-bf16"
  index_path = os.path.join(hf_weights_dir, "model.safetensors.index.json")
  
  if not os.path.exists(index_path):
    print(f"Error: Index file not found at {index_path}")
    return

  print(f"Loading index from {index_path}...")
  with open(index_path, "r") as f:
    index = json.load(f)

  weight_map = index["weight_map"]
  unique_files = set(weight_map.values())
  
  print(f"Scanning {len(unique_files)} safetensors files...")
  output_file = "hf_checkpoint_shapes.txt"
  
  with open(output_file, "w") as out:
    for filename in sorted(unique_files):
      file_path = os.path.join(hf_weights_dir, filename)
      with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
          if "kv_norm" in key or "attn" in key or "norm" in key:
            tensor_slice = f.get_slice(key)
            shape = tensor_slice.get_shape()
            out.write(f"{key}: shape={shape}\n")
            
  print(f"Done. Wrote matching key shapes to {output_file}")

if __name__ == "__main__":
  main()
