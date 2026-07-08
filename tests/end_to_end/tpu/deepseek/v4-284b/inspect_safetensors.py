from safetensors import safe_open

def main():
  path = "tests/end_to_end/tpu/deepseek/v4-284b/hf_tiny_model/model.safetensors"
  print(f"Opening safetensors at {path}...")
  with safe_open(path, framework="pt") as f:
    for k in sorted(f.keys()):
      tensor = f.get_slice(k)
      print(f"Key: {k}, Shape: {tensor.get_shape()}, Dtype: {f.get_tensor(k).dtype}")

if __name__ == "__main__":
  main()
