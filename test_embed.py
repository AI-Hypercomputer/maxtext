import torch
import numpy as np
from transformers import AutoModelForCausalLM
from safetensors import safe_open

hf_path = "Qwen/Qwen3-30B-A3B"

print("Loading model...")
hf_model = AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
hf_model.eval()

print("Loading safetensor...")
from huggingface_hub import snapshot_download
import os
import glob

# Try to find safetensors
model_dir = snapshot_download(hf_path, allow_patterns=["*.safetensors"])
sf_files = glob.glob(os.path.join(model_dir, "*.safetensors"))

sf_w = None
for f in sf_files:
    with safe_open(f, framework="pt", device="cpu") as sf:
        if "model.embed_tokens.weight" in sf.keys():
            sf_w = sf.get_tensor("model.embed_tokens.weight")
            break

print(f"safetensor weight: {sf_w.shape}, dtype {sf_w.dtype}")
print(f"pytorch weight: {hf_model.model.embed_tokens.weight.shape}, dtype {hf_model.model.embed_tokens.weight.dtype}")

# Test
torch.manual_seed(42)
dummy_input = torch.randint(0, 151936, (1, 128))

# PyTorch out
with torch.no_grad():
    pt_out = hf_model.model.embed_tokens(dummy_input).float().numpy()

# JAX/Numpy out
jax_w = sf_w.float().numpy()
jax_out = jax_w[dummy_input.numpy()]

diff = np.abs(pt_out - jax_out).max()
print(f"Max Diff: {diff}")

for i in range(128):
    d = np.abs(pt_out[0, i] - jax_out[0, i]).max()
    if d > 1e-3:
        print(f"Diff at token {i} (id {dummy_input[0, i]}): {d}")

