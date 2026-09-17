from diffusers import DiTTransformer2DModel
import torch

model_id = "facebook/DiT-XL-2-256"

try:
    # Look for transformer in subfolder
    model = DiTTransformer2DModel.from_pretrained(model_id, subfolder="transformer")
    print("Transformer keys:")
    for k, v in model.state_dict().items():
        print(f"{k}: {v.shape}")
except Exception as e:
    print(f"Error: {e}")
