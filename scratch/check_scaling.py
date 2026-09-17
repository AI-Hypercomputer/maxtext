import torch
from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
if hasattr(pipe.vae.config, 'scaling_factor'):
    print(f"VAE scaling_factor: {pipe.vae.config.scaling_factor}")
else:
    print("No scaling_factor in VAE config")

if hasattr(pipe, 'scale_factor'):
     print(f"Pipeline scale_factor: {pipe.scale_factor}")
