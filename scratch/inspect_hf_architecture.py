import torch
from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
print("Transformer Architecture:")
print(pipe.transformer)
