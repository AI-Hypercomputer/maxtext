from diffusers import DiTPipeline
import torch

# Load pipeline
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")

# Check scaling factor in VAE
print(f"VAE scaling factor: {pipe.vae.config.scaling_factor}")

# Let's see if there is any scaling in the pipeline itself
import inspect
print(inspect.getsource(pipe.__call__))
