from diffusers import AutoencoderKL
import torch

vae = AutoencoderKL.from_pretrained("facebook/DiT-XL-2-256", subfolder="vae")

# Inspect UpDecoderBlock2D forward
import inspect
print(inspect.getsource(vae.decoder.up_blocks[0].__class__.forward))
