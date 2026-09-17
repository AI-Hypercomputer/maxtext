from diffusers import DiTPipeline
import torch

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")

# Create dummy inputs
batch_size = 2
seq_len = pipe.transformer.config.sample_size * pipe.transformer.config.sample_size
latent_channels = pipe.transformer.config.in_channels
patch_size = pipe.transformer.config.patch_size

# HF expects inputs as image/latent maps
dummy_latents = torch.randn(batch_size, latent_channels, pipe.transformer.config.sample_size, pipe.transformer.config.sample_size)
dummy_timesteps = torch.tensor([10] * batch_size)
dummy_class_labels = torch.tensor([1, 2])

# Forward pass
output = pipe.transformer(dummy_latents, dummy_timesteps, dummy_class_labels).sample

print(f"HF Output shape: {output.shape}")

# Let's see how it is derived from internal linear
# We can check the source of FinalLayer in HF if it exists, or just the end of forward.
# print(inspect.getsource(pipe.transformer.__class__.forward))

