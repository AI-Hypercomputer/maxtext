from diffusers import DiTPipeline
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
emb = pipe.transformer.transformer_blocks[0].norm1.emb
print("Time Proj:", emb.time_proj)
if hasattr(emb.time_proj, 'num_features'):
     print("  num_features:", emb.time_proj.num_features)
if hasattr(emb.time_proj, 'flip_sin_to_cos'):
     print("  flip_sin_to_cos:", emb.time_proj.flip_sin_to_cos)
if hasattr(emb.time_proj, 'downscale_freq_shift'):
     print("  downscale_freq_shift:", emb.time_proj.downscale_freq_shift)

# Let's check max_wavelength or similar
if hasattr(emb.time_proj, 'max_period'): # Sometimes called max_period
     print("  max_period:", emb.time_proj.max_period)
