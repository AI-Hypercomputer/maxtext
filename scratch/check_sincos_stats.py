import numpy as np
import math

def generate_sincos_2d(seq_len, hidden_size):
    h = int(math.sqrt(seq_len))
    w = h
    assert h * w == seq_len, f"seq_len {seq_len} must be a perfect square"
    
    y_grid, x_grid = np.mgrid[:h, :w]
    width = hidden_size
    omega = np.arange(width // 4) / (width / 4.0)
    omega = 1.0 / (10000.0**omega)
    
    y = np.einsum("m,d->md", y_grid.flatten(), omega)
    x = np.einsum("m,d->md", x_grid.flatten(), omega)
    pe = np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1)
    
    return pe

# DiT-XL/2 uses seq_len=256 (for 32x32 latent grid) or seq_len=1024?
# Wait, latent size is 32x32 = 1024.
# Let's check config.max_target_length in the log or script.
# In decode_v2.sh: max_target_length=256.
# If seq_len=256, then side=16. 16x16=256.
# 16x16 patches of size 2x2 means 32x32 image?
# Wait, DiT-XL-2-256 means 256x256 image.
# Patch size 2 means 128x128 patches!
# So seq_len should be 128*128 = 16384?
# Let's check seq_len in sampler_v2.py log.

pe = generate_sincos_2d(256, 1152)
print(f"Stats for seq_len=256: min={pe.min():.4f}, max={pe.max():.4f}, std={pe.std():.4f}")

pe = generate_sincos_2d(1024, 1152)
print(f"Stats for seq_len=1024: min={pe.min():.4f}, max={pe.max():.4f}, std={pe.std():.4f}")

pe = generate_sincos_2d(4096, 1152)
print(f"Stats for seq_len=4096: min={pe.min():.4f}, max={pe.max():.4f}, std={pe.std():.4f}")
