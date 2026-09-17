import torch
import numpy as np
import math
from diffusers.models.embeddings import get_2d_sincos_pos_embed

def generate_sincos_2d_custom(seq_len, hidden_size, base_size=16):
    h = int(math.sqrt(seq_len))
    w = h
    assert h * w == seq_len, f"seq_len {seq_len} must be a perfect square"
    
    scale_h = h / base_size
    scale_w = w / base_size
    
    grid_h = np.arange(h, dtype=np.float32) / scale_h
    grid_w = np.arange(w, dtype=np.float32) / scale_w
    
    x_grid, y_grid = np.meshgrid(grid_w, grid_h) # Default is indexing='xy'
    
    width = hidden_size
    omega = np.arange(width // 4) / (width / 4.0)
    omega = 1.0 / (10000.0**omega)
    
    y = np.einsum("m,d->md", y_grid.flatten(), omega)
    x = np.einsum("m,d->md", x_grid.flatten(), omega)
    pe = np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1)
    
    return pe

embed_dim = 1152
grid_size = 16 # 16x16 grid -> seq_len = 256

pe_hf = get_2d_sincos_pos_embed(embed_dim, grid_size, output_type="pt").numpy()
pe_custom_scaled = generate_sincos_2d_custom(grid_size*grid_size, embed_dim, base_size=16)

# Unscaled version (original custom logic)
def generate_sincos_2d_unscaled(seq_len, hidden_size):
    h = int(math.sqrt(seq_len))
    w = h
    y_grid, x_grid = np.mgrid[:h, :w]
    width = hidden_size
    omega = np.arange(width // 4) / (width / 4.0)
    omega = 1.0 / (10000.0**omega)
    y = np.einsum("m,d->md", y_grid.flatten(), omega)
    x = np.einsum("m,d->md", x_grid.flatten(), omega)
    pe = np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1)
    return pe

pe_custom_unscaled = generate_sincos_2d_unscaled(grid_size*grid_size, embed_dim)

print("HF PE shape:", pe_hf.shape)
print("Scaled Custom PE shape:", pe_custom_scaled.shape)
print("Unscaled Custom PE shape:", pe_custom_unscaled.shape)

print("Scaled close to HF?", np.allclose(pe_hf, pe_custom_scaled))
print("Unscaled close to HF?", np.allclose(pe_hf, pe_custom_unscaled))

if not np.allclose(pe_hf, pe_custom_unscaled):
    print("Difference (HF - Unscaled) max:", np.abs(pe_hf - pe_custom_unscaled).max())
    print("Difference (HF - Scaled) max:", np.abs(pe_hf - pe_custom_scaled).max())
