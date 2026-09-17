import numpy as np
import jax.numpy as jnp

def unpatchify_v1(latents, side, p, c):
    # What is currently in sampler.py
    # Assumes (C, P_h, P_w) layout?
    batch_size = latents.shape[0]
    latents_2d = latents.reshape(batch_size, side, side, c, p, p)
    latents_2d = jnp.transpose(latents_2d, (0, 1, 4, 2, 5, 3))
    latents_2d = latents_2d.reshape(batch_size, side * p, side * p, c)
    return latents_2d

def unpatchify_v2(latents, side, p, c):
    # Assumes (P_h, P_w, C) layout
    batch_size = latents.shape[0]
    latents_2d = latents.reshape(batch_size, side, side, p, p, c)
    latents_2d = jnp.transpose(latents_2d, (0, 1, 3, 2, 4, 5))
    latents_2d = latents_2d.reshape(batch_size, side * p, side * p, c)
    return latents_2d

# Create a sample image (Batch=1, H=4, W=4, C=1)
# 1 2 3 4
# 5 6 7 8
# 9 10 11 12
# 13 14 15 16

img = np.arange(1, 17).reshape(1, 4, 4, 1)
print("Original Image:")
print(img[0, :, :, 0])

# Patchify with p=2
# Side = 2
# Patches should be:
# P0: 1 2, 5 6
# P1: 3 4, 7 8
# P2: 9 10, 13 14
# P3: 11 12, 15 16

# Let's see how they get flattened

# Layout 1: C, P_h, P_w (here C=1, so P_h, P_w)
# P0 flattened: 1, 2, 5, 6
# P1 flattened: 3, 4, 7, 8

# Layout 2: P_h, P_w, C
# Since C=1, both layouts are identical in sequence of values if C is 1.

# Let's try C=2 to see difference.
img = np.arange(1, 33).reshape(1, 4, 4, 2)
print("Original Image (Channel 0):")
print(img[0, :, :, 0])
print("Original Image (Channel 1):")
print(img[0, :, :, 1])

# Standard patchify (channels last usually):
# Reshape to (B, side, p, side, p, c)
side = 2
p = 2
c = 2
reshaped = img.reshape(1, side, p, side, p, c)
# Transpose to (B, side, side, p, p, c)
# axes: B(0), H_side(1), P_h(2), W_side(3), P_w(4), C(5)
# target: B(0), H_side(1), W_side(3), P_h(2), P_w(4), C(5)
transposed = np.transpose(reshaped, (0, 1, 3, 2, 4, 5))
# Flatten
flattened = transposed.reshape(1, side * side, p * p * c)

print("\nFlattened shape:", flattened.shape)

# Now try unpatchify v1 and v2
unpatchified_v1 = unpatchify_v1(flattened, side, p, c)
unpatchified_v2 = unpatchify_v2(flattened, side, p, c)

print("\nUnpatchified V1 match:", np.allclose(img, unpatchified_v1))
print("Unpatchified V2 match:", np.allclose(img, unpatchified_v2))

if not np.allclose(img, unpatchified_v1):
    print("\nV1 error sample (Channel 0):")
    print(unpatchified_v1[0, :, :, 0])
if not np.allclose(img, unpatchified_v2):
    print("\nV2 error sample (Channel 0):")
    print(unpatchified_v2[0, :, :, 0])
