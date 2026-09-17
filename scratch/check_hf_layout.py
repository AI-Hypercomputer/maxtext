import torch
import torch
import torch.nn as nn
import numpy as np

# Create a Conv2d layer similar to DiT PatchEmbed
# latent channels = 4, hidden_size = 1152, patch_size = 2
proj = nn.Conv2d(in_channels=4, out_channels=1152, kernel_size=2, stride=2)


print("PatchEmbed proj weight shape:", proj.weight.shape)

# Let's see how it applies to input
# Input shape: (B, C, H, W) -> (1, 4, 2, 2) for a single patch
x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 2, 2)
print("\nInput tensor (Channel 0):")
print(x[0, 0])
print("Input tensor (Channel 1):")
print(x[0, 1])

# Apply proj
out = proj(x)
print("\nOutput shape before flatten:", out.shape) # Should be (1, 1152, 1, 1)

# Flatten
out_flat = out.flatten(2).transpose(1, 2)
print("Output shape after flatten/transpose:", out_flat.shape) # Should be (1, 1, 1152)

# Manual Conv calculation for channel 0 of weight
# proj.weight has shape (1152, 4, 2, 2)
# Let's look at weight for output channel 0
w0 = proj.weight[0]
print("\nWeight[0] shape:", w0.shape)

# Manual convolution of x[0] and w0
manual_out = torch.sum(x[0] * w0)
print("Manual convolution out:", manual_out.item())
print("Actual proj out[0, 0, 0, 0]:", out[0, 0, 0, 0].item())

# Now let's see how the hook flattens this weight
# hook: x_flat = x.reshape(x.shape[0], -1)
w_flat = proj.weight.detach().numpy().reshape(proj.weight.shape[0], -1)
print("\nFlattened weight[0] shape:", w_flat[0].shape)
print("Flattened weight[0] elements:", w_flat[0][:4])

# If we treat x as flat vector of size 16, how should it be ordered to match w_flat[0]?
# x has shape (4, 2, 2). Flattened in row-major:
x_flat_row_major = x.flatten().numpy()
print("\nX flattened row-major elements:", x_flat_row_major[:4])

# Try with custom flattening order in sampler_v2
# (P_h, P_w, C) -> (C, P_h, P_w)
# In sampler_v2:
# x_reshaped = x.reshape(B, L, p, p, c)
# x_converted = jnp.transpose(x_reshaped, (0, 1, 4, 2, 3))
# x_flattened = x_converted.reshape(B, L, p * p * c)

# Let's emulate this for a single batch/seq element:
# x has shape (1, 4, 2, 2) which is (B, C, H, W).
# For standard flattening it is row-major of (C, H, W).
# In sampler_v2, we start with Noise of shape (B, L, P_h * P_w * C).
# Let's say noise is generated as standard JAX normal.
# If we reshape it to (p, p, c) and then transpose to (c, p, p) before flattening,
# does it match row-major of (c, p, p)?

c = 4
p = 2
noise = np.arange(1, 17).reshape(p, p, c)
print("\nNoise (P_h=0, P_w=0):", noise[0,0])

noise_converted = np.transpose(noise, (2, 0, 1))
noise_flattened = noise_converted.flatten()

# Standard tensor (B, C, P_h, P_w) equivalent:
tensor_eq = np.zeros((c, p, p))
for i in range(p):
    for j in range(p):
        for k in range(c):
            # noise was generated as (p, p, c)
            # noise[i, j, k] corresponds to P_h=i, P_w=j, C=k
            tensor_eq[k, i, j] = noise[i, j, k]

tensor_eq_flattened = tensor_eq.flatten()

print("Noise flattened:", noise_flattened[:4])
print("Tensor eq flattened:", tensor_eq_flattened[:4])
print("Do they match?", np.allclose(noise_flattened, tensor_eq_flattened))

# Now check if this flattening matches the weight flattening
# w_flat elements: [-0.08347085  0.1263273  -0.03758934  0.14736208]
# They come from torch.reshape(1152, 4, 2, 2) -> (1152, 16)
# So w_flat[0] elements are row-major of (4, 2, 2)
# Which is:
# C=0, H=0, W=0
# C=0, H=0, W=1
# C=0, H=1, W=0
# C=0, H=1, W=1
# C=1, H=0, W=0
# Let's verify element-wise multiplication equivalence
# w_flat[0] is row-major of (4, 2, 2)
# Noise flattened is row-major of (4, 2, 2) IF noise was (P_h, P_w, C) and we transposed to (C, P_h, P_w)

# Let's verify this explicitly.
w_manual = proj.weight[0].detach().numpy() # shape (4, 2, 2)

# Our noise was (2, 2, 4)
# Let's reshape it to match tensor_eq which is (4, 2, 2)
noise_tensor = np.zeros((c, p, p))
for i in range(p):
    for j in range(p):
        for k in range(c):
            noise_tensor[k, i, j] = noise[i, j, k]

# Manual convolution equivalent (element-wise mult and sum)
manual_sum = np.sum(noise_tensor * w_manual)

# Dot product of flattened arrays
dot_sum = np.dot(noise_flattened, w_flat[0])

print("\nManual sum of tensors:", manual_sum)
print("Dot sum of flattened:", dot_sum)
print("Do sums match?", np.allclose(manual_sum, dot_sum))

# If they match, then my layout conversion in sampler_v2 MUST be correct
# IF the weights were converted correctly.
# The weights were converted with flatten_and_transpose_conv:
# x_flat = x.reshape(x.shape[0], -1) -> [out, 16]
# return x_flat.T -> [16, out]
# So MaxText linear layer receives [16, out]
# And input should be [B, L, 16]
# If input is [B, L, 16], then the dot product with linear layer kernel [16, out]
# is exactly np.dot(input[b, l], kernel).
# So input[b, l] must have elements in same order as w_flat[0] (which is row-major of 4x2x2).


