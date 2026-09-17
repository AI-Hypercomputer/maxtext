from flax import nnx
import jax.numpy as jnp

rngs = nnx.Rngs(0)
conv = nnx.Conv(in_features=3, out_features=6, kernel_size=(3, 3), rngs=rngs)

print("Kernel shape:", conv.kernel.value.shape)
