from flax import nnx
import jax.numpy as jnp

rngs = nnx.Rngs(0)
conv = nnx.Conv(in_features=3, out_features=10, kernel_size=(3, 3), rngs=rngs)
print("Conv kernel shape:", conv.kernel.value.shape)

lin = nnx.Linear(in_features=3, out_features=10, rngs=rngs)
print("Linear kernel shape:", lin.kernel.value.shape)

