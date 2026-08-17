import jax
import jax.numpy as jnp

tap_pure = jnp.ones((2, 4, 32, 64, 64))
tap_hybrid = jnp.zeros((2, 8, 32, 64, 64))

print("Valid size:", min(tap_pure.flatten().size, tap_hybrid.flatten().size))
