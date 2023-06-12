import jax
import jax.numpy as jnp

# Set the seed for reproducibility
jax.random.PRNGKey(0)

# Define the shape of the arrays
array_shape = (10000, 10000)

# Create random arrays
array1 = jax.random.normal(jax.random.PRNGKey(0), shape=array_shape)
array2 = jax.random.normal(jax.random.PRNGKey(1), shape=array_shape)

res_0 = jnp.multiply(array1, array2)
# Multiply the arrays
for _ in range(50000):
    res_0 = jnp.multiply(array1, res_0)

# Print the result
print(res_0)