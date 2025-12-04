
import flax.linen as nn
import jax.numpy as jnp
from flax.linen import relu

class SimpleCNN(nn.Module):
  """A simple CNN model in Flax."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass of the SimpleCNN model.

    Args:
      x: The input tensor.

    Returns:
      The output tensor after processing through the CNN.
    """
    x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
    x = relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
    x = relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = x.reshape((x.shape[0], -1))  # Flatten

    x = nn.Dense(features=256)(x)
    x = relu(x)
    x = nn.Dense(features=10)(x)
    return x
