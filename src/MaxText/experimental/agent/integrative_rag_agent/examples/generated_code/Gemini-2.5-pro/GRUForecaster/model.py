
import flax.linen as nn
import jax.numpy as jnp
from jax import Array

class GRUForecaster(nn.Module):
  """A GRU-based time series forecaster."""
  input_size: int  # Kept for API parity with the original PyTorch model.
  hidden_size: int
  num_layers: int
  output_size: int

  @nn.compact
  def __call__(self, x: Array, deterministic: bool) -> Array:
    """Forward pass of the GRUForecaster.

    Args:
      x: The input sequence of shape (batch_size, sequence_length, input_size).
      deterministic: Disables dropout when set to True.

    Returns:
      The forecasted output of shape (batch_size, output_size).
    """
    # Initialize hidden state
    # In PyTorch, h0 is passed to the GRU layer. In Flax, the initial carry for
    # each layer in a stack is typically initialized separately.
    # Here we create a single tensor for conceptual clarity, then slice it.
    h0 = jnp.zeros((self.num_layers, x.shape[0], self.hidden_size), dtype=x.dtype)

    # GRU layers, which are often faster to train than LSTM
    # PyTorch's nn.GRU with num_layers > 1 is a stacked GRU.
    # We replicate this by iterating through layers.
    out = x
    for i in range(self.num_layers):
      # The initial carry for a Flax GRU layer is of shape (batch_size, features)
      initial_carry = h0[i]

      # The GRU layer in Flax returns (final_carry, outputs)
      _, out = nn.GRU(
          features=self.hidden_size, name=f'gru_{i}'
      )(initial_carry, out)

      # Apply dropout between layers, except for the last one
      if i < self.num_layers - 1:
        out = nn.Dropout(rate=0.2)(out, deterministic=deterministic)

    # Readout layer
    # Decode the hidden state of the last time step
    out = nn.Dense(features=self.output_size, name='fc')(out[:, -1, :])

    return out
