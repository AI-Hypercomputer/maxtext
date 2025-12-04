
import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple

# Note: The original PyTorch code includes an Attention class which is a dependency
# for LSTMAttentionForecaster. It is converted here as well for completeness.

class Attention(nn.Module):
  """A simple attention mechanism."""
  hidden_size: int

  @nn.compact
  def __call__(self, lstm_outputs: jnp.ndarray) -> jnp.ndarray:
    """Computes attention weights.

    Args:
      lstm_outputs: The outputs from the LSTM layer, with shape
        (batch_size, seq_len, hidden_size).

    Returns:
      The attention weights, with shape (batch_size, seq_len).
    """
    attn_dense = nn.Dense(features=self.hidden_size, name='attn')
    v = self.param('v', nn.initializers.uniform(), (self.hidden_size,))

    # lstm_outputs shape: (batch_size, seq_len, hidden_size)
    energy = nn.tanh(attn_dense(lstm_outputs))
    # energy shape: (batch_size, seq_len, hidden_size)

    energy = jnp.swapaxes(energy, 1, 2)
    # energy shape: (batch_size, hidden_size, seq_len)

    v_batched = jnp.expand_dims(
        jnp.tile(v, (lstm_outputs.shape[0], 1)), axis=1
    )
    # v_batched shape: (batch_size, 1, hidden_size)

    attn_weights = jnp.squeeze(jnp.matmul(v_batched, energy), axis=1)
    # attn_weights shape: (batch_size, seq_len)

    return nn.softmax(attn_weights, axis=1)


class LSTMAttentionForecaster(nn.Module):
  """An LSTM-based forecaster with an attention mechanism."""
  input_size: int
  hidden_size: int
  output_size: int

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass for the LSTMAttentionForecaster.

    Args:
      x: The input sequence, with shape (batch_size, seq_len, input_size).

    Returns:
      The output of the model, with shape (batch_size, output_size).
    """
    # Flax's LSTM infers input_size from the input `x`
    lstm_out, _ = nn.LSTM(features=self.hidden_size, name='lstm')(x)
    # lstm_out shape: (batch_size, seq_len, hidden_size)

    attn_weights = Attention(hidden_size=self.hidden_size, name='attention')(
        lstm_out
    )
    # attn_weights shape: (batch_size, seq_len)

    # Compute context vector by taking a weighted sum of LSTM outputs
    attn_weights_expanded = jnp.expand_dims(attn_weights, axis=1)
    # attn_weights_expanded shape: (batch_size, 1, seq_len)
    context = jnp.squeeze(jnp.matmul(attn_weights_expanded, lstm_out), axis=1)
    # context shape: (batch_size, hidden_size)

    # Final fully connected layer
    output = nn.Dense(features=self.output_size, name='fc')(context)
    # output shape: (batch_size, output_size)

    return output
