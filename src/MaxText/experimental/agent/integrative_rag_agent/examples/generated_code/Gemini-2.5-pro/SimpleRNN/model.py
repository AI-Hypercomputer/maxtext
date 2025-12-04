
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence


class SimpleRNN(nn.Module):
  """A simple multi-layer RNN model.

  Attributes:
    hidden_size: The number of features in the hidden state.
    num_layers: The number of recurrent layers.
    output_size: The number of output features.
  """

  hidden_size: int
  num_layers: int
  output_size: int

  def setup(self):
    """Initializes the RNN and Dense layers."""
    # In PyTorch nn.RNN, the output of layer l-1 is the input to layer l.
    # We can replicate this by creating a list of RNN layers and applying them
    # sequentially. Each nn.RNN wraps a single RNNCell.
    self.rnn_layers = [
        nn.RNN(nn.RNNCell(features=self.hidden_size), name=f'rnn_layer_{i}') for i in range(self.num_layers)
    ]
    self.fc = nn.Dense(features=self.output_size, name='fc')

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies the RNN model to the input sequence.

    Args:
      x: The input sequence of shape (batch_size, sequence_length, input_size).

    Returns:
      The output of the model of shape (batch_size, output_size).
    """
    # The PyTorch code initializes the hidden state with zeros.
    # Flax's nn.RNN does this by default if `initial_carry` is not provided.
    # We loop through the layers, passing the output sequence of one layer
    # as the input sequence to the next.
    current_sequence = x
    for rnn_layer in self.rnn_layers:
      # nn.RNN returns (final_carry, output_sequence)
      _, current_sequence = rnn_layer(current_sequence)

    # `current_sequence` is now the output of the last layer.
    # We take the output of the last time step for the final prediction.
    last_time_step_output = current_sequence[:, -1, :]

    out = self.fc(last_time_step_output)
    return out
