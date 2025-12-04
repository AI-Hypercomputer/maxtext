
import flax.linen as nn
import jax
from jax import numpy as jnp

class LSTMForecaster(nn.Module):
  """LSTM Forecaster model in Flax."""
  input_size: int
  hidden_size: int
  num_layers: int
  output_size: int
  dropout_rate: float = 0.2

  def setup(self):
    """Initializes the layers of the LSTM Forecaster."""
    # A list of LSTM layers and Dropout layers
    self.lstm_layers = [
        nn.RNN(nn.LSTMCell(features=self.hidden_size), name=f'lstm_{i}')
        for i in range(self.num_layers)
    ]
    # Dropout is applied on the outputs of each LSTM layer except the last layer
    if self.num_layers > 1:
      self.dropout_layers = [
          nn.Dropout(rate=self.dropout_rate, name=f'dropout_{i}')
          for i in range(self.num_layers - 1)
      ]
    # Fully connected layer to map the LSTM output to the desired output size
    self.fc = nn.Dense(features=self.output_size, name='fc')

  def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """
    Applies the LSTM Forecaster model to the input sequence.

    Args:
      x: The input sequence, with shape (batch, seq_length, feature).
      deterministic: If True, dropout is not applied.

    Returns:
      The forecasted output, with shape (batch, output_size).
    """
    current_input = x

    for i in range(self.num_layers):
      lstm_layer = self.lstm_layers[i]

      # Pass through the layer. initial_carry will be created internally with zeros.
      outputs = lstm_layer(current_input)

      # Apply dropout between layers (except after the last one)
      if self.num_layers > 1 and i < self.num_layers - 1:
        outputs = self.dropout_layers[i](outputs, deterministic=deterministic)

      current_input = outputs

    # We take the output from the last time step for prediction
    last_output = current_input[:, -1, :]

    # Pass through the fully connected layer
    out = self.fc(last_output)
    return out
