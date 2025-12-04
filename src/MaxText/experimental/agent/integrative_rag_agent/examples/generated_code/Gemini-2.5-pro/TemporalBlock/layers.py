
import flax.linen as nn
from jax import Array


class Chomp1d(nn.Module):
  """A 1D causal chop layer.

  This layer removes the last `chomp_size` elements from the temporal dimension
  of the input tensor. It is typically used in causal convolutional networks to
  ensure that the output at a given timestep does not depend on future inputs.

  Attributes:
    chomp_size: The number of elements to remove from the end of the sequence.
  """

  chomp_size: int

  def __call__(self, x: Array) -> Array:
    """Forward pass for the Chomp1d layer.

    Args:
      x: The input tensor of shape (batch, channels, sequence_length).

    Returns:
      The chopped tensor of shape (batch, channels, sequence_length - chomp_size).
    """
    return x[:, :, : -self.chomp_size]

import flax.linen as nn
from jax import Array


class Chomp1d(nn.Module):
  """A 1D chomping layer to make convolutions causal."""
  chomp_size: int

  @nn.compact
  def __call__(self, x: Array) -> Array:
    """Applies the chomping operation.

    Assumes input is of shape (batch, length, features).
    """
    if self.chomp_size == 0:
      return x
    return x[:, : -self.chomp_size, :]


class TemporalBlock(nn.Module):
  """A single temporal block for a TCN."""
  n_inputs: int
  n_outputs: int
  kernel_size: int
  stride: int
  dilation: int
  padding: int
  dropout: float = 0.2

  def setup(self):
    """Initializes the layers in the temporal block."""
    kernel_init_fn = nn.initializers.normal(stddev=0.01)

    self.conv1 = nn.weight_norm(
        nn.Conv(
            features=self.n_outputs,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding=self.padding,
            kernel_dilation=(self.dilation,),
            kernel_init=kernel_init_fn,
        )
    )
    self.chomp1 = Chomp1d(chomp_size=self.padding)
    self.dropout1 = nn.Dropout(rate=self.dropout)

    if self.n_inputs != self.n_outputs:
      self.downsample = nn.Conv(
          features=self.n_outputs,
          kernel_size=(1,),
          kernel_init=kernel_init_fn,
      )
    else:
      self.downsample = None

  def __call__(self, x: Array, *, deterministic: bool) -> Array:
    """Applies the temporal block to the input."""
    # Convolutional block
    out = self.conv1(x)
    out = self.chomp1(out)
    out = nn.relu(out)
    out = self.dropout1(out, deterministic=deterministic)

    # Residual connection
    res = x if self.downsample is None else self.downsample(x)

    return nn.relu(out + res)
