
import flax.linen as nn
from jax import Array

class CreditScoringMLP(nn.Module):
  """A simple MLP for credit scoring."""
  input_features: int

  def setup(self):
    """Initializes the layers of the MLP."""
    self.layer1 = nn.Dense(features=128)
    self.bn1 = nn.BatchNorm()
    self.dropout1 = nn.Dropout(rate=0.5)

    self.layer2 = nn.Dense(features=64)
    self.bn2 = nn.BatchNorm()
    self.dropout2 = nn.Dropout(rate=0.3)

    self.layer3 = nn.Dense(features=1)

  def __call__(self, x: Array, train: bool) -> Array:
    """Forward pass for the credit scoring MLP.

    Args:
      x: The input data.
      train: Whether the model is in training mode.

    Returns:
      The output logits.
    """
    x = self.layer1(x)
    x = self.bn1(x, use_running_average=not train)
    x = nn.relu(x)
    x = self.dropout1(x, deterministic=not train)

    x = self.layer2(x)
    x = self.bn2(x, use_running_average=not train)
    x = nn.relu(x)
    x = self.dropout2(x, deterministic=not train)

    # Output a single logit, which can be passed to a sigmoid for a probability
    x = self.layer3(x)
    return x
