import jax
import jax.numpy as jnp
from flax import linen as nn

class MyLayer(nn.Module):
  @nn.compact
  def __call__(self, x):
    w = self.param('w', nn.initializers.ones, (x.shape[-1],))
    return x * w

class MyDecoder(nn.Module):
  @nn.compact
  def __call__(self, x):
    # Old way
    # l1 = MyLayer(name="layer_0")(x)
    # New way
    class Boundary(nn.Module):
      @nn.compact
      def __call__(self, x):
        return MyLayer(name="layer_0")(x)
        
    Scanned = nn.scan(Boundary, variable_axes={}, variable_broadcast=True, in_axes=nn.broadcast, length=1)
    return Scanned(name="boundary")(x)

print(MyDecoder().init(jax.random.PRNGKey(0), jnp.ones((2,))))
