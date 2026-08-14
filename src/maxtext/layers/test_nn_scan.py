import jax
import jax.numpy as jnp
from flax import linen as nn

class MyLayer(nn.Module):
  @nn.compact
  def __call__(self, x):
    w = self.param('w', nn.initializers.ones, (x.shape[-1],))
    self.sow('intermediates', 'w_out', w * x)
    return x * w

class MyDecoder(nn.Module):
  @nn.compact
  def __call__(self, x):
    layer = MyLayer(name="layer_0")
    
    def run_layer(mdl, carry, _):
       x = carry
       out = layer(x)
       return out, None
       
    scanned_fn = nn.scan(
        run_layer,
        variable_axes={},
        variable_broadcast=True,
        split_rngs={'params': False},
        in_axes=nn.broadcast,
        length=1,
    )
    
    out, _ = scanned_fn(self, x, None)
    return out

key = jax.random.PRNGKey(0)
x = jnp.ones((2,))
model = MyDecoder()
vars = model.init(key, x)
print(vars.keys())
print("intermediates:", vars.get("intermediates"))
out, vars = model.apply(vars, x, mutable=['intermediates'])
print("out:", out)
print("intermediates after apply:", vars.get("intermediates"))
