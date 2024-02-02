import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
### REMEMBER export TPU_LIBRARY_PATH=/home/rwitten/maxtext/2024-02-05-16:47:29-libtpu.so
import functools
import jax
jax.config.update('jax_enable_memories', True)
jax.config.update('jax_traceback_filtering', "off")

import jaxlib
print(f"{jax.devices()=}, {jax.__version__=}, {jaxlib.__version__=}")
from jax import numpy as jnp
from jax.ad_checkpoint import Offloadable
import numpy as np




def fwd_bwd_jaxprs(f, *example_args):
  fwd_jaxpr, (y_shape, res_shape) = jax.make_jaxpr(
              lambda *args: jax.vjp(f, *args), return_shape=True)(*example_args)
  bwd_jaxpr = jax.make_jaxpr(lambda res, outs: res(outs))(res_shape, y_shape)
  return fwd_jaxpr, bwd_jaxpr

@functools.partial(jax.remat, policy=policy)
def f(x):
  x = jnp.sin(x)
  x = jnp.sin(x)
  return jnp.sum(x)

fwd_jaxpr, bwd_jaxpr = fwd_bwd_jaxprs(f, np.arange(16.))
print(f"{fwd_jaxpr=}")
print(f"{bwd_jaxpr=}")

f_with_grad = jax.jit(jax.value_and_grad(f))

value, grads = f_with_grad(np.arange(16.))

print(value)
