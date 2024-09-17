import jax
from flax import nnx
from flax import linen as nn
from flax.core import meta
from flax.linen.spmd import LogicallyPartitioned

def _maybe_unbox_value(x):
  return x
  if isinstance(x, meta.Partitioned):
    return x.unbox()
  else:
    return x
  #return x.unbox() if isinstance(x, meta.Partitioned) else x

class LinenToNNX(nnx.Module):
  def __init__(self, module: nn.Module, rngs=None):
    assert rngs is not None, "You must provide `rngs=`"
    self.linen_module = module
    self.initialized, self.rngs, self.linen_state = False, rngs, None
    self.rngs = None
    # flax's flags to keep track of train / eval
    self.use_running_average, self.deterministic = False, False
    
  #@nnx.jit
  def __call__(self, *args, **kw):
    if not self.initialized:
      rngs = nnx.Rngs(0)
      #self.linen_state = self.linen_module.init(self.rngs(), *args, **kw)
      self.linen_state = self.linen_module.init(rngs(), *args, **kw)
      self.linen_state["params"] = jax.tree.map(lambda x: nnx.Param(
        _maybe_unbox_value(x)), self.linen_state["params"], 
        is_leaf=lambda x: isinstance(x, meta.Partitioned))
      for key in (set(self.linen_state.keys()) - set(["params"])):
        self.linen_state[key] = jax.tree.map(
          lambda x: nnx.Variable(_maybe_unbox_value(x)), self.linen_state[key], 
          is_leaf=lambda x: isinstance(x, meta.Partitioned))
      self.initialized = True
      #del self.rngs
    mutable_keys = [k for k in self.linen_state.keys() if k != "params"]
    linen_state = jax.tree.map(lambda x: x.value, self.linen_state)
    ret, updates = self.linen_module.apply(linen_state, *args, **kw, 
                                           mutable=mutable_keys)
    #print(f"Update keys: {mutable_keys}")
    if not self.use_running_average or not self.deterministic:
      updates = jax.tree.map(lambda x: nnx.Variable(x), updates)
      nnx.update(self, nnx.state({"linen_state": updates}))
    return ret