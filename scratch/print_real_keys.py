import os
import sys
from typing import Sequence
import jax
import jax.numpy as jnp
from flax import nnx
from absl import app
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.models.dit import DiT

def main(argv: Sequence[str]) -> None:
    config = pyconfig.initialize(argv)
    
    devices = jax.devices()
    mesh = Mesh(devices, ('data',))
    
    print("Instantiating DiT model...", flush=True)
    rng = jax.random.PRNGKey(0)
    nnx_rngs = nnx.Rngs(rng)
    
    model = DiT(config, mesh, rngs=nnx_rngs)
    print("DiT model instantiated.", flush=True)
    
    _, abstract_params_tree, _ = nnx.split(model, nnx.Param, ...)
    
    pure_dict = abstract_params_tree.to_pure_dict()
    print("\nPure Dict Structure:", flush=True)
    def print_keys(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                print_keys(v, prefix + str(k) + ".")
        else:
            # v might be a Parameter or array, but in pure dict it should be array-like with shape
            if hasattr(d, 'shape'):
                print(f"{prefix} -> Shape: {d.shape}")
            elif hasattr(d, 'value') and hasattr(d.value, 'shape'):
                print(f"{prefix} -> Shape: {d.value.shape}")
            else:
                print(f"{prefix} -> Type: {type(d)}")

    print_keys(pure_dict)

if __name__ == "__main__":
    app.run(main)
