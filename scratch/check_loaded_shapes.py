import os
import sys
from typing import Sequence
import jax
import jax.numpy as jnp
from flax import nnx
from absl import app
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.configs import pyconfig
from maxtext.models.dit import DiT
from maxtext.utils import maxtext_utils
from maxtext.utils import max_utils

def main(argv: Sequence[str]) -> None:
    config = pyconfig.initialize(argv)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    model = DiT(config, mesh, rngs=nnx.Rngs(0))
    
    if config.load_parameters_path:
        from maxtext.common import checkpointing
        _, params, _ = nnx.split(model, nnx.Param, ...)
        checkpointing.load_params_from_path(
            config.load_parameters_path,
            params,
            config.checkpoint_storage_concurrent_gb,
            use_ocdbt=config.checkpoint_storage_use_ocdbt,
            use_zarr3=config.checkpoint_storage_use_zarr3,
        )
        print('Parameters loaded.')
        
        # Print shapes using jax.tree.map
        # params is (State, ...) or similar?
        # Let's check type
        print(f'Params type: {type(params)}')
        
        # If it is a tuple, iterate over it
        if isinstance(params, tuple):
            for i, p in enumerate(params):
                print(f'Tuple element {i} type: {type(p)}')
                # If it is state, try to iterate
                if hasattr(p, 'iter_flat'):
                    for path, val in p.iter_flat():
                        print(f'{path}: {val.value.shape if hasattr(val, value) else val.shape}')
        elif hasattr(params, 'iter_flat'):
             for path, val in params.iter_flat():
                 print(f'{path}: {val.value.shape if hasattr(val, value) else val.shape}')
        else:
            print('Cannot iterate params.')
            
    sys.exit(0)

if __name__ == '__main__':
    app.run(main)
