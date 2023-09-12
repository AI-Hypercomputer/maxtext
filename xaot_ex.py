import jax
from jax.experimental.topologies import get_topology_desc
import numpy as np
from jax.experimental import pjit

tpu_ver='v4'
devices = get_topology_desc(
    platform='tpu',
    topology_name=f'{tpu_ver}:1x2x1',
    chip_config_name='megacore' if tpu_ver == 'v4' else 'default',
    chips_per_host_bounds=(1, 2, 1),
    num_slices=2,
).devices

# This is a strange alternative
#   devices = get_topology_desc(
#       platform='tpu',
#       num_slices=2,
#   ).devices


def fun(x):
    return x * x

with jax.sharding.Mesh(np.array(devices), ('data',)):
    jitted = pjit.pjit(
        fun, in_shardings=P('data'), out_shardings=P(None, 'data')
    )
    lowered = jitted.lower(
        jax.core.ShapedArray(shape=(128, 128), dtype=np.float32)
    )
orig_compiled = lowered.compile()