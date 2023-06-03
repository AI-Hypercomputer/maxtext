# git clone git@github.com:google/aqt.git
# Also modify most of the x | y, including 100% of the x | NoneType
from aqt.jax.v2 import aqt_dot_general as aqt
from aqt.jax.v2 import config as aqt_config

import jax
from jax import lax
import jax.numpy as jnp


def my_dot_general(lhs, rhs):
    use_aqt = True

    if not use_aqt:
        return lax.dot_general(lhs, rhs, (((1,), (1,)), ((), ())))
    else:
        # create a quantization config
        aqt_cfg = aqt_config.fully_quantized(bits=8, use_fwd_quant=True)

        def noise_fn(shape, key):
            return jax.random.uniform(key, shape) - 0.5

        aqt_cfg.dlhs.lhs.noise_fn = noise_fn
        aqt_cfg.dlhs.rhs.noise_fn = noise_fn
        aqt_cfg.drhs.lhs.noise_fn = noise_fn
        aqt_cfg.drhs.rhs.noise_fn = noise_fn

        # use the config to create a quantized dot_general function
        aqt_dot_general = aqt.make_dot_general(aqt_cfg)
        return aqt_dot_general(lhs, rhs, (((1,), (1,)), ((), ())))

def loss_fn(lhs, rhs):
    return jnp.linalg.norm(my_dot_general(lhs,rhs))

grad_fn = jax.value_and_grad(loss_fn, argnums = [0,1])

lhs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
rhs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

value, grad = grad_fn(lhs,rhs)
print("value: ", value)
print("grad: ", grad)