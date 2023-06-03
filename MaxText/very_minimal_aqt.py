# git clone git@github.com:google/aqt.git
# Also modify most of the x | y, including 100% of the x | NoneType
from aqt.jax.v2 import aqt_dot_general as aqt
from aqt.jax.v2 import config as aqt_config

import jax
from jax import lax
import jax.numpy as jnp

import functools


#    dg = aqt.make_dot_general(self.cfg)
#     key = self.next_prng_key()
#     if not self.do_eval:
#       train_step = self.get_var('train_step')
#       self.update_var('train_step', train_step + 1)
#     # train_step starts from 0 and ends at exactly the total_train_step-1
#     train_step = self.get_var('train_step')
#     context = aqt.Context(key=key, train_step=train_step)
#     dg = functools.partial(dg, context=context)

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

        # add some fun key stuff
        key = jax.random.PRNGKey(0)
        context = aqt.Context(key=key, train_step=None) # None is recommended by Lukasz
        dg = functools.partial(aqt_dot_general, context=context)
        # equiv return aqt_dot_general(lhs, rhs, (((1,), (1,)), ((), ())), context=context)

        return dg(lhs, rhs, (((1,), (1,)), ((), ())))

def loss_fn(lhs, rhs):
    return jnp.linalg.norm(my_dot_general(lhs,rhs))

lhs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
rhs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

value = loss_fn(lhs, rhs) # This works with AQT
print("value from fn: ", value)

grad_fn = jax.value_and_grad(loss_fn, argnums = [0,1])
value, grad = grad_fn(lhs,rhs) # This does not work with AQT
print("value from grad_fn: ", value)
print("grad: ", grad)