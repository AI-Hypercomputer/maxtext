"""
cd ~/maxtext
python -m MaxText.muon_test2
"""

import math
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from optax.transforms import _masking
# from optax.contrib import _muon
# from optax.contrib import muon
from MaxText import muon
from MaxText.muon_dimension_number import DEEPSEEK2_DIMENSION_NUMBER, DEEPSEEK3_DIMENSION_NUMBER, get_transform_tree, get_abstract_param
import jax
from flax.linen import partitioning as nn_partitioning
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.layers import models, quantizations
import optax
from jax import numpy as jnp
import numpy as np
from MaxText.globals import MAXTEXT_PKG_DIR
import os
Transformer = models.transformer_as_linen
import sys
from MaxText import maxtext_utils
import functools


class DataIter:

  def __init__(self, rng, max_target_length, global_batch_size_to_train_on, base_emb_dim):
    self.max_target_length = max_target_length
    self.global_batch_size = global_batch_size_to_train_on
    self.embed_dim = base_emb_dim
    self.rng = rng

  def get_data(self):
    """get data"""
    lnx = jax.random.randint(self.rng, (self.global_batch_size, self.max_target_length), 0, 100, dtype=jax.numpy.int32)
    decoder_segment_ids = jax.random.randint(self.rng, (self.global_batch_size, self.max_target_length), 0, 4)
    decoder_positions = jax.random.randint(
        self.rng, (self.global_batch_size, self.max_target_length), 0, self.max_target_length
    )
    return lnx, decoder_segment_ids, decoder_positions


"""
get model
"""
model_name = "deepseek3-test"
argv = [
    None,
    #"MaxText/configs/base.yml",
    os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
    "per_device_batch_size=1",
    "max_target_length=2048",
    #"skip_jax_distributed_system=true",
    "ici_fsdp_parallelism=4",
    f"model_name={model_name}",
]
config = pyconfig.initialize(argv)
rng = jax.random.PRNGKey(0)
devices_array = maxtext_utils.create_device_mesh(config)
mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
quant = quantizations.configure_quantization(config)
model = Transformer(config, mesh=mesh, quant=quant)


"""
get data
"""
data = DataIter(rng, config.max_target_length, config.global_batch_size_to_train_on, config.base_emb_dim)
lnx, decoder_segment_ids, decoder_positions = data.get_data()
"""
init model
"""


# maxtext_state, _ = maxtext_utils.setup_decode_state(maxtext_model, config, rng, mesh, None)
# print(maxtext_state)
# breakpoint()
# #sys.exit(1)
# model_vars = maxtext_state.params
# print(model_vars)
# breakpoint()


# model_vars = maxtext_model.init(
#     {"params": rng, "aqt": rng},
#         decoder_input_tokens=lnx,
#         decoder_positions=decoder_positions,
#         decoder_segment_ids=decoder_segment_ids,
# )

# print(model_vars.keys())
# print(model_vars)
# sys.exit(1)
# print(model_vars["params"])
# model_vars = model_vars["params"]

"""
get optimizer
"""
# muon_weight_dimension_numbers = DEEPSEEK3_DIMENSION_NUMBER
abstract_param = get_abstract_param(model, config)
muon_weight_dimension_numbers  = get_transform_tree(abstract_param)
assert muon_weight_dimension_numbers == DEEPSEEK3_DIMENSION_NUMBER


learning_rate = 1e-3
muon_kwargs = {
    # Shared parameters: "nesterov" uses default
    "learning_rate": learning_rate,
    "eps": config.adam_eps,
    "mu_dtype": config.mu_dtype,
    # Muon-specific parameters: "ns_coeffs", "ns_steps", "weight_decay_mask", "adaptive" uses default
    "beta": config.muon_beta,
    "weight_decay": config.muon_weight_decay,
    "muon_weight_dimension_numbers": muon_weight_dimension_numbers,
    # AdamW-specific parameters
    "adam_b1": config.adam_b1,
    "adam_b2": config.adam_b2,
    "adam_eps_root": config.adam_eps_root,
    "adam_weight_decay": config.adam_weight_decay,
}
tx = muon.muon(**muon_kwargs)

"""
init, from inner to outer
[maxtext_utils.setup_initial_state, jit?] -> maxtext_utils.setup_training_state -> train_utils.setup_train_loop 
-> sft_trainer.train_loop  (train.train_loop complicated) 
<- train_utils.jit_train_and_eval_step <- [train_utils.jit_train_step, jit?] <- train.train_step, maxtext_utils.get_functional_train_with_signature
"""
is_training = True
unboxed_abstract_state, state_mesh_annotations, state_mesh_shardings = maxtext_utils.get_abstract_state(
    model, tx, config, rng, mesh, is_training
)
print("\n1", state_mesh_shardings)
init_state_partial = functools.partial(maxtext_utils.init_initial_state, model, tx, config, is_training)
init_state_partial.__name__ = "initialize_state"
# pylint: disable=not-callable
state = jax.jit(
    init_state_partial,
    in_shardings=None,
    out_shardings=state_mesh_shardings,
)(rng)
print("\n2", state)

train_state = max_utils.unbox_logicallypartioned(state)
print("\n3", train_state)
# dict_keys(['step', 'apply_fn', 'params', 'tx', 'opt_state'])
print(train_state.__dict__.keys())

data_sharding = maxtext_utils.get_input_data_sharding(config, mesh)

from MaxText.train import loss_fn
from MaxText.train_utils import jit_train_step


def train_step(model, config, state_mesh_shardings, state, data, dropout_rng):
  _loss_fn = loss_fn
  extra_dpo_args = []
  grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
  (loss, aux), raw_grads = grad_func(model, config, data, dropout_rng, state.params, *extra_dpo_args, is_train=True)
  raw_grads = jax.tree_util.tree_map(lambda x: x.astype(config.grad_dtype) if x.dtype == jnp.float32 else x, raw_grads)
  grads = raw_grads
  new_state = state.apply_gradients(grads=grads)
  metrics = None
  return new_state, metrics


from MaxText.data_loader import DataLoader

from MaxText.input_pipeline.synthetic_data_processing import SyntheticDataIterator

data_iterator = config, mesh

p_train_step = jit_train_step(config, model, state, state_mesh_shardings, data_sharding, train_step)


data_loader = DataLoader(config, mesh, data_iterator, None)
init_rng = rng
for step in range(10):
  nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
  example_batch = data_loader.load_next_batch()
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state, metrics = p_train_step(state, example_batch, nextrng)

# """
# start opt
# """
# def f(model_vars):
#   with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
#     logits = model.apply(
#         model_vars,
#         decoder_input_tokens=lnx,
#         decoder_positions=decoder_positions,
#         decoder_segment_ids=decoder_segment_ids,
#         # train mode
#         model_mode=MODEL_MODE_TRAIN,
#         # deterministic
#         enable_dropout=False,
#         rngs={"dropout": rng, "params": rng},
#     )
#   return jnp.sum(logits)


# opt_state = optimizer.init(model_vars)
# print(opt_state)


# for step in range(5):
#   print(step)
#   grad = jax.grad(f)(model_vars)
#   updates, opt_state = optimizer.update(grad, opt_state, model_vars)
