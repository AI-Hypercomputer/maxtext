"""
cd ~/maxtext
python -m MaxText.muon_test
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
maxtext_model = Transformer(config, mesh=mesh, quant=quant)




"""
get data
"""
data = DataIter(rng, config.max_target_length, config.global_batch_size_to_train_on, config.base_emb_dim)
lnx, decoder_segment_ids, decoder_positions = data.get_data()
"""
init model
"""


maxtext_state, _ = maxtext_utils.setup_decode_state(maxtext_model, config, rng, mesh, None)
print(maxtext_state)
breakpoint()
#sys.exit(1)
model_vars = maxtext_state.params
print(model_vars)
breakpoint()


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
abstract_param = get_abstract_param(maxtext_model, config)
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
optimizer = muon.muon(**muon_kwargs)



"""
start opt
"""
def f(model_vars):
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    logits = maxtext_model.apply(
        model_vars,
        decoder_input_tokens=lnx,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        # train mode
        model_mode=MODEL_MODE_TRAIN,
        # deterministic
        enable_dropout=False,
        rngs={"dropout": rng, "params": rng},
    )
  return jnp.sum(logits)


opt_state = optimizer.init(model_vars)
print(opt_state)


for step in range(5):
  print(step)
  grad = jax.grad(f)(model_vars)
  updates, opt_state = optimizer.update(grad, opt_state, model_vars)  

