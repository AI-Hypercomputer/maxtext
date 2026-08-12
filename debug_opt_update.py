import os
import sys
import jax
import jax.numpy as jnp
from flax import nnx
from maxtext import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.trainers.pre_train import train_compile

config = pyconfig.initialize(sys.argv)
topology_mesh = train_compile.get_topology_mesh(config)
shaped_train_args, shaped_train_kwargs, state_mesh_shardings, logical_annotations, model = train_compile.get_shaped_inputs(topology_mesh, config)

params_shardings, state_mesh_shardings = maxtext_utils.sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
input_state_mesh_shardings = maxtext_utils.sharding.build_zero1_input_state_mesh_shardings(config, state_mesh_shardings, params_shardings)
data_sharding = maxtext_utils.sharding.get_input_data_sharding(config, topology_mesh)

abstract_state = shaped_train_args[0]
state = nnx.merge(model, abstract_state)

print("state.model type:", type(state.model))
print("state.optimizer type:", type(state.optimizer))

model_params = nnx.state(state.model)
print("Total model_params leaves:", len(jax.tree_util.tree_leaves(model_params)))

for p, val in jax.tree_util.tree_leaves_with_path(model_params):
  p_str = "/".join(str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k) for k in p)
  val_arr = getattr(val, "value", val)
  sh = getattr(val_arr, "sharding", None)
  mk = getattr(sh, "memory_kind", None)
  if "pre_alpha_scale" in p_str or "A_log" in p_str:
    print(f"Leaf {p_str}: shape={getattr(val_arr, 'shape', None)}, dtype={getattr(val_arr, 'dtype', None)}, mk={mk}, sharding={sh}")
