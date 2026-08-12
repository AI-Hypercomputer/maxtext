import os
import sys
import jax
import jax.numpy as jnp
from flax import nnx
from maxtext import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.trainers.pre_train import train_compile
from maxtext.trainers.pre_train import train

config = pyconfig.initialize(sys.argv)
topology_mesh = train_compile.get_topology_mesh(config)
shaped_train_args, shaped_train_kwargs, state_mesh_shardings, logical_annotations, model = train_compile.get_shaped_inputs(topology_mesh, config)

params_shardings, state_mesh_shardings = maxtext_utils.sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
input_state_mesh_shardings = maxtext_utils.sharding.build_zero1_input_state_mesh_shardings(config, state_mesh_shardings, params_shardings)
data_sharding = maxtext_utils.sharding.get_input_data_sharding(config, topology_mesh)

(
    func_to_compile,
    in_shard,
    out_shard,
    static_argnums,
    donate_argnums,
) = maxtext_utils.get_functional_train_with_signature(
    train.train_step,
    data_sharding,
    input_state_mesh_shardings,
    model,
    config,
    params_shardings,
)

def test_func(state, batch):
  model_params = nnx.state(state.model)
  print("[INSIDE TRACE] Initial model_params A_log:", flush=True)
  for p, v in jax.tree_util.tree_leaves_with_path(model_params):
    p_str = "/".join(str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k) for k in p)
    if "A_log" in p_str and "scanned_blocks" in p_str:
      print(f"  {p_str}: aval={getattr(v, 'value', v).aval if hasattr(getattr(v, 'value', v), 'aval') else getattr(v, 'value', v)}", flush=True)

  # Run staging
  def _is_tree_leaf(x):
    return isinstance(x, (jax.sharding.Sharding, jax.sharding.PartitionSpec, nnx.Variable, jax.ShapeDtypeStruct)) or (
        hasattr(x, "shape") and hasattr(x, "dtype")
    )
  def _normalize_path(path):
    return tuple(
        str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k)
        for k in path
        if str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k) not in ["model", "optimizer", "opt_state", "value"]
    )
  sh_leaves = list(jax.tree_util.tree_leaves_with_path(state_mesh_shardings, is_leaf=_is_tree_leaf))
  sh_lookup = {_normalize_path(p): (l.value if hasattr(l, "value") else l) for p, l in sh_leaves}
  def _get_sharding(x):
    val = getattr(x, "value", x)
    if hasattr(val, "sharding") and val.sharding is not None:
      return val.sharding
    if hasattr(x, "sharding") and x.sharding is not None:
      return x.sharding
    return None
  def _get_param_sharding(path, val):
    s = _get_sharding(val)
    if s is not None and hasattr(s, "with_memory_kind"):
      return s
    norm_p = _normalize_path(path)
    if norm_p in sh_lookup:
      lookup_s = sh_lookup[norm_p]
      if hasattr(lookup_s, "with_memory_kind"):
        return lookup_s
    return None
  def _stage_to_device(path, val):
    if not maxtext_utils.sharding.is_scanned_block_param_path(path):
      return val
    val_arr = getattr(val, "value", val)
    s = _get_param_sharding(path, val)
    print(f"  [STAGE DEBUG] path={path}, s={s}", flush=True)
    if s is not None and hasattr(s, "with_memory_kind"):
      target_s = s.with_memory_kind("device")
      new_val = jax.lax.with_sharding_constraint(val_arr, target_s)
    else:
      new_val = val_arr
    if isinstance(val, nnx.Variable):
      return val.replace(value=new_val)
    return new_val

  dev_model_params = jax.tree_util.tree_map_with_path(
      _stage_to_device, model_params, is_leaf=_is_tree_leaf
  )
  nnx.update(state.model, dev_model_params)

  updated_params = nnx.state(state.model, nnx.Param)
  print("[INSIDE TRACE] After update model_params A_log:", flush=True)
  for p, v in jax.tree_util.tree_leaves_with_path(updated_params):
    p_str = "/".join(str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k) for k in p)
    if "A_log" in p_str and "scanned_blocks" in p_str:
      print(f"  {p_str}: aval={getattr(v, 'value', v).aval if hasattr(getattr(v, 'value', v), 'aval') else getattr(v, 'value', v)}", flush=True)

  pure_params = nnx.as_pure(nnx.state(state.model, nnx.Param))
  for p, v in jax.tree_util.tree_leaves_with_path(pure_params):
    p_str = "/".join(str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k) for k in p)
    if "A_log" in p_str and "scanned_blocks" in p_str:
      print(f"  [AS_PURE] {p_str}: aval={v.aval if hasattr(v, 'aval') else v}", flush=True)

  return state

jitted = jax.jit(test_func, in_shardings=in_shard, out_shardings=in_shard[0])
lowered = jitted.lower(*shaped_train_args)
print("SUCCESSFULLY LOWERED TEST FUNC!")
