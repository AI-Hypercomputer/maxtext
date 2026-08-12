import sys
import jax
from jax.sharding import NamedSharding
from flax import nnx
from maxtext import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.trainers.pre_train import train_compile

argv = list(sys.argv)
if not any("compile_topology=" in arg for arg in argv):
  argv.append("compile_topology=v6e-256")
if not any("compile_topology_num_slices=" in arg for arg in argv):
  argv.append("compile_topology_num_slices=1")
if not any("per_device_batch_size=" in arg for arg in argv):
  argv.append("per_device_batch_size=1")
if not any("skip_jax_distributed_system" in arg for arg in argv):
  argv.append("skip_jax_distributed_system=True")
config = pyconfig.initialize(argv)
topology_mesh = train_compile.get_topology_mesh(config)
shaped_train_args, shaped_train_kwargs, state_mesh_shardings, logical_annotations, model = train_compile.get_shaped_inputs(topology_mesh, config)

params_shardings, state_mesh_shardings = maxtext_utils.sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
input_state_mesh_shardings = maxtext_utils.sharding.build_zero1_input_state_mesh_shardings(config, state_mesh_shardings, params_shardings)
data_sharding = maxtext_utils.sharding.get_input_data_sharding(config, topology_mesh)

abstract_state = shaped_train_args[0]

sh_struct = jax.tree_util.tree_structure(input_state_mesh_shardings)
abs_struct = jax.tree_util.tree_structure(abstract_state)

print(f"abstract_state vs input_state_mesh_shardings structure match: {sh_struct == abs_struct}")

def get_meta(v):
  val = getattr(v, "value", v)
  sh = getattr(val, "sharding", val if isinstance(val, (jax.sharding.Sharding, NamedSharding)) else "<NO_SHARDING>")
  var_type = type(v)
  return (var_type, sh)

abs_meta = jax.tree_util.tree_map(
    get_meta,
    abstract_state,
    is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.ShapeDtypeStruct, jax.sharding.Sharding)),
)
in_meta = jax.tree_util.tree_map(
    get_meta,
    input_state_mesh_shardings,
    is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.ShapeDtypeStruct, jax.sharding.Sharding)),
)

abs_m_leaves = list(jax.tree_util.tree_leaves_with_path(abs_meta))
in_m_leaves = list(jax.tree_util.tree_leaves_with_path(in_meta))

mismatches = 0
for (p1, m1), (p2, m2) in zip(abs_m_leaves, in_m_leaves):
  if m1 != m2:
    mismatches += 1
    if mismatches <= 5:
      print(f"META MISMATCH #{mismatches}: path={p1}\n  abs={m1}\n  in ={m2}")

print(f"Total metadata mismatches: {mismatches} (out of {len(abs_m_leaves)})")



